
import os, sys
import numpy as np
import torch 
from torch.utils.data.dataset import Dataset, random_split

from Environment.base_env import Environment
from env_wrapper import GridEnv
from utilize.settings import settings
from stable_baselines3 import PPO, DDPG, TD3, SAC, A2C

env = Environment(settings, "EPRIReward")
env = GridEnv(env)  


class ExpertDataSet(Dataset):
    def __init__(self, data_path):

        # read from file
        with open(data_path, "r") as rf:
            lines = rf.readlines()

        expert_observations, expert_actions = [], []
        for line in lines:
            obs, action = line.split("\t")
            expert_observations.append(obs.split(" "))
            expert_actions.append(action.split(" "))

        # convert to torch
        self.observations = torch.tensor(expert_observations, dtype=torch.double)
        self.actions = torch.tensor(expert_actions, dtype=torch.double)

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

def train(model, device, train_loader, optimizer, criterion, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if isinstance(env.action_space, gym.spaces.Box):
            # A2C/PPO policy outputs actions, values, log_prob
            if isinstance(student, (A2C, PPO)):
                action, _, _ = model(data)
            # SAC/TD3 policy outputs actions only
            else:
                action = model(data)
            action_prediction = action.double()
        else:
            # Retrieve the logits for A2C/PPO when using discrete actions
            latent_pi, _, _ = model._get_latent(data)
            logits = model.action_net(latent_pi)
            action_prediction = logits
            target = target.long()

        loss = criterion(action_prediction, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if isinstance(env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                # SAC/TD3 policy outputs actions only
                else:
                    action = model(data)
                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                latent_pi, _, _ = model._get_latent(data)
                logits = model.action_net(latent_pi)
                action_prediction = logits
                target = target.long()

            test_loss = criterion(action_prediction, target)
    test_loss /= len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}")


def pretrain_agent(
    student, train_expert_dataset, test_expert_dataset
    batch_size=64,
    test_batch_size=64,
    epochs=1000,
    learning_rate=1e-2,
    log_interval=100,
    no_cuda=True,
    seed=1,
):
    torch.manual_seed(seed)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # continuous action space
    if isinstance(env.action_space, gym.spaces.Box):
        criterion = torch.nn.MSELoss()
    # discrete action space
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=False
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = torch.optim.RMSProp(model.parameters(), lr=learning_rate)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, log_interval):
        test(model, device, test_loader, criterion)

    # end training
    return


if __name__ == "__main__":

    # create sac student 
    policy_kwargs = dict(activation_fn=torch.nn.Tanh) # output: [-1,1]
    sac_student = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs, 
    )

    # load train/test expert dataset
    expert_dataset = ExpertDataSet("./expert_data.pkl")
    train_size = int(0.8 * len(expert_dataset))
    torch.manual_seed(0)
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, len(expert_dataset)-train_size]
        )

    # start behavior cloing
    pretrain_agent(
        sac_student, train_expert_dataset, test_expert_dataset,
        epochs=3,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        batch_size=64,
        test_batch_size=1000,
        )
    
    # save student model
    sac_student.save("sac_student")
