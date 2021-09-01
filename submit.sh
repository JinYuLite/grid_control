

# copy agent file
ln -s src/stable_baselines3 submission
ln -s src/utils.py submission
ln -s src/Agent/sac_agent.py submission/agent.py
cp src/sac_trial.zip submission/model.zip

# zip submission 
zip -r submission.zip submission

# upload submission.zip on the website

