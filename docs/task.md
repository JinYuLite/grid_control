
## 任务记录

### 9.5
- [x] 利用SAC算法训练一个agent，并且提交到在线系统;
    - 结果：线上15.3， 线下20.0

### 9.12     
- [x] 整理目前使用的动作空间和状态空间
    - [文档可见](https://github.com/mikezhang95/grid_control/blob/main/docs/env.md#%E5%90%91%E9%87%8F%E5%8C%96observation)
    - 动作空间对于结果更加敏感
    - 现在的电压动作只能增加，是不合理的

- [x] 尝试Imitation Learning
    - 从表格读取第一个时刻的gen_p，作为label
    - gen_v随机生成，作为label
    - 结果：效果无提升
    - 原因：只有第一时刻的数据，样本数量、多样性不足
    
### 9.19

- [x] 参加赛道研讨会

- [x] 合理化Action Space
    - action只训练gen_p, fix gen_v=0
    - 结果：效果提升非常明显，线上从15 -> 42
    - gen_p scale扩大10倍
    - 结果：效果提升非常明显，线上从42 -> 207

### 9.30
- [ ] 增加规则，避免违反电网规则 / 初始化选择断面

- [ ] RL算法优化，包括调参、使用其他算法、并行计算
    * FrameStack
    * 多进程

- [ ] Observation Space优化，包括特征归一化、特征选择、Graph Learning

- [ ] Action Space优化，包括在训练gen_p的模型上微调gen_v

- [ ] Reward Space优化，包括归一化，现在Scale太大

- [ ] 环境增加噪声，提高鲁棒性



## 实验结果

| model_name    | 训练结果 | 线上结果 | 
|---------------|---------|----------|
|  sac_trial    |  20.0   |  15.3    |
|  sac_genp_t1  |  207.2  |  42.3    |
|  sac_genp_t10 |  1613.8 |  206.6   | 

[【腾讯文档】电网比赛实验结果](https://docs.qq.com/sheet/DYWNnSXJNSlVhS09G) 

## QAs:

1. 潮流前，潮流后什么意思
A: PowerFlow计算收敛之前和之后的状态

2. grid_loss 是怎么计算的，每一步只有一个值？
A: 传输线路上的耗损

3. 断面
A: == step

4. forecast reader给了后一时刻的信息
A: 通过预测下一时刻的信息更好的控制机器，之类直接给出下一时刻的真实负载和新能源发电信息

5. legal act space 是怎么计算出来的
A: 通过爬坡功率和发电机的最大最小功率一起计算得到
