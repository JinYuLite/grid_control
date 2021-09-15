
# 任务总结

## 进度更新

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
    - 结果：效果不好只用第一时刻的数据学不出来 等有了模型再尝试 

### 9.19

- [x] 参加赛道研讨会

- [ ] 使用优化方法计算PowerFlow，和RL策略结合；
    - 结合方式包括Imitation Learning，Rule-based混合

    
- [ ] Play with Observation/Action/Reward Space；
    - 特征归一化
    - FrameStack：缓解partially observed的问题
    - 表示学习：Embedding和Graph Learning
    - 奖励归一化Scale
    - 动作噪声/离散化

## 实验结果
| model_name | 训练结果 | 线上结果 | 
|------------|---------|---------|
|  sac_trial |   20.0  | 15.3    |
 
## QAs:

1. 潮流前，潮流后什么意思
    > PowerFlow计算收敛之前和之后的状态

2. grid_loss 是怎么计算的，每一步只有一个值？
    > 传输线路上的耗损

3. 断面
    > = step

4. forecast reader给了后一时刻的信息
    > 通过预测下一时刻的信息更好的控制机器，之类直接给出下一时刻的真实负载和新能源发电信息。

5. legal act space 是怎么计算出来的

6. 如何满足平衡机上下限约束

