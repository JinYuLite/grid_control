
# 提交模型
> 训练好模型后提交的线上平台    


## 打包提交文件
1. 将Agent主文件和模型拷贝到`submission`目录下       
    ```bash
        ln -s ../src/Agent/sac_agent.py agent.py
        cp ../src/sac_trial.zip model.zip
    ```
！注意`agent.py`必须包含一个名为`Agent`的类。`Agent`类`__init__`函数输入参数为`settings`和`this_directory_path`，以及`act`函数输入参数为`obs`、`reward`和`done`。

2. 拷贝依赖文件，将`agent.py`依赖的文件拷贝到`submission`目录下，一般只需要创建软连接即可，例如
    ```bash
        ln -s ../src/stable_baselines3 .
    ```

3. 压缩`submission`文件夹
    ```bash
         zip -r submission.zip submission
    ```
    
4. 上传压缩文件



## 评估镜像线下测试流程

1. 下载docker镜像：grid_competition_test_latest.tar.gz​

2. 在终端，执行下面命令导入docker镜像（grid_competition_test:latest） 

   ```shell
   docker load < grid_competition_test_latest.tar.gz 
   ```

3. 在你的提交zip压缩文件所在目录运行下面命令进入镜像

   ```shell
   docker run -it -v $PWD:/submit -w /evaluate grid_competition_test:latest bash 
   ```

4. 在镜像内，输入下面命令，测试你的提交文件（例如：YourSubmission.zip）： 

   ```shell
   python test.py /submit/YourSubmission.zip
   ```



