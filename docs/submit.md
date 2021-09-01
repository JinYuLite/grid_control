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



