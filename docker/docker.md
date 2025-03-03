<center><h1>Docker</h1></center>

### 一、DockerFile

1. 



### 二、Docker命令

1. 查看所有镜像：`docker images`

2. 查看所有容器：`docker ps -a`

3. 拉取镜像：`docker pull [name]`

4. 新建、启动容器：`docker run [option] IMAGE [COMMAND] [ARG...]`

   ```
   [option]
   --name		给容器命名
   -it			交互式运行容器，分配一个伪终端
   -v			挂载卷，host_dir:container_dir
   -network	指定网络模式
   -d			后台运行容器
   ```

   例如：

   ```shell
   docker run -it -v $(pwd):/root cw-sim:sigcomm23ae bash -c "cd ns-3.19; ./waf configure --build-profile=optimized; ./waf"
   
   docker run -it -p 2222:22 -v $(pwd):/root --name ns3 my-ns3-ssh
   ```

   

5. 开启、关闭、挂载容器：`docker start/stop/attach`

6. 提交新的容器为镜像：`docker commit <容器ID> <my-image>`

7. 删除容器的时候可以只写前三个字符。

8. 可以在 `VSCode` 中直接 `attach` 到窗口，避免了 `ssh`。

   

9. （最后一行留空）