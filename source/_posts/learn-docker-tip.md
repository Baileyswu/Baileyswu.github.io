---
title: Docker 学习笔记
date: 2018-12-06 16:38:17
categories: 技术
tags:
 - Docker
author: Danliwoo
---

## 为什么要用 Docker

实验室每个人都有自己在跑的实验，各种包的配置版本可能会不一样。下包的时候可能需要管理员权限，也就是一不小心把公有的包都给重写了，给别人造成了麻烦。再者，在多台机器上跑一个代码，重搭环境实在太累人了，谁不想一键搞定所有的搭建工作呢。所以，就把这种脏活累活都交给 Docker 来做吧！

## Docker 架构

Docker 使用客户端-服务器 (C/S) 架构模式，使用远程API来管理和创建Docker容器。  
容器与镜像的关系类似于面向对象编程中的对象与类。  
Docker 容器（对象）通过 Docker 镜像（类）来创建。  

## 安装测试
```sh
> wget -qO- https://get.docker.com/ | sh
> sudo usermod -aG docker danliwoo
> service docker start
> # restart the shell!!!
> docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/

```

## 运行容器

> -t:在新容器内指定一个伪终端或终端。  
-i:允许你对容器内的标准输入 (STDIN) 进行交互。  
-d:容器开在后台。  
-P:将容器内部使用的网络端口映射到我们使用的主机上。  
-v:将本地的文件夹挂载到容器里。  
--name:给容器取个名字。

### 简单的虚拟机
```sh
> docker run -i -t ubuntu:15.10 /bin/bash
```
就像装了个 ubuntu:15.10 的虚拟机，可以在里面玩耍


```sh
> docker run -d ubuntu:15.10 /bin/sh -c "while true; do echo hello world; sleep 1; done"
088f1ec5c9988346ef06fb1a717ae65b1d751ea3649061095cb2cdd26d106f3e
```
查看日志等信息
```sh
> docker ps
一堆正在跑的 docker 的信息。
> docker ps -a
> docker logs [CONTAINER ID]
> docker logs [NAMES]
都可以输出当前的日志信息。
> docker stop xxx
```

### Web 容器
```sh
> docker pull training/webapp  
载入镜像
> docker run -d -P training/webapp python app.py
docker:5000 Host:32768
> docker run -d -p 2333:5000 training/webapp python app.py
docker:5000 Host:2333 
> docker run -d -p 127.0.0.1:5000:5000/udp training/webapp python app.py
绑定 UDP 端口
> docker logs -f xxx
看谁访问了我的网页
> docker top xxx
就跟 top 作用很像
> docker inspect
查看底层的配置和状态信息
```

### GPflow 容器
```sh
docker run -it -p 8888:8888 gpflow/gpflow
```

复活暂停的容器  
```sh
> docker container start [containerName]
```

想进入后台正在运行的程序
```sh
docker attach [containerName]
```

### 批量删除停止的容器

删除容器
```sh
> docker ps -a
CONTAINER ID        IMAGE                  COMMAND             CREATED             STATUS                     PORTS                                    NAMES
3ed73013d8ce        danliwoo/gpmed:part3   "/bin/bash"         About an hour ago   Exited (0) 6 seconds ago                                            frosty_shaw
2e1af5e8495b        danliwoo/gpmed:part3   "/bin/bash"         4 weeks ago         Up 2 days                  22/tcp, 80/tcp, 0.0.0.0:2333->8888/tcp   peaceful_keller
> docker rm 3e
3e
这里只需要敲容器 ID 的前两位，就可以删除指定容器（除非不唯一，则会提醒）。
```

容器很多，一个一个删也太费事了
```sh
> docker container prune
WARNING! This will remove all stopped containers.
Are you sure you want to continue? [y/N] y
Deleted Containers:
61e0c59a43b75a76f04ca90905b6e9b02caf67d13333c9a0f0e487a6077810b5
b7212cb2065a3cb2816f9efe07a0bc8203a1e44c916cb4385ae477c9f8d8d127
21d69920d8c9b02b34e51c0e01347f0d411046668e5d2698218df857a3ce14f0
0d4c407b221ef99e7b623186593255744aa157f2db1e51975b0d647497c62960
49a9fe9a9cf45fa2983cf51ed41497c5b14a1052f76fab3de0441f4a57660159

Total reclaimed space: 1.306kB

```

### 挂载系统文件到容器中
```sh
> docker run -v [本地目录]:[容器目录] [镜像]
> docker run -it -p 17813:8888 -p 6006:6006 -v ~/tensorflow-tutorial:/app danliwoo/gpmed:part3
```

## 镜像管理
```sh
> docker images
列出了所有本机下载到的镜像。
> docker search xxx
搜出了一堆可用镜像
> docker pull xxx
> docker run xxx
> docker commit -m="msg" -a="author" [container] [reponame]:[tag]
```

构建新镜像
[Dockerfile](Dockerfile)
```sh
> docker build -t [imagename]:[tag] [dir]
dir 是 Dockerfile 所在的目录。
```

Docker 允许上传镜像，每个镜像相当于是一个 repository，在这之前需要注册和登录账号。
```sh
> docker login
输入用户名和密码（如果没有需要先在网页上注册）
> docker push [username]/[repository]:[tag]
然后可以把新的镜像 push 到远端仓库中。
```

## 服务

一份镜像在产生容器时，将其限制的内存空间、要产生的容器数量等信息配置在 `docker-compose.yml` 中。

```sh
> docker stack deploy -c docker-compose.yml getstartedlab   
```

参考网站  

- [Runoob](http://www.runoob.com/docker/docker-tutorial.html)  
- [官方文档](https://docs.docker.com/get-started/)
