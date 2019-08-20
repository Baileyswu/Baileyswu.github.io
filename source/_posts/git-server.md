---
title: Git 服务器搭建
date: 2018-11-15 16:35:44
categories: 技术
tags:
 - Git
 - Centos
 - Linux
 - ssh
---

本博客是摘自 [runoob 教程](http://www.runoob.com/git/git-server.html)后的注解。经过亲测，可以加上几个小操作做到更 Easy...

## 材料
- 一台有公网 IP 的服务器（那么你在世界每个角落都可以访问）
- 或者一台有内网 IP 的服务器（那么你只有在内网或者通过 VPN 可以访问，就看能不能 ping 到）
- linux 系统的 root 权限

## 成品

- 一台提供Git服务的服务器。

## 步骤

接下来将以 Centos 为例搭建 Git 服务器。

### 安装Git
```bash
$ yum install curl-devel expat-devel gettext-devel openssl-devel zlib-devel perl-devel
$ yum install git
```
运气好的话，有人装好了你就不用再装了。

接下来我们 创建一个git用户组和用户，用来运行git服务：（如果你没有管理员权限的话，就不要再挣扎了）
```bash
$ groupadd git
$ useradd git -g git
```
为了方便后面的操作，你可以给这个 git 账户设置一个密码
```bash
$ passwd git
```

### 创建证书登录

查看/home/git/.ssh/authorized_keys文件，如果没有该文件则创建它：

```bash
$ cd /home/git/
$ mkdir .ssh
$ chmod 755 .ssh
$ touch .ssh/authorized_keys
$ chmod 644 .ssh/authorized_keys
```
收集所有需要登录的用户的公钥，公钥位于id_rsa.pub文件中，把我们的公钥导入到/home/git/.ssh/authorized_keys文件里，一行一个。或者使用

```bash
$ cat id_rsa.pub >> ~/.ssh/authorized_keys
```

如果觉得一个一个粘贴累，也可以在每个登陆用户的 .ssh/ 目录下直接复制公钥过去
```bash
ssh-copy-id git@xx.xx.xx.xx
```
等价于把你的公钥传递过去，但是也需要 git 的密码，就是刚才设置的那个。

### 初始化Git仓库

首先我们选定一个目录作为Git仓库，假定是/home/gitrepo/runoob.git，在/home/gitrepo目录下输入命令：
```bash
$ cd /home
$ mkdir gitrepo
$ chown git:git gitrepo/
$ cd gitrepo

$ git init --bare runoob.git
Initialized empty Git repository in /home/gitrepo/runoob.git/
```
以上命令 Git 创建一个空仓库，服务器上的 Git 仓库通常都以 .git 结尾。
注意：如果不使用`--bare`参数，初始化仓库后，提交master分支时报错。这是由于git默认拒绝了push操作，需要.git/config添加如下代码：

```bash
[receive]
denyCurrentBranch = ignore
```
推荐使用：`git --bare init`初始化仓库。
然后，把仓库所属用户改为 git ：

```bash
$ chown -R git:git runoob.git
```

### 克隆仓库

```bash
$ git clone git@xx.xx.xx.xx:/home/gitrepo/runoob.git
Cloning into 'runoob'...
warning: You appear to have cloned an empty repository.
Checking connectivity... done.
```

当然，你也可以不用命令行，我觉得 GitHub Desktop 挺好用的。不仅可以 clone github 上的 .git ，也可以 clone 你设置的服务器上的 .git ，只要在 clone a repository 的时候选择 URL，填对位置，如：git@xx.xx.xx.xx:/home/gitrepo/runoob.git 

### 想再创建仓库
```bash
ssh git@xx.xx.xx.xx
cd /home/gitrepo
git init --bare second.git
```

突发奇想，想本地直接 add 到服务器上失败了，用户权限不够
```bash
$ git remote add origin git@xx.xx.xx.xx:/home/gitrepo/gpdm-code.git
$ git push -u origin master
Warning: remote port forwarding failed for listen port 52698
fatal: '/home/repo/gpdm-code.git' does not appear to be a git repository
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.

```

可以去 [Git 官网](https://git-scm.com/book/zh/v2/%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8A%E7%9A%84-Git-%E5%8D%8F%E8%AE%AE) 学习一波.

## TODO
- 设置不同人的 git 账户
- 用户有权限上传 .git
- win10 主机作为 git server
