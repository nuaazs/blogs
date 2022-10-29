# docker 常用命令大全

## 启动与关闭

```shell
# 启动docker
systemctl start docker

# 关闭docker
systemctl stop docker

# 重启docker
systemctl restart docker

# docker设置随服务启动而启动
systemctl enable docker

# 查看docker运行状态
systemctl status docker

# 查看docker版本号信息
docker version
docker info

# docker 帮助命令
docker --help
# 比如忘记了拉取命令，可以查看参数使用方法
docker pull --help
```



## 镜像命令

查看自己服务器中docker镜像列表

```shell
docker images

# 搜索镜像
docker search 镜像名
# 搜索STARS>9000的mysql镜像
docker search --filter=STARS=9000 mysql

# 拉取镜像不加tag即拉取docker仓库中该镜像的最新版本latest，加tag则拉取指定版本
docker pull 镜像名
docker pull 镜像名:tag

# 拉取最新版mysql
docker pull mysql

# 拉取指定版本
docker pull mysql:5.7.30

# 运行镜像
docker run 镜像名
docker run 镜像名:Tag

# 删除镜像，删除一个
docker rmi -f 镜像名/镜像ID

# 删除多个，其镜像ID或者镜像名用空格隔开
docker rmi -f 镜像名/镜像ID 镜像名/镜像ID 镜像名/镜像ID 镜像名/镜像ID

# 删除全部镜像 -a 意为显示全部， -q 意思为只现实ID
docker rmi -f $(docker images -aq)

# 强制删除镜像
docker image rm 镜像名/镜像ID

# 保存镜像
# 将我们的镜像保存为tar压缩文件这样方便镜像转移和保存
# 然后可以在任何一台安装了docker的服务器上加载这个镜像
docker save 镜像名/镜像ID -O 镜像保存在哪个位置与名字
docker save tomcat -o /myimg.tar

# 加载镜像
docker load -i 镜像保存的位置

# 加载文件恢复为镜像
docker load -i myimg.tar

# 镜像标签
app:1.0.0 基础镜像
# 分离为开发环境
app:develop-1.0.0
# 分离为alpha环境
app:alpha-1.0.0

docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
docker tag 源镜像名:TAG 想要生成的镜像名:TAG
# 如果省略TAG，则会为镜像默认打上latest TAG
docker tag aaa bbb
# 上面的操作等于 docker tag aaa:latest bbb:latest
# 我们根据镜像quay.io/minio/minio添加一个新的镜像，名为aaa 标签Tag设置为1.2.3
docker tag quay.io/minio/minoi:1.2.3 aaa:1.2.3

# 我们根据镜像app-user:1.0.0 添加一个新的镜像名为app-user标签Tag设置为alpha-1.0.0
docker tag app-user:1.0.0 app-user:alpha-1.0.0
```



## 容器命令

```shell
# 正在运行容器列表
docker ps

# 查看所有容器，包括正在运行和已经停止的
docker ps -a

# 运行一个容器
# -it表示与容器进行交互式启动
# -d 表示可后台运行容器（守护式运行）
# --name 给要运行的容器起的名字
# /bin/bash 交互路径
# -p 映射端口
# -v 宿主机文件存储位置:容器内文件位置 映射目录
# --restart=always表示该容器随docker服务启动而自动启动
docker run -it -d --name 要取的别名 镜像名:Tag /bin/bash

# 拉取并启动
docker pull redis:5.0.5
docker run -it -d --name redis001 redis:5.0.5 /bin/bash

# 停止容器
docker stop 容器名/容器ID
# 启动容器
docker start 容器名/容器ID
# kill容器
docker kill 容器名/容器ID

# 容器文件拷贝:无论是否开启容器，都可以进行拷贝
docker cp 容器名/容器ID:容器内路径 容器外路径
docker cp 容器外路径 容器名/容器ID：容器内路径

# 查看容器日志
docker logs -f --tail=要查看末尾多少行 容器ID


# docker 启动所有的容器
docker start $(docker ps -a | awk '{ print $1}' | tail -n +2)

# docker 关闭所有的容器
docker stop $(docker ps -a | awk '{ print $1}' | tail -n +2)

# docker 删除所有的容器
docker rm $(docker ps -a | awk '{ print $1}' | tail -n +2)

# docker 删除所有的镜像
docker rmi $(docker images | awk '{print $3}' |tail -n +2)
```



## 修改启动配置

```shell
docker  update --restart=always 容器Id 或者 容器名

或

docker container update --restart=always 容器Id 或者 容器名
```



## 更换容器名

```shell
docker rename 容器名/容器ID 新容器名
```



## 自己提交一个镜像

我们运行的容器可能在镜像的基础上做了一些修改，有时候我们希望保存起来，封装成一个更新的镜像，这时候我们就需要使用 commit 命令来构建一个新的镜像。

```shell
docker commit -m="提交信息" -a="作者信息" 容器名/容器ID 提交后的镜像名:TAG
```

我们拉取一个tomcat镜像，并持久化运行，且设置与宿主机进行端口映射

```shell
docker pull tomcat
docker run -itd -p 8080:8080 --name tom tomcat /bin/bash
```

修改后提交

```shell
docker commit -a="leilei" -m="第一次打包镜像，打包后直接访问还会404吗" 231f2eae6896 tom:1.0
```



## 运维命令

可能有时候发布会遇到如下错误:

```
docker: write /var/lib/docker/tmp/GetImageBlob325372670: no space left on device
```

这个错误是docker在写入的时候报错无机器无空间。

查看docker工作目录

```shell
sudo docker info | grep "Docker Root Dir"
```

查看docker磁盘占用总体情况

```shell
du -hs /var/lib/docker
```

查看磁盘使用具体情况

```shell
docker system df
```

删除无用的容器和镜像

```shell
# 删除异常停止的容器
docker rm `docker ps -a | grep Exited |awk '{print $1}'`

# 删除名称或者标签为none的镜像
docker rmi -f `docker images | grep '<none>' | awk '{print $3}'`
```

清楚所有无容器使用的镜像

注意，此命令只要是镜像无容器使用（容器正常运行）都会被删除，包括容器临时停止。

```shell
docker system prune -a
```

查找大文件

```shell
find / -type f -size +100M -print0 | xargs -0 du -h | sort -nr
```

查找指定docker使用目录下大雨指定大小的文件

```shell
find // -type f -size +100M -print0 | xargs -0 du -h | sort -nr | grep '/var/lib/docker/overlay2/*'
```

ex：我这里是查找 /var/lib/docker/overlay2/* 开头的且大于100m的文件