使用**Ubuntu**的过程中，无论用来干什么，都会有文件上的交流，必不可免的就是压缩文件，Ubuntu系统中自带了部分格式的压缩软件，但是win系统习惯的rar格式文件解压需要下载相关软件，现整理如下：

## 1.文件格式及解压工具

1. `*.tar` 用 `tar` 工具
2. `*.gz` 用 `gzip` 或者 `gunzip` 工具
3. `*.tar.Z`，`*.tar.bz2`，`*.tar.gz` 和 `*.tgz` 用 tar 工具
4. `*.bz2` 用 `bzip2` 或者用 `bunzip2` 工具
5. `*.Z` 用 `uncompress` 工具
6. `*.rar` 用 `unrar` 工具
7. `*.zip` 用 `unzip` 工具

## 2.具体使用简介

filename，表示文件名

dirname，表示路径地址

`.tar` 文件

功能： 对文件目录进行**打包备份（仅打包并非压缩）**

```shell
tar -xvf filename.tar         # 解包
tar -cvf filename.tar dirname # 将dirname和其下所有文件（夹）打包
```

```shell
tar命令有5个常用的选项：

-c 建立新的归档文件
-r 向归档文件末尾追加文件
-x 从归档文件中解出文件
可以这样记忆，创建新的文件是c，追加在原有文件上用r，从文件中解压出用x

-O 将文件解开到标准输出
-v 处理过程中输出相关信息
-f 对普通文件操作           －－－似乎一直都要用f，不然的话，可能会不显示
-z 调用gzip来压缩归档文件，与-x联用时调用gzip完成解压缩
-Z 调用compress来压缩归档文件，与-x联用时调用compress完成解压缩

-t ：查看 tarfile 里面的文件！
特别注意，在参数的下达中， c/x/t 仅能存在一个！不可同时存在！

-p ：使用原文件的原来属性（属性不会依据使用者而变）
-P ：可以使用绝对路径来压缩！
-N ：比后面接的日期(yyyy/mm/dd)还要新的才会被打包进新建的文件中！
--exclude FILE：在压缩的过程中，不要将 FILE 打包！
```



`.gz` 文件

```shell
gunzip filename.gz            # 解压1
gzip -d filename.gz           # 解压2
gzip filename                 # 压缩，只能压缩文件
```



`.tar.gz`文件、 `.tgz`文件

```shell
tar -zxvf filename.tar.gz               # 解压
tar -zcvf filename.tar.gz dirname       # 将dirname和其下所有文件（夹）压缩
tar -C dirname -zxvf filename.tar.gz    # 解压到目标路径dirname
```



`.bz2`文件

```shell
bzip2 -zk filename                      #将filename文件进行压缩
bunzip2 filename.bz2                    #解压
bzip2 -d filename.bz2                   #解压
```



`.tar.bz2`文件

```shell
tar -jxvf filename.tar.bz               #解压
```



`.Z`文件

```shell
uncompress filename.Z                   #解压
compress filename                       #压缩
```



`.tar.Z` 文件

```text
tar -Zxvf filename.tar.Z                #解压
tar -Zcvf fflename.tar.Z dirname        #压缩
```



`.rar` 文件

```shell
rar x filename.rar                      #解压
rar a filename.rar dirname              #压缩
```



`.zip`文件

```shell
unzip -O cp936 filename.zip            # 解压（不乱码）
zip filename.zip dirname               # 将dirname本身压缩
zip -r filename.zip dirname            # 压缩，递归处理，将指定目录下的所有文件和子目录一并压缩
```

使用过程中如提示以下问题，安装相应的工具即可。

![img](https://pic4.zhimg.com/80/v2-5055592423e61ca6ace8596d859b1a83_720w.png)

安装方式：

```shell
sudo apt install ***                  
#例如：上述问题解决方式为sudo apt install bunzip
```

