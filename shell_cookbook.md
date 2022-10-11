## 0. shell基础

### 0.1 循环

#### 类c语言

```shell
for ((i=1; i<=100; i ++))
do
	echo $i
done
```



#### in使用

```shell
for i in {1..100}
do
	echo $i
done
```



#### seq使用

```shell
seq - print a sequence of numbers
```

```shell
for i in `seq 1 100`
do
	echo $i
done
```



### shell脚本实现取当前时间

shell 实现获取当前时间，并进行格式转换的方法：

#### 原格式输出

2018年 09月 30日 星期日 15:55:15 CST

```
time1=$(date)
echo $time1
```

 

#### 时间串输出

20180930155515

```
1 #!bin/bash
2 time2=$(date "+%Y%m%d%H%M%S")
3 echo $time2
```

 

 3）2018-09-30 15:55:15

```
#!bin/bash
time3=$(date "+%Y-%m-%d %H:%M:%S")
echo $time3
```

2018.09.30

```
#!bin/bash
time4=$(date "+%Y.%m.%d")
echo $time4
```

#### 注意

1、date后面有一个空格，shell对空格要求严格

2、变量赋值前后不要有空格

![img](https://img2018.cnblogs.com/blog/1487994/201809/1487994-20180930160052444-1960270008.png)

 

#### 解释

```
1 Y显示4位年份，如：2018；y显示2位年份，如：18。
2 m表示月份；M表示分钟。
3 d表示天；D则表示当前日期，如：1/18/18(也就是2018.1.18)。
4 H表示小时，而h显示月份。
5 s显示当前秒钟，单位为毫秒；S显示当前秒钟，单位为秒。
```

 

## 1. 批量替换文件内容

shell编程中替换文件中的内容用到四个命`sed`，`find` ，`grep`，`awk`
下面是三种使用替换的方法

### 方法一

```shell
find -name '要查找的文件名' | xargs perl -pi -e 's|被替换的字符串|替换后的字符串|g'
```

下面这个例子就是将当前目录及所有子目录下的所有`pom.xml`文件中的`http://repo1.maven.org/maven2` 替换为`http://localhost:8081/nexus/content/groups/public`.

```shell
find -name 'pom.xml' | xargs perl -pi -e 's|http://repo1.maven.org/maven2|http://localhost:8081/nexus/content/groups/public|g'
```



这里用到了Perl语言, `perl -pi -e` 在Perl 命令中加上`-e` 选项，后跟一行代码，那它就会像运行一个普通的Perl 脚本那样运行该代码.
从命令行中使用Perl 能够帮助实现一些强大的、实时的转换。认真研究正则表达式，并正确地使用，将会为您省去大量的手工编辑工作。

```shell
find -name 'pom.xml' | xargs perl -pi -e 's|http://repo1.maven.org/maven2|http://localhost:8081/nexus/content/groups/public|g'
```



### 方法二

Linux下批量替换多个文件中的字符串的简单方法。用sed命令可以批量替换多个文件中的字符串。

用sed命令可以批量替换多个文件中的 字符串。

```shell
sed -i "s/原字符串/新字符串/g" `grep 原字符串 -rl 所在目录`
```


例如：我要把`mahuinan`替换 为`huinanma`，执行命令：

```shell
sed -i "s/mahuinan/huinanma/g" 'grep mahuinan -rl /www'
```

这是目前linux最简单的批量替换字符串命令了！
具体格式如下：

```shell
sed -i "s/oldString/newString/g" `grep oldString -rl /path`
```


实例代码：

```shell
sed -i "s/大小多少/日月水火/g" `grep 大小多少 -rl /usr/aa`
sed -i "s/大小多少/日月水火/g" `grep 大小多少 -rl ./`
```



### 方法三

在日程的开发过程中，可能大家会遇到将某个变量名修改为另一个变量名的情况，如果这个变量是一个**局部变量**的话，`vi`足以胜任，但是如果是某个全局变量的话，并且在很多文件中进行了使用，这个时候使用`vi`就是 一个不明智的选择。这里给出一个简单的shell命令，可以一次性将所有文件中的指定字符串进行修改：

```shell
grep "abc" * -R | awk -F: '{print $1}' | sort | uniq | xargs sed -i 's/abc/abcde/g'
```

批量替换 配置文件中的IP：

```shell
grep "[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}" * -R | awk -F: '{print $1}' | sort | uniq | xargs sed -i 's/[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}/172\.0\.0\.1/g'
```





## 2. 批量替换文件名

批量改名，删除文件名中多余字符。目录下文件名为如下，要求去掉`_finished`。

```shell
stu_102999_1_finished.jpg

stu_102999_2_finished.jpg

stu_102999_3_finished.jpg

stu_102999_4_finished.jpg

stu_102999_5_finished.jpg
```

**可以实现的方法有很多种：**

**方法一：for循环结合sed替换**

```shell
for file in `ls *.jpg`;do mv $file `echo $file|sed 's/_finished//g'`;done;
```

 

**方法二：ls结合awk，输出交给bash执行**

```shell
ls *.jpg |awk -F "_finished" '{print "mv "$0" "$1$2""}'|bash
```

 

**实际执行的命令如下，以_finished作为分隔符，mv及变量 需要加双引号**

```shell
ls *.jpg |awk -F "_finished" '{print "mv "$0" "$1$2""}'
mv stu_102999_1_finished.jpg stu_102999_1.jpg
mv stu_102999_2_finished.jpg stu_102999_2.jpg
mv stu_102999_3_finished.jpg stu_102999_3.jpg
mv stu_102999_4_finished.jpg stu_102999_4.jpg
mv stu_102999_5_finished.jpg stu_102999_5.jpg
```



**方法三：rename改名**

```shell
rename "_finished" "" *.jpg
```



**方法四：for循环加变量部分截取**

```
for file in `ls *.jpg`;do mv $file `echo ${file%_finished*}.jpg`;done;
```



**不使用echo也可以实现**

```
for file in `ls *.jpg`;do mv $file ${file%_finished*}.jpg;done;
```

**更改后结果如下：**

stu_102999_1.jpg
stu_102999_2.jpg
stu_102999_3.jpg
stu_102999_4.jpg
stu_102999_5.jpg



## 3. 批量编号文件

顺序编号

```shell
#!/bin/bash
#rename.sh 
#批量编号
n=1 #number to start
for f in *
do
    if [ "${f}" = "rename.sh" ] || [ "${f}" = "resume.sh" ]
    then
        continue
    fi
    
    fname2ch=`(printf "%03d" ${n})`
    let n+=1
    mv "$f" "${fname2ch} $f"
done
```



恢复文件名

```shell
#!/bin/bash
#resume.sh 
#恢复文件名
 
for f in *
do
    if [ "${f}" = "rename.sh" ] || [ "${f}" = "resume.sh" ]
    then
        continue
    fi
    mv "$f" "${f:4}"
done
```





## 4. 批量运行Gate



## 5. alias命令参数

```shell
alias log='new() { /home/iint/zhaosheng/CT/log/"$1".txt ;}; new'
```



## 6. Shell脚本中循环语句for,while,until用法

### for

 for循环的运作方式，是讲串行的元素意义取出，依序放入指定的变量中，然后重复执行含括的命令区域（在do和done 之间），直到所有元素取尽为止。

 其中，串行是一些字符串的组合，彼此用$IFS所定义的分隔符（如空格符）隔开，这些字符串称为字段。

for的语法结构如下：

```shell
for 变量 in 串行
do
   执行命令
done
```

例1:用for循环在家目录下创建aaa1-aaa10，然后在aaa1-aaa10创建bbb1-bbb10的目录

```shell
#!/bin/bash
for a in {1..10}
do
    mkdir /datas/aaa$a
    cd /datas/aaa$a
    for b in {1..10}
    do
        mkdir bbb$b
    done
done
```

例2: 列出var目录下各子目录占用磁盘空间的大小。

```shell
#!/bin/bash
DIR="/var"
cd $DIR
for k in $(ls $DIR)
do
  [ -d $k ] && du -sh $k
done
```



### while

```shell
while 条件测试
do
    执行命令
done
```

例1:  while循环，经典的用法是搭配转向输入，读取文件的内容，做法如下

```shell
#!/bin/bash
while read a
do
    echo $a
done < /datas/6files
```

```shell
#!/bin/bash
while read kuangl
do
    echo ${kuangl}
done < /home/kuangl/scripts/testfile
```

 行2，使用read有标准输入读取数据，放入变量kuangl中，如果读到的数据非空，就进入循环。

 行4，把改行数据显示出来

 行5，将/home/kuangl/scripts/testfile的内容转向输入将给read读取。

例2:

```shell
#!/bin/bash
declare -i i=1
declare -i sum=0
while ((i<=10))
do
    let sum+=i
    let i++
done
echo $sum
```

```shell
#!/bin/bash
declare -i i=1
declare -i sum=0
while ((i<=10))
do
    let sum+=i
    let ++i
done
echo $sum
```

例3: while99乘法表

```shell
#!/bin/bash
a = 1
b = 1
while ((a<=9))
do
    while((b<=a))
    do
        let "c=a*b" #声明变量c
        echo -n "$a*$b=$c"
        let b++
    done
    let a++
    let b=1  # 因为每个乘法表都是1开始乘，所以b要重置
    echo ""  # 显示到屏幕换行
done
```



### until循环

while循环的条件测试是测真值，until循环则是测假值。

until循环的语法：

```shell
until 条件测试
do
    执行命令
done
```

说明：

 行1，如果条件测试结果为假（传回值不为0），就进入循环。

 行3，执行命令区域。这些命令中，应该有改变条件测试的命令，这样子，才有机会在有限步骤后结束执行until 循环（除非你想要执行无穷循环）。

 行4，回到行1，执行until命令。

例1

```shell
#!/bin/bash
declare -i i=10
declare -i sum=0
until ((i>10))
do
  let sum+=i
  let ++i
done
echo $sum
```

 行2-3，声明i和sum为整数型

 行4，如果条件测试：只要i值未超过10，就进入循环。

 行6，sum+=i和sum=sum+i是一样的，sum累加上i。

 行7，i的值递增1，此行是改变条件测试的命令，一旦i大于10，可终止循环。

 行8，遇到done，回到行6去执行条件测试

 行9，显示sum的值为10

例2: until99乘法表

```shell
#!/bin/bash
a = 1
b = 1
until ((a>9)) # until 和 while相反，条件为假的执行
do
    until((b>a))
    do
        let "c=a*b"
        echo -n "$a*$b=$c"
        let b++
    done
    let a++
    let b=1
    echo ""
done
```

行4，如果条件测试：只要a值未超过9，就进入循环，一旦超过9就不执行，until和while条件相反，条件真就done结束

行6，b>a,一旦b大于a就不执行了





## 7. shell无限循环

### for实现

```shell
#!/bin/bash
set i=0
set j=0
for((i=0;i<10;))
do
    let "j=j+1"
    echo "-------------j is $j -------------------"
done
```

### while实现

```shell
#!/bin/bash
set j=2
while true
do
    let "j=j+1"
    echo "----------j is $j--------------"
done
```

## 8. 在字符串中引用变量

用双引号

```shell
localhost:~ xxxx$ a='abc'
localhost:~ xxxx$ echo $a
abc

localhost:~ xxxx$ b='bbbb$a' #使用单引号 括起字符串，会原样输出变量名
localhost:~ xxxx$ echo $b
bbbb$a
l
localhost:~ xxxx$ b="bbbb$a"  #使用双引号 括起字符串，手会引用变量值
localhost:~ xxxx$ echo $b
bbbbabc
```



## 9. 输出输入重定向以及管道命令、标准输入标准输出标准错误

标准输入（stdin）-0、标准输出（stdout）-1、标准错误（stderr）-2

标准输入重定向：`cat < in.txt`

标准输出重定向：覆盖模式> ,追加模式>>

将标准输出重定向到待定文件`echo "123" > out.txt`

将标准错误重定向到待定文件`cmd 2>error.txt`

将标准输出和标准错误定向到不同文件 `cmd 1>out.txt 2>err.txt`

将标准输出和标准错误重定向到同一个文件 `cmd > out_err.txt 2>&1`

`/dev/null`是一个特殊的文件，写入到它的内容都会被丢弃；如果尝试从该文件读取内容，那么什么也读不到。但是`/dev/null`文件非常有用，将命令的输出重定向到它，会起到”禁止输出“的效果，如` $ command > /dev/null 2>&1`



## 10. bash的任务管理工具：&, ctrl-z, ctrl-c, jobs, fg, bg, kill

`&`表示把任务放在后台运行`python test.py > log.txt &`运行test.py程序，并置于后台运行，日志信息重定向到log.txt

`ctrl+c`是强制中断程序的执行。

`ctrl+z`是将任务终端，但是此任务没有结束，仍然在进程中，只是维持挂起状态。

`fg`重新启动被前台中断的任务，把后台的命令调至前台继续运行`fg %job_num`

`bg`将后台中断的进程继续运行

`jobs`查看当前有多少在后台运行的命令

`kill`杀死进程，只有第九种信号（SIGKILL）才可以无条件终止进程，其他信号进程都有权利忽略，下面是常用的信号：

1. HUP 1 终端断线
2. INT 2 中断（同 Ctrl + C）
3. QUIT 3 退出（同 Ctrl + \）
4. TERM 15 终止
5. KILL 9 强制终止
6. CONT 18 继续（与STOP相反， fg/bg命令）
7. STOP 19 暂停（同 Ctrl + Z）

`pgrep`和`pkill`根据名字查找进程或发送信号。

## 11. 利用ssh进行远程登陆

密码登陆 `ssh -p port username@webserverip`

密钥登陆 `ssh -i ~/.ssh/id_rsa_1 username@webserverip`

利用ssh-keygen生成密钥对，公密: ../.ssh/id_rsa.pub 私密:../.ssh/id_rsa 将公密放到服务器

ssh配置文件 `/etc/ssh/sshd_config`设置端口，设置是否允许密码登陆、是否需要进行密钥验证等等。

```shell
Port 32200
RSAAuthentication yes
PubkeyAuthentication yes
```

## 12. 基本的文件管理工具

less, head, tail, tail -f, ln, ln -s, chown, chmod, du, df, fdisk, mkfs, lsblk, inode

ln 硬链接

ln -s 软连接



## 13. 基本的网络管理工具ifconfig

ifconfig查看网络内容，启动或关掉网卡，修改网络ip， 修改mac地址



## 14. 版本控制git



## 15. 熟悉正则表达式，使用grep

grep 命令常用于查找文件中符合条件的字符串：

-i:忽略大小写

-o:只显示匹配的内容

-v:不匹配符合的内容

-A:除了显示匹配的内容之外，还显示该行之前的n行

-B:除了显示匹配的内容之外，还显示该行之后的n行

-C:除了显示匹配的内容之外，还显示改行前后的n行



## 16. bash命令操作快捷键

ctrl+w:删除键入的最后一个单词

ctrl+u:删除行内光标坐在位置之前的内容

alt+b,alt+f:可以以单词为单位移动光标

ctrl+a:将光标移至行首

ctrl+e:将光标移至行尾

ctrl+k:删除光标至行尾的所有内容

ctrl-l:清屏



## 17. 历史记录

#### 键入 history 查看命令行历史记录

`!n`（n是命令编号）就可以再次执行

`!$`表示上次键入的最后一个参数

`!!`上次键入的命令

`alt-.`循环地向前显示历史记录

`ctrl+r`进行历史命令搜索，重复按下会向下继续匹配，按下enter会执行当前匹配的历史命令。



## 18. xargs命令



## 19. 查看进程监听的端口

使用`netstat -lntp`或`ss -plat`检查哪些进程在监听端口（默认是检查TCP端口，添加参数`-u`则检查UDP端口）

或者`lsof -iTCP -sTCP:LISTEN -P -n`

`netstat`部分参数：

- -t (tcp) 仅显示tcp相关选项
- -u (udp) 仅显示udp相关选项
- -n 拒绝显示别名，能显示数字的全部转化为数字
- -l 仅列出在Listen(监听)的服务状态
- -p 显示建立相关链接的程序名

查看结果如下：

```shell
# netstst -lntp | grep 32200
tcp        0      0 0.0.0.0:32200           0.0.0.0:*               LISTEN      492/sshd
```



## 20.查看开启的套接字和文件

### lsof(list open files)是一个列出当前系统打开文件的工具。

##### lsof 输出各列信息含义:

| 字段    | 含义                         |
| ------- | ---------------------------- |
| COMMAND | 进程名称                     |
| PID     | 进程标识符                   |
| USER    | 进程所有者                   |
| FD      | 文件描述符                   |
| TYPE    | 文件类型                     |
| DEVICE  | 指定磁盘名称                 |
| SIZE    | 文件大小                     |
| NODE    | 索引节点（文件在磁盘的标识） |
| NAME    | 打开文件的确切名称           |

部分lsof命令：

```shell
lsof -i:8080 # 查看8080端口占用
lsof abc.txt # 显示开启文件abc.txt的进程
lsof -c abc  # 显示abc进程所打开的文件
lsof -c -p 1234 # 列出进程号为1234的进程所打开的文件
lsof -g gid  # 显示归属gid的进程情况
lsof +d /usr/local/ # 显示目录下被进程开启的文件
lsof +D /usr/local/ # 同上，但是会搜索目录下的目录，时间较长
lsof -d 4 # 显示fd为4的进程
lsof -i -U # 显示所有打开的端口喝UNIX domain文件
```

如果你删除了一个文件，但通过 du 发现没有释放预期的磁盘空间，请检查文件是否被进程占用： lsof | grep deleted | grep "filename-of-my-big-file"

## 21. 括号的使用

在bash脚本中，子shell（使用括号`(...)`）是一种组织参数的便捷方式。一个常见的例子是临时地移动工作路径，代码如下：

```shell
# do something in current dir
(cd /some/other/dir && other-command)
# continue in original dir
```



## 22. 括号扩展

使用括号扩展（`{...}`）来减少数据相似文本，并自动化文本组合

```shell
mv foo.{txt,pdf} some-dir # 同时移动两个文件
cp somefile(,.bak) # 会被扩展成 cp somefile somefile.bak
mkdir -p test-{a,b,c} /subtest-{1,2,3} # 会被扩展成所有可能地组合，并创建一个目录树
```



## 23. 将web服务器上当前目录下所有文件（以及子目录）暴露给你所处网络地所有用户

```python
python -m SimpleHTTPServer 7777 # (使用端口7777 和 Python2)
python -m http.server 7777 # (使用端口 7777 和 Python3)
```



## 24. 文件查找

在当前路径下查找`find . -name '*something'`

在所有路径下通过文件名查找文件，使用`locate something`



## 25.使用 sort 和 uniq，uniq 的 -u 参数和 -d 参数

#### sort命令用于将文本文件内容加以排序

- `-b` 忽略每行前面开始出现地空格字符
- `-c` 检查文件是否已经按照顺序排序
- `-d` 排序时处理英文字母、数字以及空格字符，忽略其他的字符
- `-f` 排序时将小写字母视为大写字母
- `-i` 排序时，除了040至176之间的ASCII字符外，忽略其他的字符
- `-m` 将几个排序好的文件进行合并
- `-M` 将前面3个字母依照月份的缩写进行排序
- `-n` 依照数值的大小排序
- `-u` 意味着时唯一的(unique)，输出的结果是去重之后的
- `-o <输出文件>`  将排序后的结果存入指定的文件
- `-r` 以相反的顺序来排序
- `-t <分隔符>` 指定排序时所用的栏位分隔字符
- `+<起始栏位>-<结束栏位>` 以指定的栏位来排序，范围由起始栏位到结束栏位的前一栏位
- `--help`



### uniq命令用于检查及删除文本文件中重复出现的行列, 一般与sort命令结合使用

- `-c` 或 `--count` 在每列旁边显示该行

- `-d` 或 `--repeated` 仅显示重复出现的行列。
- `-f <栏位>` 或 `--skip-fields=<栏位>` 忽略比较指定的栏位。
- `-s <字符位置>` 或 `--skip-chars=<字符位置>` 忽略比较指定的字符。
- `-u` 或 `--unique` 仅显示出现一次的行列
- `-w<字符位置>` 或 `--check-chars=<字符位置>` 指定要比较的字符
- `--help`
- `[输入文件] `指定已排序好的文本文件。如果不指定此项，则从标准读取数据
- `[输出文件] `指定输出的文件。如果不指定此选项，则将内容显示到标准输出设备（显示终端）

#### 当重复的行并不相邻时，uniq 命令是不起作用的,所以uniq命令往往和sort命令一起使用



## 26. 使用cut，paste和join来更改文件



## 27. 运用wc计算新行数(-l)，字符数(-m)，单词数(-w)以及字节数(-c)

wc命令用于计算字数，利用wc指令我们可以计算文件的Byte数、字数或是列数：

- `-c`或`--bytes`或`--chars`只显示Bytes数
- `-l`或`--lines` 只显示行数。
- `-w`或`--words` 只显示字数,单词数。



## 28.使用tee将标准输出复制到文件甚至标准输出

tee命令用于读取标准输入的数据，并将其内容输出成文件



## 29.使用awk和sed进行简单的数据处理

AWK是一种处理文本文件的语言，是一个强大的文本分析工具，相对于grep的查找，sed的编辑，awk在其对数据分析并生成报告时，显得尤为强大。简单来说awk就是把文件逐行的读入，以空格为默认的分隔符将每行切片，切开的部分再进行各种分析处理。



## 30. 使用repren来批量重命名文件，或是再多个文件中搜索替换内容。（有时候rename命令也可以批量重命名，但是要注意，他在不同的Linux发行版的功能并不完全一样）

```shell
# 将文件、目录和内容全部重命名 foo -> bar:
repren --full --preserve-case --from foo --to bar .

# 还原所有备份文件 whatever.bak -> whatever:
repren --renames --from '(.*)\.bak' --to '\1' *.bak

# 用rename实现上述功能(若可用)
 rename 's/\.bak$//' *.bak
```



## 31. shuf可以以行为单位来打乱文件的内容或从文件中随机选取多行



## 32. 对于二进制文件，使用hd, hexdump或者xxd使其以十六进制显示，使用bvi, hexedit或者biew来进行二进制编辑。



## 33. 拆分文件可以使用split(按大小写拆分)和csplit(按模式拆分)



## 34. 使用getfacl和setfacl以保存和恢复文件权限

```shell
getfacl -R /some/path > permissions.txt
setfacl --restore=permissions.txt
```



## 35. cpu状态

获取cpu和硬盘的状态，通常使用top(htop更佳)，iostat和iotop。

而`iostat -mxz 15`可以让你获悉CPU和每个硬盘分区的基本信息和性能表现。


## 36. 内存状态

若要了解内存状态，运行并了解`free`和`vmstat`的输出。值得留意的是"cached"的值，它指的是Linux内核用来作为文件缓存的内存大小，而与空闲内存无关。



## 37. 查看当前使用的系统

使用`uname`,`uname -a (Unix/kernel信息)`或者`lsb_release -a (Linux发行版信息)`



## 38. 单行脚本

当需要对文本文件做集合交并差运算时，sort和uniq会是好帮手。此处假设 a 与 b 是两内容不同的文件。这种方式效率很高，并且在小文件和上 G 的文件上都能运用（注意尽管在 /tmp 在一个小的根分区上时你可能需要 -T 参数，但是实际上 sort 并不被内存大小约束），参阅前文中关于 `LC_ALL` 和 sort 的 -u 参数的部分。

```shell
sort a b | uniq > c # c是a并b
sort a b | uniq -d > c # c是a交b
sort a b b | uniq -u > c # c是a-b
```

使用`grep .`每行都会附上文件名，或者`head -100`每个文件有一个标题，来阅读检查目录下所有文件的内容。这在检查一个充满配置文件的目录（如/sys,/proc,/etc）时特别好用。

```shell
ls hosts out.txt| head -100 *
```

要持续检测文件改动，可以使用watch，例如检查某个文件夹中文件的改变，可以用`watch -d -n 2 'ls -rtlh|tail'`

或者在排查wifi设置故障时要监测网络设置的更改，可以用`watch -d -n 2 ifconfig`

如果想在文件数上查看大小、日期，这可能看起来像递归版的`ls -l`但比`ls -lR`便于理解

```shell
find . -type f -ls
```

计算文本文件第三列中所有数的和（比python快）

```shell
awk '{x += $3 } END {print x}' myfile
```



## 39. 用shell把文件复制一百份
```shell
for i in {1..100};do echo -ne "hello-$i.txt ";done | xargs -n 1 cp hello.txt
```



## 40. 批量修改文件名

```shell
# 把 xxx10.csv 都改为 xxx_real.csv
rename "s/10/_real/" *.csv
```

