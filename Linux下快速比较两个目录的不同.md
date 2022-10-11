背景：处理数据集时不小心删除了一个不知道文件名的数据，想要对比targets和inputs文件夹下的文件，看看删除的是哪个。

dir1: targets文件夹地址

dir2: inputs文件夹地址

### 方法1 diff

`diff -r dir1 dir2`

diff会对每个文件中的每一行都做比较，所以文件较多或者文件较大的时候会非常慢。而且我只是需要对比文件名，他们的文件内容本就不同。



### 方法2 diff和tree结合

```bash
diff <(tree -Ci --noreport $dir1) <(tree -Ci --noreport $dir2)
```

tree的`-C`选项是输出颜色，如果只是看一下目录的不同，可以使用该选项，但在结合其他命令使用的时候建议不要使用该选项，因为颜色也会转换为对应的编码而输出；

`-i`是不缩进，建议不要省略-i，否则diff的结果很难看，也不好继续后续的文件操作；

`--noreport`是不输出报告结果，建议不要省略该选项。

该方法效率很高。



### 方法3 find和diff结合

既然diff会比较文件，那就把目录写到两个目录文件里，然后对比这两个文本文件即可。

```bash
find dir1 -printf "%P\n" | sort > file1
find dir2 -printf "%P\n" | sort > file2
diff file1 file2
```

```bash
find directory1 -printf "%P\n" | sort > file1
find directory2 -printf "%P\n" | sort | diff file1 -
```

结果中：`<`代表的行是dir1中有而dir2没有的文件，`>`则相反。

不要省略`-printf "%P\n"`，此处的`%P`表示find的结果中**去掉前缀路径**，详细内容`man find`。例如，`find /root/ -printf "%P\n"`的结果中将显示`/root/a/xyz.txt`中去掉`/root/`后的结果：`a/xyz.txt`。

效率很高，输出也简洁。如果不想使用`-printf`，那么先进入各目录再find也行。

```bash
[root@node1 ~]# (cd /root/a;find . | sort >/tmp/file1)    
[root@node1 ~]# (cd /root/b;find . | sort | diff /tmp/file1 -)
2d1
< ./1.png
4a4
> ./4.png
```



### 方法4 使用rsync（没看懂）

```bash
rsync -rvn --delete dir1/ dir2 | sed -n '2,/^$/{/^$/!p}'
deleting a/xyz.avi
rsync.txt
```

新建文件夹/542D0.mp4

其中deleting所在的行就是directory2中多出的文件。其他的都是directory中多出的文件。

如果想区分出不同的是目录还是文件。可以加上"-i"选项。

```bash
rsync -rvn -i --delete directory1/ directory2 | sed -n '2,/^$/{/^$/!p}'
*deleting  a/xyz.avi
>f+++++++++ rsync.txt
>f+++++++++ 新建文件夹/542D0.mp4
```

其中>f+++++++++中的f代表的是文件，d代表的目录。

**上面的rsync比较目录有几点要说明：**

一定不能缺少-n选项，它表示dry run，也就是试着进行rsync同步，但不会真的同步。

第一个目录(directory1/)后一定不能缺少斜线，否则表示将directory1整个目录同步到directory2目录下。

其它选项，如`"-r -v --delete"`也都不能缺少，它们的意义想必都知道。

sed的作用是过滤掉和文件不相关的内容。

以上rsync假定了比较的两个目录中只有普通文件和目录，没有软链接、块设备等特殊文件。如果有，请考虑加上对应的选项或

者使用-a替代-r，否则结果中将出现`skipping non-regular file`的提示。但请注意，如果有软链接，且加了对应选项(-l或-a或其他

相关选项)，则可能会出现fileA-->fileB的输出。

效率很高，因为rsync的原因，筛选的可定制性也非常强。



### 方法5 图形化方法vimdiff

```bash
vimdiff <(cd dir1; find . | sort) <(cd dir2; find . | sort)
# 或者
vimdiff <(find dir1 -printf "%P\n"| sort) <(find dir2 -printf "%P\n"| sort)
```





### 方法6 图形化方法meld

meld是python写的一个图形化文件/目录比较工具，所以必须先安装图形界面或设置好图形界面接受协议。它的功能非常丰富，和win下的`beyond compare`有异曲同工之妙。meld具体的使用方式就不介绍了。