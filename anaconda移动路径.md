|anaconda|

在210实验室HP服务器上安装深度学习套件。
发现空间不足，是当时他们挂载磁盘的时候把4T最大的盘挂载在了home目录下，所以我之前用root用户安装的anaconda和torch之类的很快将空间占满了。
有两种解决方法：

1. 热挂载，给root目录下挂载更多的空间，我没有尝试，毕竟还有程序在跑，不想出问题。可以参考[这篇博客](https://blog.csdn.net/yuxuan_08/article/details/108824742?utm_term=centos%20%E5%B0%91%E4%BA%86%E4%B8%80%E4%BA%9B%E7%A9%BA%E9%97%B4%20%E6%8C%82%E8%BD%BD%E7%A1%AC%E7%9B%98&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-1-108824742&spm=3001.4430)。
2. 直接把anaconda目录移到home里面的某个用户里面去：
但是这样的话会导致找不到python,conda等命令。
解决方法：
首先查看bashrc里面之前的anaconda部分，把对应路径改过来，以及他source的文件中的对应路径也要改。
之后建立软连接：
```shell
ln -s /现在的python路径/python /以前的python路径
ln -s /home/Anaconda3/bin/python3.9 /home/Anaconda3/bin/python 
```
现在conda 命令正常了。参考[博客](https://blog.csdn.net/qq_34342853/article/details/123020957)。