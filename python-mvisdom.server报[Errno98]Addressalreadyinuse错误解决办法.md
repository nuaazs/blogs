## 问题

当输入：`python3 -m visdom.server`的时候，会报错：“`address already in use`”。
原因是地址被占用。



## 解决办法

先在命令行中输入：lsof -i tcp:8097，其中8097是端口号。
用以下命令结束这个pid:

```shell
kill -9 # 是被占用的
```

这时候再输入，就没有内容返回了：

```shell
lsof -i tcp:8097
```

`python3 -m visdom.server`

