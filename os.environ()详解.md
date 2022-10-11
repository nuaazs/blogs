## 一、简介

对于官方的解释，`environ`是一个字符串所对应环境的映像对象。这是什么意思呢？举个例子来说，`environ['HOME']`就代表了当前这个用户的主目录。os.environ 返回一个系统变量和系统变量值 的字典（键值对）

## 二、key字段详解

作为一个渗透测试学习者来说，对系统的足够了解是基本的要求，下面就通过对os.environ中的key解读的角度来认识系统。

windows：

```
os.environ['HOMEPATH']:当前用户主目录。
os.environ['TEMP']:临时目录路径。
os.environ[PATHEXT']:可执行文件。
os.environ['SYSTEMROOT']:系统主目录。
os.environ['LOGONSERVER']:机器名。
os.environ['PROMPT']:设置提示符。
```

linux：

```
os.environ['USER']:当前使用用户。
os.environ['LC_COLLATE']:路径扩展的结果排序时的字母顺序。
os.environ['SHELL']:使用shell的类型。
os.environ['LAN']:使用的语言。
os.environ['SSH_AUTH_SOCK']:ssh的执行路径。
```

