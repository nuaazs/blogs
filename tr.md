# tr命令用法

## 使用方法

1.1 作用

tr命令用于字符串转化、替换和删除，主要用于删除文件中的控制符或进行字符串转换等。

1.2 tr命令格式

```shell
# 用法1：命令的执行结果交给tr处理，其中string1用于查询，string2用于转化处理
commands | tr 'string1' 'string2'
# 用法2：对来自于filename文件中的内容进行字符替换
tr 'string1' 'string2' < filename
# 用法3：对来自filename文件的内容查询string1并进行相应处理，比如删除等。
tr option 'string1' < filename
```

1.3 tr命令常用的选项

```
-d: 删除字符串
-s: 删除所有重复出现的字符序列，只保留第一个，即将重复出现字符串压缩为一个字符串
```

1.4 常用的匹配字符串

```
# 匹配所有小写字母
a-z或者[:lower:]

# 匹配所有大写字母
A-Z或者[:upper:]

# 匹配所有数字
0-9或者[:digit:]

# 匹配所有字母和数字
[:alnum:]

# 匹配所有字母
[:alpha:]

# 匹配所有空白
[:blank:]

# 匹配所有水平或者垂直的空格
[:space:]

# 匹配所有控制字符
[:cntrl:]
```