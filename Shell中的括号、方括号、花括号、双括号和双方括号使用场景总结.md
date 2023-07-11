# Shell中的括号、方括号、花括号、双括号和双方括号使用场景总结

## 1. 括号()

括号一般在命令替换时与美元符号$配合使用，用于获取括号内命令的输出，如

```shell
#!/bin/bash
# 输出今年的年份
year=$(date +%Y)
echo "This year is $year"
```



## 2. 方括号[]

Shell中的方括号一般有两种使用场景，一种是和美元符号$搭配用于Shell中整型数据运算；另一种是单独使用，作为test命令的简写形式。

### 2.1 搭配$用于整型计算

```shell
# 1. 用于整型数据计算
var1=100
var2=200
# 可以使用 $+方括号来表示整型运算
var3=$[ $var1+$var2+1 ]
# 也可以使用 $+双括号来表示整型运算
var4=$(($var1 + $var2 + 1))
echo $var3
echo $var4
```

由于 if-then语句 不能测试命令状态码之外的条件，所以Bash Shell提供了 test命令 用于帮助 if-then语句 测试其他的条件，如数值比较、字符串比较、文件比较等，而test命令的简写形式就是方括号`[ ]`，其中**第一个方括号和第二个方括号之前都必须加上空格**，否则会报错


### 2.2 数值比较

| 比较      | 描述                   |
| --------- | ---------------------- |
| n1 -eq n2 | 检查n1是否与n2相等     |
| n1 -ge n2 | 检查n1是否大于或等于n2 |
| n1 -gt n2 | 检查n1是否大于n2       |
| n1 -le n2 | 检查n1是否小于或等于n2 |
| n1 -lt n2 | 检查n1是否小于n2       |
| n1 -ne n2 | 检查n1是否不等于n2     |

```shell
#!/bin/bash
# 2. 数值比较
n1=20
n2=10
if [ $n1 -ge $n2 ]; then
    echo "n1 is greater than or euqal to n2"
else
    echo "n1 is less than n2"
fi
```

**Bash Shell只能直接处理整数，赋值浮点数会报错**



### 2.3 字符串比较（注意符号两边两边要有空格）

比较	描述（注意符号两边需要空格）
str1 = str2	检查str1是否和str2相同
str1 != str2	检查str1是否和str2不同
str1 < str2	检查str1是否比str2小
str1 > str2	检查str1是否比str2大
-n str1	检查str1的长度是否非0
-z str1	检查str1的长度是否为0

```shell
# 3. 字符串比较
user=root
if [ $(whoami) = $user ]; then
    echo "root is online"
else
    echo "root is offline"
fi
```



### 2.4 文件比较

比较	描述
-d file	检查file是否存在并是一个目录
-e file	检查file是否存在
-f file	检查file是否存在并是一个文件
-r file	检查file是否存在并可读
-s file	检查file是否存在并非空
-w file	检查file是否存在并可写
-x file	检查file是否存在并可执行
-O file	检查file是否存在并属当前用户所有
-x file	检查file是否存在并可执行
-G file	检查file是否存在并且默认组与当前用户相同
file1 -nt file2	检查file1是否比file2新
file1 -ot file2	检查file1是否比file2旧

```shell
# 4. 文件比较
fileName=test3
if [ -e $fileName ]; then
    echo "$fileName  exists"
else
    echo "$fileName doesn't exists"
fi
```



## 3. 花括号{}

花括号一般用于需要变量和字符串组合输出时，若想要实现变量后拼接字符串就需要使用花括号

```shell
#!/bin/bash
# 花括号使用练习
var=50
var1=100
var2=200
# 若想要实现var变量后拼接字符串就需要使用花括号
echo $var1 ${var}1
echo $var2 ${var}2

```



## 4. 双括号(( ))

**双括号**允许在比较语句中使用**高级数学表达式**，**也可以与美元符号搭配，用于整型数据计算**

| 符号  | 描述     |
| ----- | -------- |
| val++ | 后增     |
| val-- | 后减     |
| ++val | 先增     |
| --val | 先减     |
| ！    | 逻辑求反 |
| ～    | 按位求反 |
| **    | 幂运算   |
| <<    | 左移位   |
| >>    | 右移位   |
| &     | 布尔与   |
| \|    | 布尔或   |
| &&    | 逻辑与   |
| \|\|  | 逻辑或   |

```shell
#!/bin/bash
# 双括号使用练习
# 用于高级数学表达式
var1=10
if (($var1 >= 10)); then
    for ((i = 0; i < 3; i++)); do
        echo $i
    done
fi
```

```shell
# 用于整型数据计算
var1=100
var2=200
# 可以使用$+双方括号来表示整型运算
var3=$[ $var1+$var2+1 ]
# 也可以使用$+双括号来表示整型运算
var4=$(($var1 + $var2 + 1))
echo $var3
echo $var4
```



## 双方括号

双方括号提供了针对字符串比较的高级特性，能够使用数学符号比较字符串，并实现了模式匹配

```shell
#!/bin/bash
# 双方括号使用练习
fileName=test5
if [[ $fileName==test* ]]; then
    echo "This is a test file!"
    if [[ $fileName==test5 ]]; then
        echo "This file is test5!"
    fi
fi

```

- **注意：不是所有的Shell都支持双方括号**
