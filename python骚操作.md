## 巧用else语句（重要）

python的else 子句不仅能在 if 语句中使用，还能在 for、while 和 try 等语句中使用，这个语言特性不是什么秘密，但却没有得到重视。

for：

```python
l=[1,2,3,4,5]
for i in l:
    if i=='6':
        print(666)
        break
else:
    print(999)
```

如果不这么实现，我们只能设置一个变量来记录了：

```python
l=[1,2,3,4,5]
a=1
for i in l:
    if i=='6':
        print(666)
        a=0
        break
if a:
    print(999)
```

while和for类似

看一下try：

```python
try:
    a()
except OSError:
    #语句1
else:
    #语句2
```

仅当 try 块中没有异常抛出时才运行 else 块。

**总结一下else：**

for:

　　仅当 for 循环运行完毕时（即 for 循环没有被 break 语句中止）才运行 else 块。

while:

　　仅当 while 循环因为条件为假值而退出时（即 while 循环没有被break 语句中止）才运行 else 块。

try:

　　仅当 try 块中没有异常抛出时才运行 else 块。

即，如果`异常`或者 `return`、`break` 或 `continue` 语句导致**控制权跳到了复合语句的主块之外**，那么`else` 子句也会被跳过。

  按正常的理解应该是“要么运行这个循环，要么做那件事”。可是，在循环中，else 的语义恰好相反：“运行这个循环，然后做那件事。”



## except的用法和作用

`try/except`: 捕捉由PYTHON自身或写程序过程中引发的异常并恢复

`except`: 捕捉所有其他异常

`except name`: 只捕捉特定的异常

`except name, value`: 捕捉异常及格外的数据(实例)

`except (name1,name2) `: 捕捉列出来的异常

`except (name1,name2),value`:  捕捉任何列出的异常，并取得额外数据

`else`: 如果没有引发异常就运行

`finally`: 总是会运行此处代码

## Python自省

这个也是python彪悍的特性.**自省就是面向对象的语言所写的程序在运行时,所能知道对象的类型.简单一句就是运行时能够获得对象的类型**.比如`type()`,`dir()`,`getattr()`,`hasattr()`,`isinstance()`.



## python容器

列表：元素可变（任何数据类型），有序（可索引），append/insert/pop；

元组：元素不可变，但元素中的可变元素是可变的；有序（可索引）；而且元组可以被散列，例如作为字典的键。

集合：无序（不可被索引）、互异

字典：无序，键值对（key：value），key唯一不可重复


## map（）

`map()`函数接收两个参数，一个是函数，一个是`Iterable`，`map`将传入的函数依次作用到序列的每个元素，并把结果作为新的`Iterator`返回。（重点理解）

举例说明，比如我们有一个函数`f(x)=x2`，要把这个函数作用在一个`list [1, 2, 3, 4, 5, 6, 7, 8, 9]`上，就可以用`map()`实现如下：

```python
>>> def f(x):
...     return x * x
...
>>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```

`map()`作为高阶函数，事实上它把运算规则抽象了，因此，我们不但可以计算简单的`f(x)=x2`，还可以计算任意复杂的函数，比如，把这个list所有数字转为字符串：



## reduce

`reduce`把一个函数作用在一个序列`[x1, x2, x3, ...]`上，这个函数必须接收两个参数，`reduce`把结果继续和序列的下一个元素做累积计算

简单例子：

```python

>>> from functools import reduce
>>> def fn(x, y):
        return x * 10 + y
 
>>> reduce(fn, [1, 3, 5, 7, 9])
13579
```

结合一下，我们可以自己写出int（）函数

```python'

from functools import reduce
 
a={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
 
def charnum(s):
    return a[s]
 
def strint(s):
    return reduce(lambda x, y: x * 10 + y, map(charnum, s))
```



## split

Python **split()** 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串。

语法：

```pyhon
str.split(str="", num=string.count(str))
```

简化：

`str.split("")`



## 理论结合实际

1）我们可以写出这一行代码

```python
print(" ".join(input().split(" ")[::-1]))
```

实现功能，leetcode原题：给定一个句子（只包含字母和空格），将句子中的单词位置反转，单词用空格分割，单词之间只有一个空格，前后没有空格。比如：（1）“hello xiao mi” - >“ mi xiao hello “

 

2）再举一例：

将两个整型数组按照升序合并，并且过滤掉重复数组元素

输入参数：

int* pArray1 ：整型数组1

intiArray1Num：数组1元素个数

int* pArray2 ：整型数组2

intiArray2Num：数组2元素个数

```python
a,b,c,d=input(),list(map(int,input().split())),input(),list(map(int,input().split()))
print("".join(map(str,sorted(list(set(b+d))))))
```



## filter

Python内建的`filter()`函数用于过滤序列。

和`map()`类似，`filter()`也接收一个函数和一个序列。和`map()`不同的是，`filter()`把传入的函数依次作用于每个元素，然后根据返回值是`True`还是`False`决定保留还是丢弃该元素。

简单例子，删掉偶数：

```python
def is_odd(n):
    return n % 2 == 1
 
list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
# 结果: [1, 5, 9, 15]
```

我们可以用所学知识实现埃氏筛：

```python

#先构造一个从3开始的奇数序列：
def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n
#这是一个生成器，并且是一个无限序列。
 
#筛选函数
def _not_divisible(n):
    return lambda x: x % n > 0
#生成器
def primes():
    yield 2
    it = _odd_iter() # 初始序列
    while True:
        n = next(it) # 返回序列的第一个数
        yield n
        it = filter(_not_divisible(n), it) # 构造新序列
```

利用`filter()`不断产生筛选后的新的序列

`Iterator`是惰性计算的序列，所以我们可以用Python表示“全体自然数”，“全体素数”这样的序列，而代码非常简洁。



## sorted

```python
>>> sorted([36, 5, -12, 9, -21])
[-21, -12, 5, 9, 36]
#可以接收一个key函数来实现自定义的排序，例如按绝对值大小排序：
>>> sorted([36, 5, -12, 9, -21], key=abs)
[5, 9, -12, -21, 36]
```

我们再看一个字符串排序的例子：

```python
>>> sorted(['bob', 'about', 'Zoo', 'Credit'])
['Credit', 'Zoo', 'about', 'bob']
```

默认情况下，对字符串排序，是按照ASCII的大小比较的，由于'Z' < 'a'，结果，大写字母Z会排在小写字母a的前面。

现在，我们提出排序应该忽略大小写，按照字母序排序。要实现这个算法，不必对现有代码大加改动，只要我们能用一个key函数把字符串映射为忽略大小写排序即可。忽略大小写来比较两个字符串，实际上就是先把字符串都变成大写（或者都变成小写），再比较。

这样，我们给sorted传入key函数，即可实现忽略大小写的排序：

```python
>>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower)
['about', 'bob', 'Credit', 'Zoo'
```

要进行反向排序，不必改动key函数，可以传入第三个参数reverse=True：

```python
sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True)
['Zoo', 'Credit', 'bob', 'about']
```

从上述例子可以看出，高阶函数的抽象能力是非常强大的，而且，核心代码可以保持得非常简洁。

sorted()也是一个高阶函数。用sorted()排序的关键在于实现一个映射函数。
