## sorted

help里给的解释

```python
>>> help(sorted)
Help on built-in function sorted in module __builtin__:

sorted(...)
    sorted(iterable, cmp=None, key=None, reverse=False) --> new sorted list
```

一般，sorted 对字符串排序，ASCII比较大小的时候,是比较两个数中的第一个字符。ASCII码的大小规则，`0－9＜A－Z＜a－z`。由于'Z' < 'a'，结果，**大写字母Z会排在小写字母a的前面**，忽略大小写来比较两个字符串，实际上就是先把字符串都变成大写（或者都变成小写），再比较。这样，我们给`sorted`传入`key`函数，即可实现忽略大小写的排序：

```python
#Python lower() 方法转换字符串中所有大写字符为小写
#这一步把大写变小写，再sorted
>>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower)
 ['about', 'bob', 'Credit', 'Zoo']
```



## 测试

```python
from operator import itemgetter 
students = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
print(sorted(students,key=itemgetter(0)))#key值设置成第一个域来排序
print(sorted(students, key=lambda t: t[1]))#key值设置称t[1],第二个域来排序
print(sorted(students, key=lambda t: t[0],reverse=True))#key值设置成t[0],第一个域来排序，注意这有个反转，反转就是把顺序倒过来　  print(sorted(students, key=itemgetter(1), reverse=True))
```

```python
>>>[('Adam', 92), ('Bart', 66), ('Bob', 75), ('Lisa', 88)] 
>>>[('Bart', 66), ('Bob', 75), ('Lisa', 88), ('Adam', 92)]
>>>[('Lisa', 88), ('Bob', 75), ('Bart', 66), ('Adam', 92)]
>>>[('Adam', 92), ('Lisa', 88), ('Bob', 75), ('Bart', 66)]
```

其中key的参数为**一个函数或者`lambda`函数**，如下例子对`lamdba`的解释，我对`lamdba`的理解就是这东西是自定义函数

```python
r = lambda x,y:x*y
r(2,3) # 6
```



`itemgetter`是从`operator`里面倒进来的，可以用来当key的参数。

如果要比较一个班学生的成绩，但是同时有B同学，一个人得12分，一个人10分，那就来上两个域进行排序。

如下：根据第二个域和第三个域进行排序

```python
#key值设置成第一个域来排序
a = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]

print(sorted(a, key=itemgetter(1,2)))
>>>[('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)]# 但是姓名没有排序，所以给第一个域来一波操作

print(sorted(a, key=itemgetter(0,1,2)))
>>>[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
```



## 数模比赛

