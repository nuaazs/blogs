# Python中一切皆对象
## type/object/class的关系
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/0LRmZ.jpg)
实线代表继承，虚线代表实例。
type实现了一切皆对象，连自己都是自己的对象。
object是所有类的基类。

## Python常见的内置类型
### 对象的三个特征





$$- \frac { h ^ { 2 } } { 2 h } \frac { \partial ^ { 2 } W ( x , t ) } { \partial x ^ { 2 } } + U ( x , t )  ( x , t ) = i n \frac { \partial \pi ( x , t ) } { \partial t }$$



1. 身份：在内存中的地址
```python
a=1
id(a)
#501794480
```
2. 类型
3. 值

### None(全局只有一个）
```python
a = None
b = None
id(a) == id(b) #True
### 数值
### 迭代类型
### 序列类型
1. list
2. bytes/bytearray/memoryview(二进制序列)
3. range
4. tuple
5. str
6. array
### 映射类型
1. dict
### 集合
1. set
2. frozenset
### 上下文管理类型（with）
### 其他
1. 模块类型
2. class和实例
3. 函数类型
4. 方法类型
5. 代码类型
6. object对象
7. type类型
8. ellipsis类型
9. notimplemented类型


# 2 Python魔法函数
## 2.1 什么是魔法函数
双下划线开头的函数，双下划线结尾，不要自己定义。
​```python
class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list
    def __getitem__(self, item):
        # 如果没有迭代器__iter__就会调用getitem
        return self.employee[item]
company = Company(["tom","bob"])
for em in company:
    print(em)
    
 company1 = company[:1] # 还可以切片
 print(len(company)) #先尝试调用__len__ 如果没有就用__getitem__
```
 ## 2.2 python有哪些魔法函数
 ### 2.2.1 非数学运算

 1. __str__ 和 __repr__
 ```python
class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list
    def __getitem__(self, item):
        # 如果没有迭代器__iter__就会调用getitem
        return self.employee[item]
	def __str__(self):
	    return ",".join(self.employee)
	def __repr__(self):
	    return ",".join(self.employee)
company = Company(["zhaosheng","linzhiyi","luhong"])
print(company) # __Str__
company # jupyternotebook 里面 __repr__ 开发模式
repr(company) # python内部其实是这样调用魔法函数：company.__repr__()
 ```
2. __len__

 ### 2.2.2 数学运算

# 3.深入对象
## 3.1 鸭子类型和多态
```python
class Cat(object):
    def say(self):
        print("i am a cat")
class Dog(object):
    def say(self):
        print("i am a dog")
class Duck(object):
    def say(self):
        print("i am a duck")
        
animal = Cat
animal().say()

animal_list = [Cat, Dog, Duck]
for animal in animal_list:
    animal().say()

a = ["zhaosheng1","zhaosheng2"]
b = ["zhaosheng2","zhaosheng1"]

name_tuple = ["zhaosheng3","zhaosheng4"]
name_set = set()
name_set.add("zhaosheng5")
name_set.add("zhaosheng6")
a.extend(b) # ["zhaosheng1","zhaosheng2","zhaosheng2","zhaosheng1"]

```

extend()只要传递一个可迭代对象就可以了，其实tuple,set都是可迭代对象，之前实现了__getitem__的company对象也可以。
```python
a.extend(name_set)
a.extend(name_tuple)
```
## 3.2 抽象基类（abc模块）
python中的抽象基类也是不能实例化的
动态语言变量是没有类型的，只是一个符号而已，一个类型可以复制给任何变量。从语言层面讲，他已经是多态的了。
动态语言不需要指明变量的类型，所以少了检查错误的机会，只有运行的时候才会知道错误。
python信奉的时鸭子类型。
和java中最大的区别是：实现一个class的时候是不需要继承一个指定的类型的。

抽象基类：在这个基础的类中设置一些方法，所有用抽象基类的类都要

### 应用场景
1. 检查一个类是否有某个方法
```python
class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list
    def __getitem__(self, item):
        # 如果没有迭代器__iter__就会调用getitem
        return self.employee[item]
	def __str__(self):
	    return ",".join(self.employee)
	def __repr__(self):
	    return ",".join(self.employee)
company = Company(["zhaosheng","linzhiyi","luhong"])
print(hasattr(com,"__len__"))
print(len(company)) # 3

# 我们在某些情况之下想要判定某个对象的类型
from collections.abc import Sized
isinstance(com,Sized) # 自己写代码时更倾向于判断那类型，而不是用hasattr。 如果没有抽象基类就不行。

# 我们需要强制某个子类必须实现某些方法
# 实现了一个web框架，继承cache（redis,cache,memorychache）
# 需要设计一个抽象基类，指定子类必须实现某些方法
import abc
class CacheBase():
    def get(self,key):
        #pass
        raise NotImplementedError
        # 如果不重写就会报异常
        
    def set(self, key, value):
        pass
        
# 但只是这样也不好，我们希望在初始化的时候就报异常
# 要用到abc模块
# 这就是抽象基类带来的好处
class CacheBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get(self,key):
        pass
    @abc.abstractmethod
    def set(self, key, value):
        pass
```

## 3.3 尽量使用instance而不是type && 类变量和实例变量
```python
class A:
    pass
class B(A):
    pass 
b = B()

print(isinstance(b,B))
print(isinstance(b,A))

print(type(b))
print(type(b) is B) # True
# is 和 = 不一样
# is 实际上是判断是否为一个对象，id是否相同，id(B)
print(type(b) is A) # False
# type(b) 已经指向了 B, A是另一个对象，显然不相等
# 而如果用isinstance 则会内部自动查询继承关系

## 3.4 类变量和实例变量
​```python
class A:
    aa = 1 # 类变量
    def __init__(self,x,y):
        # self实际上是一个实例
        self.x = x # 实例变量
        self.y = y
a = A(2,3)
print(a.x, a.y, a.aa)
print(A.aa)

A.aa = 11 #修改了类变量 实例变量a.aa也会变
a.aa = 100 # 修改了实例变量，类变量A.aa不会变
```



## 类属性和实例属性及其查找顺序

由下而上

```python
class A:
    name = "A"
    def __init__(self):
        self.name = "obj"
a = A()
print(a.name) # 第一个查实例的name 如果没有就是类


# 多继承的时候就很负责

```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/UtNhU.jpg)

### MRO算法

DFS: A->B->D->C->E（深度优先）

深度优先：但如果是菱形的话就有问题。Python2.3改为广度优先

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/pXUQm.jpg)



python2.3: 广度优先



BFS: A->B->C->D->E

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/C0gQe.jpg)

python3:  C3算法

```python
# 新式类（默认继承object）
class D:
    pass

class C(D):
    pass

class B(D):
    pass

class A(B,C):
    pass
print(A.__mro__) # 查找顺序
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/Xn1TS.jpg)



## 3.5 类方法、静态方法和实例方法

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
        
        def tomorrow(self):
            self.day += 1
        
        #静态方法 采用硬编码，如果Date名字变了 这个里面也要边
        @staticmethod
        def parse_from_string(date_str):
            year, month, day = tuple(date_str.split("-"))
            return Date(int(year),int(month),int(day))
        
        # 类方法，就不是硬编码了，把class传了进去 cls只是简写而已，想写啥就写啥，只是一种规范。
        @classmethod
        def from_string(cls,date_str):
            year, month, day = tuple(date_str.split("-"))
            return Date(int(year),int(month),int(day))
        
        
        # Staticmethod有用场景，例如判断一个字符串是否合法，其实没有必要传递对象进来
        @staticmethod
        def valid_str(date_str):
            year, month, day = tuple(date_str.split("-"))
            # 判断步骤
            return True
    def __str__(self):
        return "{year}/{month}/{day}".format(year=self.year,month=self.month,day=self.day)
if __name__ == "__main__":
    new_day = Date(2018,12,31)
    new_day.tomorrow() # tomorrow(new_day)
    print(new_day)
    
    date_str = "2018-12-31"
    # 用staticmethod完成初始化
    new_day = Date.parse_from_string(date_str)
```



## 3.6 数据封装和私有属性

Java里面有private,protect

Python里面没有这些东西

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
        
        def tomorrow(self):
            self.day += 1
        
        #静态方法 采用硬编码，如果Date名字变了 这个里面也要边
        @staticmethod
        def parse_from_string(date_str):
            year, month, day = tuple(date_str.split("-"))
            return Date(int(year),int(month),int(day))
        
        # 类方法，就不是硬编码了，把class传了进去 cls只是简写而已，想写啥就写啥，只是一种规范。
        @classmethod
        def from_string(cls,date_str):
            year, month, day = tuple(date_str.split("-"))
            return Date(int(year),int(month),int(day))
        
        
        # Staticmethod有用场景，例如判断一个字符串是否合法，其实没有必要传递对象进来
        @staticmethod
        def valid_str(date_str):
            year, month, day = tuple(date_str.split("-"))
            # 判断步骤
            return True
    def __str__(self):
        return "{year}/{month}/{day}".format(year=self.year,month=self.month,day=self.day)

    
class User:
    def __init__(self,birthday):
        self.birthday = birthday
    def get_age(self):
        #返回年龄
        return 2018 - self.birthday.year
if __name__ == "__main__":
    user = User(Date(2018,12,31))
    print(user.get_age())
    print(user.birthday) # 如果不想用户看到birthday ： 用双下划綫！
```

```python
#把User改成
class User:
    def __init__(self, birthday):
        self.__birthday = birthday # 双下划綫开头 函数也是一样，可以在函数前加下划线
    def get_age(self):
        ...
if __name__ == "__main__":
    user = User(Date(2018,12,31))
    print(user.get_age())
    print(user.birthday) # 报错 之可以通过公共方法get_age获得
    
    print(user._User__birthday) #其实还是可以访问到，只是要一些规律，所以python不是绝对安全的。
```



## 3.9 python自省机制中常用的

自省是通过一定的机制查询到对象的内容结构

```python
class Person:
    name = "user"
class Student(Person):
    def __init__(self,school_name):
        self.scool_name = ""
if __name__ == "__main__":
    user = Student("iint")
    
    #通过__dict__查询属性
    print(user.__dict__)
    print(user.name) #name 没有进入__dict__：是因为name是属于Person的
    user.__dict__["school_addr"]= "abc" #动态操作对象
    print(Person.__dict__)
    
    print(dir(user)) #比 __dict__更加强大，可以列出所有属性
```



## 3.10 python中的super函数

```python
class A:
    def __init__(self):
        print("A")
class B(A):
    def __init__(self):
        print("B")
        super(B,self).__init__() # Python2用法 
        super().__init__() #python3简化了
#既然重写了B的构造函数，为什么还要调用super

from threading import Thread
class MyThread(Thread):
    def __init__(self, name, user):
        self.user = user
        super().__init__(name=name) #把构造函数交给了父类去实例化
    
# super到底执行顺序是什么
class C(A):
    def __init__(self):
        print("C")
        super().__init__() #python3简化了

        
class D(B,C):
    def __init__(self):
        print("D")
        super().__init__()
        
if __name__ == "__main__":
    b = B()
```



## 3.11 django rest framework中对多继承的使用经验

不推荐用多继承，设计不好的话很容易造成继承关系的混乱，MRO会造成一些预料不到的问题。

mixin:

1. mixin类功能单一
2. mixin不和基类关联，可以和任意基类组合
3. 在mixin中不要使用super用法



## 3.12 Python中的上下文管理器：with语句

```python
# try except finally
try:
    print("code started")
    raise keyError
except KeyError as e:
    print("key error")
```

```python
# try except finally
try:
    print("code started")
    raise IndexError
except KeyError as e:   # 捕获不到了
    print("key error")
else:
    print("other error") # 没用，因为只有没有异常的时候才回到else
```

```python
# try except finally
try:
    print("code started")
    raise IndexError
except KeyError as e:   # 捕获不到了
    print("key error")
else:
    print("other error") # 没用，因为只有没有异常的时候才回到else
finally:
    print("finally")  # 如果有异常最后还是会走到finally
   
```

finally里面有return就会return finally里面的



上下文管理器：实际上是python的一个协议，上下文管理器协议

```python
class Sample():
    def __enter__(self):
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        print("exit")
    def do_somethine(self):
        print("doing something")
with Sample() as sample: # 进入时会先调用 __enter__ 出来时调用__exit__
    sample.do_something()
```



## 3.13 contextlib简化上下文管理器

```python
import contextlib

# 将函数变成上下文管理器,他修饰的函数必须是一个生成器
@contextlib.contextmanager
def file_open(file_name):
    print("file open") # enter之前的代码
    yield {}
    print("file end")  # yield之后就是exit的逻辑

    
    
with file_open("aa.txt") as f_opened:
    print("file processing")
# file open
# file processing
# file end
```



## 3.14 小结



# 4. 自定义序列类

## 4.1 序列类型的分类

容器序列：list,tuple,deque

扁平序列：str, bytes, bytearray, aray.array

按照是否可变：可变序列list, deque, bytearray, array，不可变序列str, tuple, bytes

```python
my_list = []
my_list.append(1)
my_list.append("a") # list可以接收不同类型的数据
```

## 4.2 序列类型的协议

```python
from collections import abc

```



## 4.3 序列操作中的+,+=和extend的区别

```python
from collections import abc
# 初始化的两种方法
a = [1,2]
#a = list()

c = a + [3,4]
print(c)

#+= 就地加
a += [3,4] # a:[1,2,3,4]

#改为元组的话
a += (3,4) #还是一样的效果

# 但是如果 c= a+(3,4) 就会报错
# 所以+= 和 +接收的参数可以为任意的序列类型

a.extend((3,4))
a.extend(range(3))
# extend也是可以的
# 但是不能这样写a = a.extend(range(3))，extend是没有返回值的，是原地操作

a.append([1,2])
# a: [1,2,[1,2]]
# append()和extend是不一样的

a.append((1,2)) # a:[1,2,(1,2)]

```



## 4.4 实现可切片对象

```python
class Group:
    #支持切片操作
    def __init__(self,group_name,company_name,staffs):
        self.group_name = group_name
        self.company_name = company_name
        self.staffs = staffs
    def __reversed__(self):
        self.staffs.reverse()
    def __getitem__(self,item):
        #return self.staffs[item]
        cls = type(self) #取到当前对象的type
        if isinstance(item,slice):
            return cls(group_name=self.group_name,company_name=self.company_name,staffs=staffs[item])
        if isinstance(item,numbers,Integral):
            return cls(group_name=self.group_name,company_name=self.company_name,staffs=staffs[item])
    def __len__(self):
        return len(self.staffs)
    
    
    def __iter__(self):
    	return iter(self.staffs)
    
    def __contains__(self, item):
        if item in self.staffs:
            return True
        else:
            return False
staffs = ["bobby1","imooc","bobby2","bobby3"]
group  = Group(company_name="imooc",group_name="user",staffs=staffs)
sub_group = group[:2]

# 如果想切片后还是一个Group
# queryset[:1]

# 做切片操作python会干什么？
# 如果是切片，那么传递进去的就是slice切片对象，是python解释器自己初始化的，交给getitem使用
# list实现了这个方法所以他就可以这么用
# 对序列操作不一定是slice对象，也可以是group

print(len(group))
if "bobby1" in group:
    print("yes")
```



## 4.5 bisect处理已排序的序列

```python
import bisect
from collections import deque
# 用来处理已排序的序列(升序)，要有序列的概念，不一定是list。
# 二分查找
# inter_list = []
inter_list = deque()
bisect.insort(inter_list, 3)
bisect.insort(inter_list, 2)
bisect.insort(inter_list, 4)
bisect.insort(inter_list, 5)
bisect.insort(inter_list, 7)
bisect.insort(inter_list, 6)
print(inter_list) #是一个排序好的序列，推荐使用bisect来 维护已排序的序列，二分查找效率高

print(bisect.bisect(inter_list,3)) # 查找要插入的位置 3，在（3之后），bisect_right
print(bisect.bisect_left(inter_list,3)) # 2

90,91
90-100 A
80-90  B
```



## 4.6 什么时候不应该使用列表，而是用更好的内置结构

array ,deque

平常用的最多的就是list，其实有更好的选择，用python中其他的内置结构。

```python
# array, deque
# array 其实就是c语言中的数组，数组存储是连续的内存空间，性能很高
# list中有哪些方法？(list是用c语言写的)
# 

import array
# array 和 list一个重要区别
# array只能存放指定的数据类型

my_array = array.array("i") #int
my_array.append(1)
my_array.append("abc") # TypeError

```



## 4.7 列表推导式，生成器表达式，字典推导式

### 列表推导式（也叫作列表生成式）

通过一行代码来生成列表

提取出1-20之间的奇数

```python
odd_list = []
for i in range(21):
    if i%2 == 1:
        odd_list.append(i)
print(odd_list)


#用列表生成式 一行代码完成
odd_list = [i for i in range(21) if i % 2 == 1 ]
print(type(odd_list)) # List


def hadle_item(item):
    return item * item
odd_list = [hadle_item(i) for i in range(21) if i % 2 == 1 ]
```



### 生成器表达式

```python
odd_gen = (i for i in range(21) if i % 2 == 1)
# 变成一个小括号，不是set也不是tuple，它变成了一个生成器
type(odd_list) # generator object

for item in odd_gen:
    print(item)

odd_list = list(odd_gen)
```



### 字典推导式

```python
my_dict = {"bobby1":22,"bobby2":23,"zhaosheng":5}
# 想把key和value做一个颠倒
reversed_dict = {value:key for key,value in my_dict.items()}
print(reversed_dict)
```



### 集合推导式

```python
my_set = {key for key, value in my_dict.items()} #可控性更高
# my_set = set(my_dict.keys())
print(type(my_set)) #Set
print(my_set)
```



## 小结



# 5. 深入set和dict

## 5.1 collections中dict

```python
from collections_abc import Mapping,MutableMapping
a = {}
# dict类型 a实际上不是继承了MutableMapping，他只是实现了一些魔法函数
print(isinstance(a,MutableMapping))
```

## 5.2 dict常用方法



```python
a = {"bobby1":{"company":"imooc"},
     "bobby2":{"company":"imooc2"}
    }
a.clear()

# copy 返回浅拷贝
new_dict = a.copy()
new_dict["bobby1"]["company"] = "imooc3"
# a也一起变化了 嵌套的dict会指向同一个值

# 深拷贝
import copy
new_dict = copy.deepcopy(a)

# fromkyes
new_list = ["bobby1","bobby2"]
new_dict = dict.fromkeys(new_list,{"company":"imooc"})
```



## 5.3 dict子类

```python
#不建议继承list和dict
class Mydict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value*2)
my_dict = Mydict(one=1) # 不生效
my_dict["one"] = 1
print(my_dict)

from collections import UserDict

class Mydict(UserDict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value*2)
my_dict = Mydict(one=1)
# my_dict["one"]=1
print(my_dict)

from collections import defaultdict
my_dict = defaultdict(dict)
my_value = my_dict["bobby"]

```

## 5.4 set和fronzenset(不可变集合) 

set无序 不重复

```python
s = set('abcde')
```



#  7. 元类编程

## 7.4 new和init的区别

```python
class User:
    def __new__(cls,*args,**kwargs): # new在init之前
        #pass
        print("in new")
        return super().__new__(cls)
    def __init__(self):
        print("in init")
        self.name = name
        #pass
```

new是用来控制对象的生成过程，在对象生成之前

init是用来完善对象的

如果new方法不返回对象则不会调用init函数



## 7.5 什么是元类

类也是对象，type是用来创建类的一个类，它实际上是object的子类。

如果要动态的创建一个类

```python
def create_class(name):
    if name == "user":
        class User:
            def __str__(self):
        return "user"
    elif name == "company":
        class Company:
            def __str__(self):
                return "company"
        return Company

#type是可以用来创建类的。 动态创建类
#User = type("User",(),{})

def say(self):
    return "I am user"
    #return self.name

class BaseClass:
    def answer(self):
        return "i am baseclass"

    
    
# 什么是元类，元类是创建类的类
# 对象 <- class（对象） <-type
# type就是元类

class User(metaclass=MetaClass):
    pass 

#python 中类的实例化过程
#type去创建类对象，实例
# 有了metaclass后，创建类实例化的过程中首先寻找metaclass属性
# 如果找到了，就通过metaclass创建user类。


if __name__ == "__main__":
    MyClass = create_class("user")
    my_obj = MyClass()
    print(my_obj) # user
    
    
    # Method 2
    User = type("User",(),{})
    my_obj = User
    print(my_obj) # user
    
    User = type("User",(),{"name":"user","say":say})
    my_obj = User()
    print(my_obj.say()) # I am user
    #
    
    User = type("User",(BaseClass,),{"name":"user","say":say})
    my_obj = User()
    print(my_obj.answer()) # I am user
    #
```



## 7.6 元类实现ORM

需求：定义一个class

ORM: 对类操作，把数据写入表当中，脱离sql语句



```python
import numbers
class IntField:
    def __init__(self, db_column,min_value=None, max_value=None):
        self._value=None
        
        self.min_value = min_value
        self.max_value = max_value
        self.db_cloumn = db_column
        if min_value is not None:
            if isinstance(min_value,numbers.Intergal):
                raise ValueError("min value must be int")
            elif min_value < 0:
                raise ValueError("min value must be positive int")
        if max_value is not None:
            if isinstance(max_value,numbers.Intergal):
                raise ValueError("max_value  must be int")
            elif max_value < 0:
                raise ValueError("max_value  must be positive int")
        if max_value is not None and min_value is not None:
            if min_value > max_value:
                raise ValueError("min must smaller than max")
        
    # 数据描述符
    
    def __get__(self, instance, owner):
        return self._value
    def __set__(self, instance, value):
        if not isinstance(value,numbers.Integral):
            raise ValueError("int value need")
        if value<self.max_value or value > self.max_value:
            raise ValueError("value must between min_value and max_value")

class CharField:
    def __init__(self, db_column, value, max_length=None):
        self._value = value
        self.db_cloumn = db_column
        
        if max_length is None:
            raise ValueError("you must spcify max_length for charfiled")
        self.max_length = max_length
    	
        
    def __get__(self, instance, owner):
        return self._value
    def __set__(self,instance,value):
        if not isinstance(value,str):
            raise ValueError("int value need")
        if len(value)>self.max_length:
            raise ValueError("value len excess len of max_length")
        self._value = value

class ModelMetaClass(type):
    def __new__(cls, *args, **kwargs):
        
        
        
class User:
    name = CharField(db_column="",max_length=10)
    age = IntField(db_column="",min_value=0,max_value=100)
    
    class Meta:
        db_table = ""

if __name__ == "__main__":
    user = User()
    user.name = "bobby"
    user.age= 28
    user.save()
```

