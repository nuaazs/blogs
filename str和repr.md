Python的字符串既可以单引号也可以双引号，无区别。
如果字符串本身含有单引号或者双引号，则需要对字符串进行处理。处理方法有两种，一是使用不同的引号将字符串括起来，还有一种就是使用转义字符。

Python使用`+`号进行两个字符串的拼接。但是字符串和数值不能直接使用`+`号进行拼接，需要使用`str`或者`repr`对数值类型进行转化。
`str`与`repr`均可以将数值转为字符串，但是区别为：
1. `str`为Python的内置类型，而`repr()`为一个函数，还有就是`repr`会以Python表达式的方式输出字符串。

```python
string = 'zhao sheng!'
p = 666
print(string+str(p)) # zhao sheng!666
print(string+repr(p)) # zhao sheng!666
print(string) # zhao sheng!
pinrt(repr(string)) # 'zhao sheng!'zhao sheng!
print(type(repr(string))) # <class 'str'>
print(type(string)) # <class 'str'>
```

可以看出，字符串直接输出是不带有引号的，而经过`repr`输出后是**带有引号**的。其实repr的作用就相当于`''`的作用。
repr官方解释：
> Return a string containing a printable representation of an object.For many types, this function makes an attempt to return a string that would yield an object with the same value when passed to eval().otherwise the representation is a string enclosed in angle brackets that contains the name of the type of the object together with additional information often including the name and address of the object.A class can control what this function returns for its instances by defining a __repr__() method.


> 返回一个包含可打印展示出来对象的字符串。对于许多类型来说，当字符串传到eval()函数中时，这个函数尝试返回一个与对象值相同的字符串。否则就显示一个被一对尖括号包裹的字符串，这个字符串包含对象类型的名字，以及其他一些额外的信息，一般为这个对象的名字和地址。

eval函数
```python
x = 1
str = 'x+1'
print(str) # x+1
print(repr(str)) # 'x+1'
eval(str) # 2
eval(repr(str)) # 'x+1'
```
可以看出如果不使用repr，字符串x+1就会在eval函数中被当成表达式计算为2，而使用repr函数就不会被当成表达式计算，即
```python
eval(repr(obj)) == obj
```
2. 关于类和对象
```python
class user:
    def get_user(self):
        return "hello zhaosheng"
u = user()
print(repr(user)) # <class '__main__.user'>
print(repr(u)) # <__main__.user object at xxxxx>
```
如果自定义一个user类，将类传到`repr`函数中，则会返回一对尖括号包裹的字符串，从前往后依次为类型，名称。
将user的对象u传到repr函数中就会多返回对象在内存中的地址。这个方法和直接`print(user)`,`print(u)`的效果是一样的。

3. 关于`__repr__()`方法
```python
class user:
    def get_user(self):
        return "hello zhaosheng"
    def __repr__(self):
        return "hello zhaosheng repr"
u = user()
print(repr(user)) # <class '__main__.user'>
print(repr(u)) #  hello zhaosheng repr
```
在一个类中定义了`__repr__()`函数，将类的对象传到`repr`函数中就会自动执行`__repr__`函数打印字符串，不再打印尖括号包裹的对象信息的字符串。此时执行`print(u)`也一样，和定义`__str__()`方法效果是一样的。


4. `__str__` 和 `__repr__` 的差别是什么
```python
import datatime
today = datetime.date.today()
str(today) # '2021-08-07'
repr(today) # 'datetime.date(2021, 8, 7)'
```
`__str__` 的返回结果可读性强。
`__repr__` 的返回结果应更准确。目的在于调试，便于开发者使用。将 `__repr__ `返回的方式直接复制到命令行上，是可以直接执行的。

5. 自己写class的时候要加`__repr__`
如果你没有添加 __str__ 方法，Python 在需要该方法但找不到的时候，它会去调用 __repr__ 方法。能保证类到字符串始终有一个有效的自定义转换方式。