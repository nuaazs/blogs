### 静态方法

我们知道在其他语言中静态方法一般使用static修饰，静态方法的主要特点是不需要new出对象，直接通过类名就可以访问，也可以通过对象访问。需要使用`staticmethod`装饰器装饰方法

```python
class A:
    @staticmethod
    def staticfunc():
        print("A")
A.staticfunc() #A
```

### 类方法

类方法和静态方法类似，也可以直接通过类名访问，不过要使用classmethod装饰方法，而且参数第一个类本身,为了方便的操作类本身的属性和方法。

```python
class B:
    @classmethod
    def classfunc(cls):
        cls.a = 10
B.classfunc()
print(B.a) #10
```



### 对比

静态方法和类方法都可以通过类名访问，那直接用静态方法不就好了？举个例子看一下

```python
class A:
    @staticmethod
    def staticfunc():
        A.a = 10
A.staticfunc()
print(A.a)#10

class B:
    @classmethod
    def classfunc(cls):
        cls.b = 20;
B.classfunc()
print(B.b)#20
```

通过上面的例子我们可以看到静态方法也可以操作类本身，为什么还要在发明一个类方法？上面例子我们观察到，静态方法是通过类名来操作类属性的写死在程序中，而类方法是通过参数来操作类属性的，如果子类继承了使用静态方法的类，那么子类继承的静态方法还是在操作父类，子类需要重新定义静态方法才能操作子类。