类装饰器顾名思义用类写的装饰器，首先看类的定义

```python
class A:
    def __init__(self,arg = 'a'):
        print(arg)
    def __call__(self,arg): # 使对象可以像函数一样调用
        print(arg)
        self.doSomething()
    def doSomething(self):
        print("do some thing")
a = A(1) # 实例化一个对象
a(2) # 把对象当函数来调用，需要实现__call__方法
```

上面的例子中我们看到类可以向函数一样调用，下面我们就实现一个类的装饰器:

```python
from functools import wraps
class A:
    def __call__(self,func):
        @wraps(func) # 会把func信息导入到wrapper函数中
        def wrapper(*args,**kw):
            self.doSomething()
            return func(*args,**kw)
        return wrapper
    def doSomething(self):
        print('doSomething')

@A() #相当于 func=A()(func) ，A实现了__call__方法所以A()产生的实例可以调用
def func():
    print("func")
func()
'''
结果：
doSomething
func
'''
```

有了装饰器后为什么还要发明 一个类装饰器呢？我们看下面一段代码体会他的优点：

```python
from functools import wraps
class A:
    def __call__(self,func):
        @wraps(func)
        def wrapper(*args,**kw):
            self.doSomething()
            return func(*args,**kw)
        return wrapper
class B(A): #B继承A类，重写A类的doSomething方法
    def doSomething(self):
        print("B doSomething")
        
@B()
def func():
    print("func")
func()
'''
结果：
B doSomething
func
'''
```

总结：使用类装饰器可以通过继承的方式扩展类装饰器的行为。

