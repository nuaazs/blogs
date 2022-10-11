## 场景

通过配置文件获取服务器上配置的服务名及运行端口号，编写python脚本检测服务上服务是否在运行？

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# fileName: config.py
# 服务配置
class config:
    serviceList = 'service1,service2,service3'
    service1 = '服务1'
    service1Port = 8001
    service2 = '服务2'
    service2Port = 8002
    service3 = '服务3'
    service3Port = 8003
```

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# fileName: envCheck.py
import socket
from config import config

config = config
serviceList = config.serviceList

# 判断某端口服务是否运行
def portCheck(host, port):
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.settimeout(1)
    try:
        sk.connect((host, port))
        # print '在服务器 %s 上服务端口 %d 的服务正在运行!' % (host, port)
        return True
    except Exception:
        # print '在服务器 %s 上服务端口 %d 的服务未运行!' % (host, port)
        return False
    sk.close()

# 基础服务运行状态检测
def envCheck():
    for serviceName in serviceList.split(','):
        host = '127.0.0.1'  # 必须为字符串格式,如:'127.0.0.1'
        servicePort = ''.join(['config.',serviceName,'Port'])
        port = eval(servicePort)  # 端口必须为数字
        if portCheck(host, port):
            print u"在%s服务器上服务端口为 %s 的 %s 服务正在运行......" % (host, port, serviceName)
        else:
            print u"在%s服务器上服务端口为 %s 的 %s 服务未运行!" % (host, port, serviceName)


if __name__ == "__main__":
    envCheck()
```

这个里面使用到了将字符串作为变量名的方式从配置中获取服务端口，下面我们具体看下除了这种方式以外还有哪些方式可以实现

一共有三种实现方法：

### 法一：locals()

```python
servicePort = ''.join(['config.',serviceName,'Port'])
port = locals()[servicePort]
print("%s:%d"%(serviceName,port))
```

#### 描述

**locals()** 函数会以字典类型返回当前位置的全部局部变量。

对于函数, 方法, lambda 函式, 类, 以及实现了 __call__ 方法的类实例, 它都返回 True。

#### 语法

locals() 函数语法：

```
locals()
```

#### 参数

- 无

#### 返回值

返回字典类型的局部变量。

#### 实例

以下实例展示了 locals() 的使用方法：

```python
>>>def runoob(arg):    # 两个局部变量：arg、z
...     z = 1
...     print (locals())
... 
>>> runoob(4)
{'z': 1, 'arg': 4}      # 返回一个名字/值对的字典
>>>
```



### 法二：vars()

```python
port = vars()[servicePort]
```

#### 描述

**vars()** 函数返回对象object的属性和属性值的字典对象。

#### 语法

vars() 函数语法：

```
vars([object])
```

#### 参数

- object -- 对象

#### 返回值

返回对象object的属性和属性值的字典对象，如果没有参数，就打印当前调用位置的属性和属性值 类似 locals()。

#### 实例

以下实例展示了 vars() 的使用方法：

```python
>>>print(vars())
{'__builtins__': <module '__builtin__' (built-in)>, '__name__': '__main__', '__doc__': None, '__package__': None}
>>> class Runoob:
...     a = 1
... 
>>> print(vars(Runoob))
{'a': 1, '__module__': '__main__', '__doc__': None}
>>> runoob = Runoob()
>>> print(vars(runoob))
{}
```







### 法三： eval()

```python
port = eval(servicePort)
```



#### 描述

eval() 函数用来执行一个字符串表达式，并返回表达式的值。

#### 语法

以下是 eval() 方法的语法:

```
eval(expression[, globals[, locals]])
```

#### 参数

- expression -- 表达式。
- globals -- 变量作用域，全局命名空间，如果被提供，则必须是一个字典对象。
- locals -- 变量作用域，局部命名空间，如果被提供，可以是任何映射对象。

#### 返回值

返回表达式计算结果。

------

#### 实例

以下展示了使用 eval() 方法的实例：

```python
>>>x = 7
>>> eval( '3 * x' )
21
>>> eval('pow(2,2)')
4
>>> eval('2 + 2')
4
>>> n=81
>>> eval("n + 4")
85
```

