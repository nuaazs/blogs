# Python通过钩子远程加载模块

## 问题

你想自定义Python的import语句，使得它能从远程机器上面透明的加载模块。

## 解决方案

有很多种方法可以做这个， 不过为了演示的方便，我们开始先构造下面这个Python代码结构：

```
testcode/
    spam.py
    fib.py
    grok/
        __init__.py
        blah.py
```

这些文件的内容并不重要，不过我们在每个文件中放入了少量的简单语句和函数， 这样你可以测试它们并查看当它们被导入时的输出。例如：

```python
# spam.py
print("I'm spam")

def hello(name):
    print('Hello %s' % name)

# fib.py
print("I'm fib")

def fib(n):
    if n < 2:
        return 1
    else:
        return fib(n-1) + fib(n-2)

# grok/__init__.py
print("I'm grok.__init__")

# grok/blah.py
print("I'm grok.blah")
```

这里的目的是允许这些文件作为模块被远程访问。 也许最简单的方式就是将它们发布到一个web服务器上面。在testcode目录中像下面这样运行Python：

```bash
cd testcode
python -m http.server 15000
>>> Serving HTTP on 0.0.0.0 port 15000 ...
```

服务器运行起来后再启动一个单独的Python解释器。 确保你可以使用 `urllib` 访问到远程文件。例如：

```python
from urllib.request import urlopen
u = urlopen('http://localhost:15000/fib.py')
data = u.read().decode('utf-8')
print(data)
```

从这个服务器加载源代码是接下来本节的基础。 为了替代手动的通过 `urlopen()` 来收集源文件， 我们通过自定义import语句来在后台自动帮我们做到。

加载远程模块的第一种方法是创建一个显式的加载函数来完成它。例如：

```python
import imp
import urllib.request
import sys
def load_module(url):
    u =urllib.request.urlopen(url)
    source = u.read().decode('utf-8')
    mod = sys.modules.setdefault(url, imp.new_module(url))
    code = compile(source,url,'exec')
    mod.__file__ = url
    mod.__package__ = ''
    exec(code,mod.__dict__)
    return mod
```

这个函数会下载源代码，并使用 `compile()` 将其编译到一个代码对象中， 然后在一个新创建的模块对象的字典中来执行它。下面是使用这个函数的方式：

```python
>>> fib = load_module('http://localhost:15000/fib.py')
I'm fib
>>> fib.fib(10)
89
>>> spam = load_module('http://localhost:15000/spam.py')
I'm spam
>>> spam.hello('Guido')
Hello Guido
>>> fib
<module 'http://localhost:15000/fib.py' from 'http://localhost:15000/fib.py'>
>>> spam
<module 'http://localhost:15000/spam.py' from 'http://localhost:15000/spam.py'>
>>>
```