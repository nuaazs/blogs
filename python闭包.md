## 1. 概念
闭包并不只是一个python中的概念，在函数式编程语言中应用较为广泛。理解python中的闭包一方面是能够正确的使用闭包，另一方面可以好好体会和思考闭包的设计思想。

> 在计算机科学中，闭包（英语：Closure），又称词法闭包（Lexical Closure）或函数闭包（function closures），是引用了自由变量的函数。这个被引用的自由变量将和这个函数一同存在，即使已经离开了创造它的环境也不例外。所以，有另一种说法认为闭包是由函数和与其相关的引用环境组合而成的实体。闭包在运行时可以有多个实例，不同的引用环境和相同的函数组合可以产生不同的实例。



简单来说就是一个函数定义中引用了函数外定义的变量，并且该函数可以在其定义环境外被执行。这样的一个函数我们称之为`闭包`。实际上闭包可以看做一种更加广义的函数概念。因为其已经不再是传统意义上定义的函数。

根据我们对编程语言中函数的理解，大概印象中的函数是这样的：

> 程序被加载到内存执行时，函数定义的代码被存放在代码段中。函数被调用时，会在栈上创建其执行环境，也就是初始化其中定义的变量和外部传入的形参以便函数进行下一步的执行操作。当函数执行完成并返回函数结果后，函数栈帧便会被销毁掉。函数中的临时变量以及存储的中间计算结果都不会保留。下次调用时唯一发生变化的就是函数传入的形参可能会不一样。函数栈帧会重新初始化函数的执行环境。

C++中有`static`关键字，函数中的`static`关键字定义的变量**独立于函数之外**，而且会**保留函数中值的变化**。函数中使用的全局变量也有类似的性质。

但是闭包中引用的函数定义之外的变量是否可以这么理解呢？但是如果函数中引用的变量既不是全局的，也不是静态的(python中没有这个概念)。应该怎么正确的理解呢？



## 2.闭包初探

为了说明闭包中引用的变量的性质，可以看一下下面的这个例子：

```python
def outer_func():
    loc_list = []
    def inner_func(name):
        loc_list.append(len(loc_list)+1)
        print("%s loc_list = %s" % (name, loc_list))
    return inner_func


clo_func_0 = outer_func()
clo_func_0('clo_func_0')
clo_func_0('clo_func_0')
clo_func_0('clo_func_0')
clo_func_1 = outer_func()
clo_func_1('clo_func_1')
clo_func_0('clo_func_0')
clo_func_1('clo_func_1')
```

从上面这个简单的例子应该对闭包有一个直观的理解了。运行的结果也说明了闭包函数中引用的父函数中local variable既不具有C++中的全局变量的性质也没有static变量的行为。

在python中我们称上面的这个loc_list为闭包函数inner_func的一个自由变量(free variable)。

> If a name is bound in a block, it is a local variable of that block. If a name is bound at the module level, it is a global variable. (The variables of the module code block are local and global.) If a variable is used in a code block but not defined there, it is a *free variable*.

在这个例子中我们至少可以对闭包中引用的自由变量有如下的认识：

- 闭包中的引用的自由变量只和具体的闭包有关联，闭包的每个实例引用的自由变量互不干扰。
- 一个闭包实例对其自由变量的修改会被传递到下一次该闭包实例的调用。

由于这个概念理解起来并不是那么的直观，因此使用的时候很容易掉进陷阱。



## 3.闭包陷阱

下面先来看一个例子：

```python
def my_func(*args):
    fs = []
    for i in xrange(3):
        def func():
            return i * i
        fs.append(func)
    return fs

fs1, fs2, fs3 = my_func()
print (fs1())
print (fs2())
print (fs3())
```

 上面这段代码可谓是典型的**错误使用闭包**的例子。程序的结果并不是我们想象的结果0，1，4。实际结果全部是4。

这个例子中，`my_func`返回的并不是一个闭包函数，而是一个包含三个闭包函数的一个list。这个例子中比较特殊的地方就是返回的所有闭包函数均引用父函数中定义的同一个自由变量。

但这里的问题是为什么`for`循环中的变量变化会影响到所有的闭包函数？尤其是我们上面刚刚介绍的例子中明明说明了同一闭包的不同实例中引用的自由变量互相没有影响的。而且这个观点也绝对的正确。

那么问题到底出在哪里？应该怎样正确的分析这个错误的根源。

其实问题的关键就在于在返回闭包列表`fs`之前`for`循环的变量的值已经发生改变了，而且这个改变会影响到所有引用它的内部定义的函数。因为在函数my_func返回前其内部定义的函数并不是闭包函数，只是一个内部定义的函数。

当然这个内部函数引用的父函数中定义的变量也不是自由变量，而只是当前block中的一个`local variable`。



```python
def my_func(*args):
    fs = []
    j = 0
    for i in xrange(3):
        def func():
            return j * j
        fs.append(func)
    j = 2
    return fs
```

这段代码逻辑上与之前的例子是等价的。这里或许更好理解一点，因为在内部定义的函数`func`实际执行前，对局部变量j的任何改变均会影响到函数func的运行结果。

函数`my_func`一旦返回，那么内部定义的函数`func`便是一个闭包，其中引用的变量j成为一个只和具体闭包相关的自由变量。后面会分析，这个自由变量存放在Cell对象中。

使用`lambda`表达式重写这个例子：

```python
def my_func(*args):
    fs = []
    for i in xrange(3):
        func = lambda : i*i
        fs.append(func)
    return fs
```



经过上面的分析，我们得出下面一个重要的经验：

> 返回闭包中不要引用任何循环变量，或者后续会发生变化的变量。这条规则本质上是在返回闭包前，闭包中引用的父函数中定义变量的值可能会发生不是我们期望的变化。

### 正确写法

```python
def my_func(*args):
    fs = []
    for i in xrange(3):
    	def func(_i = i):
            return _i * i
        fs.append(func)
    return fs
```

或者：

```python
def my_func(*args):
```

