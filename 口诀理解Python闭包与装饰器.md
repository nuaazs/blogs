# 口诀理解Python闭包与装饰器


今天在B站看到一个[视频](https://www.bilibili.com/video/BV1ZJ411y7Te?from=search&seid=10099699472686434241)，解释的很好，总结一下。

## 作用域理解
> 作用域，是栋楼，下楼套上楼；
> 读变量，往下搜，一直到一楼；
> 改变量，莫下楼，除非你放global；



### 读取变量

```python
msg = "我是全局变量msg"    # 一楼

def secondFloor():		# 二楼
	print(msg)
	def thirdFloor():	# 三楼
        print(msg)
    return thirdFloor
sf = secondFloor()
sf()

# 输出 "我是全局变量msg"
```



如果在二楼修改。

```python
msg = "我是全局变量msg"    # 一楼

def secondFloor():		# 二楼
    
	msg = "我是secondFloor里的msg"
	def thirdFloor():	# 三楼
        print(msg)
    return thirdFloor
sf = secondFloor()
sf()

# 输出 "我是secondFloor里的msg"
```



所以python是从上往下的读取变量。读变量，往下搜，一直到一楼；



### 如果修改字符串

```python
msg = "我是全局变量msg"    # 一楼

def secondFloor():		# 二楼
	msg += "haha"
	def thirdFloor():	# 三楼
        print(msg)
    return thirdFloor
sf = secondFloor()
sf()

#报错
```

可以在读取，但是不能改。如果要改必须用：

```python
msg = "我是全局变量msg"    # 一楼

def secondFloor():		# 二楼
    global
	msg += "haha"
	def thirdFloor():	# 三楼
        print(msg)
    return thirdFloor
sf = secondFloor()
sf()

#报错
```



## 闭包：至少两层楼，楼下变量管上楼，return上楼不动手

```python
# 闭包

def secondFloor():		# 二楼
	msg = "我是secondFloor"
	def thirdFloor():	# 三楼
        print(msg)
    return thirdFloor  # 不动手 == 没有括号
```



```python
# 接收
sf = secondFloor()
sf()   # 接受后再动手
```










## 装饰器：客人空手来，还得请上楼；干啥都同意，有参给上楼

装饰器是在闭包基础上的，先满足闭包的要求。



老板：能不能记录一下函数`origin()`运行的时间。

答：用装饰器



### 客人空手来，还得请上楼

```python
# 闭包

def secondFloor(func):		# 用形参func 是因为可能不只一个函数

	def thirdFloor():		# 三楼
        print("start")
        func()				# 请上楼， 括号 == 干啥都同意
        print("end")
    return thirdFloor  # 不动手 == 没有括号

def origin():
    print("我是原函数")
    
# 使用
sf = secondFloor(origin)


# start
# 我是原函数
# end
```

### 语法糖

```python
@secondFloor
def origin():
	print("我是源函数")

origin()
```



### 传参

客人是空手来的，要在上楼给一个形参：

```python
# 闭包

def secondFloor(func):		# 用形参func 是因为可能不只一个函数

	def thirdFloor(arg):		# 三楼
        print("start")
        func(arg)				# 请上楼， 括号 == 干啥都同意
        print("end")
    return thirdFloor  # 不动手 == 没有括号

def origin():
    print("我是原函数")
    
# 使用
info = "hi"
@secondFloor
def origin(info):
	print(info)

origin()
```

