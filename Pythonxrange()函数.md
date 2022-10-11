## 描述

**xrange()** 函数用法与`range`完全相同，所不同的是生成的不是一个数组，而是一个生成器。

## 语法

xrange 语法：

```python
xrange(stop)
xrange(start, stop[, step])
```

参数说明：

- start: 计数从 start 开始。默认是从 0 开始。例如 `xrange(5) `等价于` xrange(0， 5)`
- stop: 计数到 stop 结束，但不包括 stop。例如：`xrange(0， 5)` 是` [0, 1, 2, 3, 4] `没有 `5`
- step：步长，默认为1。例如：`xrange(0， 5) 等价于 xrange(0, 5, 1)`

## 返回值

返回生成器。

## 实例

以下实例展示了 xrange 的使用方法：

```python
>>>xrange(8)
# xrange(8)

list(xrange(8))
# [0, 1, 2, 3, 4, 5, 6, 7]

xrange(3,5)
# xrange(3, 5)

list(xrange(3,5))
# [3,4]

range(3,5)
# [3,4]

xrange(0,6,2)
# xrange(0.6.2)

list(xrange(0,6,2))
# [0, 2, 4]
```

 由上面的示例可以知道： 要生成很大的数字序列的时候，用`xrange`会比`range`性能优很多，因为不需要一上来就开辟一块很大的内存空间 ，这两个基本上都是在循环的时候用：

```python
for i in range(0,100):
    print(i)

for i in xrange(0,100):
    print(i)
```

这两个输出的结果都是一样的，实际上有很多不同，`range`会直接生成一个list对象：

```python
a = range(0,100)
print(type(a))
# <type 'list'>

print(a)
# ....

print(a[0],a[1])
# 0 1
```

 而`xrange`则不会直接生成一个`list`，而是每次调用返回其中的一个值：

```python
a = xrange(0,100)
print(type(a))
# <type 'xrange'>

print(a)
# xrange(100)

print(a[0],a[1])
# 0 1
```

所以 `xrange`做循环的性能比`range`好 ，尤其是返回很大的时候，尽量用`xrange`吧，除非你是要返回一个列表。