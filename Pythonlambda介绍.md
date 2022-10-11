# Python lambda介绍
## 1. lambda是什么？
```python
g = lambda x:x+1

# g(1) -> 2
# g(2) -> 3
```
Lambda作为一个表达式定义了一个匿名函数， 上例的代码`x`为入口参数， `x+1`为函数体， 用函数表示为：
```python
def g(x):
	return x+1
```
非常容易理解，在这里lambda简化了函数定义的书写形式。是代码更为简洁，但是使用函数的定义方式更为直观，易理解。

Python中，也有几个定义好的全局函数方便使用的，`filter`, `map`, `reduce` ...
```python
>>> foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
>>>
>>> print filter(lambda x: x % 3 == 0, foo)
[18, 9, 24, 12, 27]
>>>
>>> print map(lambda x: x * 2 + 10, foo)
[14, 46, 28, 54, 44, 58, 26, 34, 64]
>>>
>>> print reduce(lambda x, y: x + y, foo)
139
```
上面例子中的`map`的作用，非常简单清晰。但是，Python是否非要使用`lambda`才能做到这样的简洁程度呢？在对象遍历处理方面，其实Python的`for..in..if`语法已经很强大，并且在易读上胜过了`lambda`。
比如上面map的例子，可以写成：
```python
print[x for x in foo if x %3 ==0 ]
```
同样也是比lambda的方式更容易理解。

## 2. 为什么使用lambda？
```python
processFunc = collapse and (lambda s:" ".join(s.split())) or (lambda s:s)
```
在`Visual Basic`,你可能想要创建一个函数， 接受一个字符串参数和一个`collapse`参数，并使用if语句确定是否压缩空白， 然后返回相应的值。
这种方法是低效的， 因为函数可能需要处理每一种可能的情况， 每次调用它， 它将不得不在给出想要的东西之前， 判断是否要压缩空白。
在Python中， 你可以将决策逻辑拿到函数外面， 而定义一个裁剪过的`lambda`函数，更为高效和优雅。

## 总结
1. lambda 定义了一个匿名函数
2. lambda 并不会带来程序运行效率的提高，只会使代码更简洁。
3. 如果可以使用for...in...if来完成的，坚决不用lambda。
4. 如果使用lambda，lambda内不要包含循环，如果有，我宁愿定义函数来完成，使代码获得可重用性和更好的可读性。