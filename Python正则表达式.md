## 一、re模块简介

这是一个Python处理文本的**标准库**。

```python
import re
```

[re模块官方文档](https://link.zhihu.com/?target=https%3A//docs.python.org/zh-cn/3.8/library/re.html)
[re模块库源码](https://link.zhihu.com/?target=https%3A//github.com/python/cpython/blob/3.8/Lib/re.py)

## 二、re模块常量

常量即表示不可更改的变量，一般用于做标记。re模块中有9个常量，常量的值都是`int`类型！

![img](https://pic4.zhimg.com/80/v2-192d31a9de5470456c61e96d9a9d1aab_720w.jpg)


上图我们可以看到，所有的常量都是在`RegexFlag枚举类`来实现，这是在Python 3.6做的改版。在Python 3.6以前版本是直接将常量写在re.py中，使用枚举的好处就是方便管理和使用！

![img](https://pic2.zhimg.com/80/v2-d442174c6eb02902f5d3e209b9939dc1_720w.jpg)



下面我们来快速学习这些常量的作用及如何使用他们，按常用度排序！



### 2.1 IGNORECASE

**语法：** `re.IGNORECASE` 或简写为 `re.I`

**作用：** 进行忽略大小写匹配。

**代码案例：**

![img](https://pic1.zhimg.com/80/v2-6f44888e6114a98b2202a92a8adb7d5c_720w.jpg)


在默认匹配模式下**大写字母B**无法匹配**小写字母b**，而在 忽略大小写 模式下是可以的。



### 2.2 ASCII

**语法：** `re.ASCII` 或简写为 `re.A`

**作用：** 顾名思义，ASCII表示ASCII码的意思，让 `\w`, `\W`, `\b`, `\B`, `\d`, `\D`, `\s` 和 `\S` 只匹配ASCII，而不是Unicode。

**代码案例：**

![img](https://pic3.zhimg.com/80/v2-1a6b313162711f831dc781303e32f172_720w.jpg)


在默认匹配模式下`\w+`匹配到了所有字符串，而在**ASCII**模式下，只匹配到了a、b、c（ASCII编码支持的字符）。

注意：这只对字符串匹配模式有效，对字节匹配模式无效。



### 2.3  DOTALL

**语法：** `re.DOTALL` 或简写为 `re.S`

**作用：** DOT表示`.`，ALL表示所有，连起来就是`.`匹配所有，包括换行符`\n`。**默认模式下`.`是不能匹配行符`\n`的**。

**代码案例：**

![img](https://pic4.zhimg.com/80/v2-b4ba1028a21217a29497952101c8947b_720w.jpg)


在默认匹配模式下`.`并没有匹配换行符`\n`，而是将字符串分开匹配；而在**re.DOTALL**模式下，换行符`\n`与字符串一起被匹配到。

注意：**默认匹配模式下`.`并不会匹配换行符`\n`**。



### 2.4 MULTILINE

**语法：** `re.MULTILINE` 或简写为 `re.M`

**作用：** 多行模式，当某字符串中有换行符`\n`，默认模式下是不支持换行符特性的，比如：行开头 和 行结尾，而多行模式下是支持匹配行开头的。

**代码案例：**

![img](https://pic4.zhimg.com/80/v2-67991439abbbb34c38b239af08793bbf_720w.jpg)


正则表达式中`^`表示匹配行的开头，默认模式下它只能匹配字符串的开头；而在多行模式下，它还可以匹配 换行符`\n`后面的字符。

注意：正则语法中`^`匹配行开头、`\A`匹配字符串开头，单行模式下它两效果一致，多行模式下`\A`不能识别`\n`。



### 2.5 VERBOSE

**语法：** `re.VERBOSE` 或简写为 `re.X`

**作用：** 详细模式，可以在正则表达式中加注解！

**代码案例：**

![img](https://pic1.zhimg.com/80/v2-cfe888f6f0db818d416e343ff51beb08_720w.jpg)


默认模式下并不能识别正则表达式中的注释，而详细模式是可以识别的。

当一个正则表达式十分复杂的时候，详细模式或许能为你提供另一种注释方式，但它不应该成为炫技的手段，建议谨慎考虑后使用！



### 2.6 LOCALE

**语法：** `re.LOCALE` 或简写为 `re.L`

**作用：** 由当前语言区域决定 `\w`, `\W`, `\b`, `\B` 和大小写敏感匹配，这个标记只能对byte样式有效。**这个标记官方已经不推荐使用**，因为语言区域机制很不可靠，它一次只能处理一个 "习惯”，而且只对8位字节有效。

**注意：** 不推荐使用



### 2.7 UNICODE

**语法：** `re.UNICODE` 或简写为 `re.U`

**作用：** 与 ASCII 模式类似，匹配unicode编码支持的字符，但是 Python 3 默认字符串已经是Unicode，所以有点冗余。



### 2.8 DEBUG

**语法：** `re.DEBUG`

**作用：** 显示编译时的debug信息。

**代码案例：**

![img](https://pic2.zhimg.com/80/v2-b4eeb795e2a9480935d4fe5233632615_720w.jpg)



### 2.9 TEMPLATE

**语法：** re.TEMPLATE 或简写为 re.T

**作用：** 猪哥也没搞懂TEMPLATE的具体用处，源码注释中写着：disable backtracking(禁用回溯)，有了解的同学可以留言告知！

![img](https://pic3.zhimg.com/80/v2-2177dfe09c9b6d797a1882c322e976c2_720w.jpg)



### 2.10  常量总结



1. 9个常量中，前5个（IGNORECASE、ASCII、DOTALL、MULTILINE、VERBOSE）有用处，两个（LOCALE、UNICODE）官方不建议使用、两个（TEMPLATE、DEBUG）试验性功能，不能依赖。
2. 常量在re常用函数中都可以使用，查看源码可得知。

![img](https://pic3.zhimg.com/80/v2-0eb3458cd7289709bb70add59efcd962_720w.jpg)



1. 常量可叠加使用，因为常量值都是2的幂次方值，所以是可以叠加使用的，叠加时请使用 `|` 符号，请勿使用`+` 符号！

![img](https://pic3.zhimg.com/80/v2-6656131dde9861872b16e4cf6be1c43e_720w.jpg)





最后来一张思维导图总结一下re模块中的常量吧，**需要高清图或者xmind文件的同学可在「裸睡的猪」后台回复：re** 获取。

![img](https://pic4.zhimg.com/80/v2-8739802053ac450de80b35f6bcf90acb_720w.jpg)



## 三、re模块函数

re模块有12个函数，猪哥将以功能分类来讲解；这样更具有比较性，同时也方便记忆。

### 3.1 查找一个匹配项

查找并返回一个匹配项的函数有3个：**search、match、fullmatch**，他们的区别分别是：



1. **search：** 查找任意位置的匹配项
2. **match：** 必须从字符串开头匹配
3. **fullmatch：** 整个字符串与正则完全匹配



我们再来根据实际的代码案例比较：

**案例1:**

```python
import re
text = "南n京航u空航天a大学a"
pattern = r'航u空'
print('search(任意位置):',re.search(pattern,text).group()) # 航u空
print('match(从头开始):',re.match(pattern,text)) # None
print('fullmatch(必须完全匹配):',re.fullmatch(pattern,text)) # None
```

**案例2:**

```python
import re
text = "南n京航u空航天a大学a"
pattern = r'南n京'
print('search(任意位置):',re.search(pattern,text).group()) # 南n京
print('match(从头开始):',re.match(pattern,text)) # <re.Match object; span=(0, 3), match='南n京'>
print('fullmatch(必须完全匹配):',re.fullmatch(pattern,text)) # None
```

**注意：查找 一个匹配项 返回的都是一个匹配对象（Match）。**



### 3.2 查找多个匹配项

讲完查找一项，现在来看看查找多项吧，查找多项函数主要有：**findall函数** 与 **finditer函数**：

1. **findall：** 从字符串任意位置查找，**返回一个列表**
2. **finditer**：从字符串任意位置查找，**返回一个迭代器**

两个方法基本类似，只不过一个是返回列表，一个是返回迭代器。我们知道列表是一次性生成在内存中，而迭代器是需要使用时一点一点生成出来的，内存使用更优。

![img](https://pic1.zhimg.com/80/v2-4c2d71ce1088c2637266ab35a50c86c4_720w.jpg)


如果可能存在大量的匹配项的话，建议使用**finditer函数**，一般情况使用**findall函数**基本没啥影响。

### 3.3 分割

**re.split(pattern, string, maxsplit=0, flags=0)** 函数：用 **pattern** 分开 string ， **maxsplit**表示最多进行分割次数， **flags**表示模式，就是上面我们讲解的常量！



![img](https://pic4.zhimg.com/80/v2-c08fa8e4a6a445a37e36761ea0a79d27_720w.jpg)


**注意：`str`模块也有一个 split函数 ，那这两个函数该怎么选呢？**
str.split函数功能简单，不支持正则分割，而re.split支持正则。

**关于二者的速度如何？** 猪哥实际测试一下，在相同数据量的情况下使用`re.split`函数与`str.split`函数**执行次数** 与 **执行时间** 对比图：

![img](https://pic4.zhimg.com/80/v2-a0dfb7664a05c07f7b81d12702b0c52f_720w.jpg)


通过上图对比发现，1000次循环以内`str.split`函数更快，而循环次数1000次以上后`re.split`函数明显更快，而且次数越多差距越大！

**所以结论是：在 不需要正则支持 且 数据量和数次不多 的情况下使用`str.split`函数更合适，反之则使用`re.split`函数。**

注：具体执行时间与测试数据有关！

### 3.4 替换

替换主要有**sub函数** 与 **subn函数**，他们功能类似！

先来看看**sub函数**的用法：

**re.sub(pattern, repl, string, count=0, flags=0)** 函数参数讲解：repl替换掉string中被pattern匹配的字符， count表示最大替换次数，flags表示正则表达式的常量。

值得注意的是：**sub函数**中的入参：**repl替换内容既可以是字符串，也可以是一个函数哦！** 如果repl为函数时，只能有一个入参：Match匹配对象。



![img](https://pic4.zhimg.com/80/v2-acb374226ed45d39a4a7ba1e0d50712f_720w.jpg)



**re.subn(pattern, repl, string, count=0, flags=0)** 函数与 **re.sub函数** 功能一致，只不过返回一个元组 (字符串, 替换次数)。

![img](https://pic3.zhimg.com/80/v2-b057eee5f88e4427c78df37d74501bc2_720w.jpg)



### 3.5 编译正则对象

**compile函数** 与 **template函数** 将正则表达式的样式编译为一个 正则表达式对象 （正则对象Pattern），这个对象与re模块有同样的正则函数（后面我们会讲解Pattern正则对象）。

![img](https://pic4.zhimg.com/80/v2-ef6e89e8b83012863bca4b4a200bc7fb_720w.jpg)


而**template函数** 与 **compile函数** 类似，只不过是增加了我们之前说的**re.TEMPLATE** 模式，我们可以看看源码。

![img](https://pic4.zhimg.com/80/v2-b05daad758fc2a39ab3b87004d022887_720w.jpg)



### 3.6 其他

**re.escape(pattern)** 可以转义正则表达式中具有特殊含义的字符，比如：`.` 或者 `*` ，举个实际的案例：

![img](https://pic1.zhimg.com/80/v2-903a19ca6560faa5ff97e09b16e59390_720w.jpg)


**re.escape(pattern)** 看似非常好用省去了我们自己加转义，但是使用它很容易出现转义错误的问题，所以并不建议使用它转义，**而建议大家自己手动转义！**

**re.purge()** 函数作用就是清除 **正则表达式缓存**，具体有什么缓存呢？我们来看看源码就知道它背地里干了 什么：

![img](https://pic1.zhimg.com/80/v2-17371b6785ee8112fdfdfab08826c694_720w.jpg)


看方法大概是清除缓存吧，我们再来看看具体的案例：

![img](https://pic1.zhimg.com/80/v2-faa3ba56f8137d908f269b044248cf00_720w.jpg)


猪哥在两个案例之间使用了**re.purge()** 函数清除缓存，然后分别比较前后案例源码里面的缓存，看看是否有变化！

![img](https://pic3.zhimg.com/80/v2-161cac2bd5c2f5f2e42f7759b2cae632_720w.jpg)



### 3.7 总结

同样最后来一张思维导图总结一下re模块中的函数吧，**需要高清图或者xmind文件的同学可在微信公众号「裸睡的猪」后台回复：re** 获取。

![img](https://pic3.zhimg.com/80/v2-c022af882c868bde000896bc3271f6e6_720w.jpg)



## 四、re模块异常

re模块还包含了一个正则表达式的编译错误，当我们给出的**正则表达式是一个无效的表达式**（就是表达式本身有问题）时，就会raise一个异常！

我们来看看具体的案例吧：

![img](https://pic2.zhimg.com/80/v2-857ad9feebc68f088f573af472569155_720w.jpg)


上图案例中我们可以看到，在编写正则表达式中我们多写了一个括号，这导致执行结果报错；而且是在其他所有案例执行之前，所以说明是在正则表达式编译时期就报错了。

注意：这个异常一定是 正则表达式 本身是无效的，与要匹配的字符串无关！

## 五、正则对象Pattern

关于`re`模块的常量、函数、异常我们都讲解完毕，但是完全有必要再讲讲**正则对象Pattern**。

### 5.1 与re模块 函数一致

在`re`模块的函数中有一个重要的函数 **compile函数** ，这个函数可以预编译返回一个正则对象，此正则对象拥有与`re`模块相同的函数，我们来看看**Pattern类**的源码。

![img](https://pic1.zhimg.com/80/v2-6ae609e027806c511f578a24d623412c_720w.jpg)


既然是一致的，那到底该用**re模块** 还是 **正则对象Pattern** ？

而且，有些同学可能看过`re`模块的源码，你会发现其实**compile函数** 与 其他 **re函数**（search、split、sub等等） 内部调用的是同一个函数，最终还是调用正则对象的函数！

![img](https://pic1.zhimg.com/80/v2-793ac58ceff4191cb328869cb2a69ca4_720w.jpg)


也就是说下面 两种代码写法 底层实现 其实是一致的：

\# re函数 re.search(pattern, text) # 正则对象函数 compile = re.compile(pattern) compile.search(text)

那还有必要使用**compile函数** 得到正则对象再去调用 **search函数** 吗？直接调用re.search 是不是就可以？

### 5.2 官方文档怎么说

关于到底该用**re模块** 还是 **正则对象Pattern** ，官方文档是否有说明呢？



![img](https://pic1.zhimg.com/80/v2-5ba334ea73634a5d608c36ce752a5cf4_720w.jpg)


官方文档推荐：**在多次使用某个正则表达式时推荐使用正则对象Pattern** 以增加复用性，因为通过 **re.compile(pattern)** 编译后的模块级函数会被缓存！

### 5.3 实际测试又如何？

上面官方文档推荐我们在 **多次使用某个正则表达式时使用正则对象**，那实际情况真的是这样的吗？

我们在实测一下吧



![img](https://pic2.zhimg.com/80/v2-ed971d8cf18adf225f558bf45371ff45_720w.jpg)


猪哥编写了两个函数，一个使用**re.search函数** 另一个使用 **compile.search函数** ，分别(不同时)循环执行**count次**(count从1-1万)，比较两者的耗时！

得出的结果猪哥绘制成折线图：

![img](https://pic1.zhimg.com/80/v2-53f9618228c6509bd5235e939c2c523c_720w.jpg)


得出的结论是：100次循环以内两者的速度基本一致，当超出100次后，使用 **正则对象Pattern** 的函数 耗时明显更短，所以比**re模块** 要快！

通过实际测试得知：Python 官方文档推荐 **多次使用某个正则表达式时使用正则对象函数** 基本属实！

## 六、注意事项

Python 正则表达式知识基本讲解完毕，最后稍微给大家提一提需要注意的点。

### 6.1 字节串 与 字符串

模式和被搜索的字符串既可以是 Unicode 字符串 (str) ，也可以是8位字节串 (bytes)。 但是，Unicode 字符串与8位字节串不能混用！

### 6.2 .r 的作用

正则表达式使用反斜杠（’’）来表示特殊形式，或者把特殊字符转义成普通字符。

而反斜杠在普通的 Python 字符串里也有相同的作用，所以就产生了冲突。

解决办法是对于正则表达式样式使用 Python 的原始字符串表示法；在带有 ‘r’ 前缀的字符串字面值中，反斜杠不必做任何特殊处理。

### 6.3 正则查找函数 返回匹配对象

查找一个匹配项（search、match、fullmatch）的函数返回值都是一个 **匹配对象Match** ，需要通过**match.group()** 获取匹配值，这个很容易忘记。

![img](https://pic4.zhimg.com/80/v2-5c3e53c27fcd5f643edd80d29410519f_720w.jpg)


另外还需要注意：**match.group()** 与**match.groups()** 函数的差别！

### 6.4 重复使用某个正则

如果要重复使用某个正则表达式，推荐先使用 **re.compile(pattern)函数** 返回一个正则对象，然后复用这个正则对象，这样会更快！

### 6.5 Python 正则面试

笔试可能会遇到需要使用Python正则表达式，不过不会太难的，大家只要记住那几个方法的区别，会正确使用，基本问题不大。