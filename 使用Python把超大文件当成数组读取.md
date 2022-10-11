# 使用Python把超大文件当成数组来读取

## 1. 前言

在进行深度学习训练的时候，需要将样本文件加载到模型中进行训练，训练的样本文件很大，动辄几个G、十几个G的文件一次加载到内存，再加上深度学习模型训练的内存占用，内存很快就溢出了，那么就迫切需要用按需加载的方式来代替全部加载的方式。

## 2. 解决思路

训练样本文件基本是**按行进行存储的文本文件**，那么在第一次加载的时候，只需要存储每行的开始位置和每行的长度，在需要读取内容时，根据行开始位置和长度即可读取到内容。

实现路径是：自定义实现FileContentList类，实现`Iterator`和`Iterable`接口，同时支持切片操作。

## 3. 代码

### 3.1 声明类

```python
class FileContentList:
    """
    FileContentList allow you read file content as a list object.
    
    FileContentList('test.txt'):
            Create an object and load the contents of the file,
            it will not load everything, only the index of all lines.
    len(FileContentList('test.txt')):
            Return the numbers of lines in file.
    FileContentList('test.txt')[n]:
            Returns the content of the n'th line of the file.
    """
```

### 3.2 构造函数

*`self.filepath` 、`self.encoding`、`self.buffer_size` :*在构建函数中需要传入文件路径、文件编码格式和文件读取缓冲区大小（与加载速度有关）。

*`self.file`:* 同时在初始化时默认打开文件，避免文件频繁打开和关闭。

*`self.index`* : 记录每行文本在文件中的开始位置和每行长度。

*`self.iter_num`:* 记录当前迭代记录号。

*`self.loaded`:* 记录文件是否已初始化，即加载每行文本在文件中的开始位置和每行长度。

```python
def __init__(self, file_path, encoding='utf-8',buffer_size=1024*500):
    """
    Init FileContentList Object
    :param file_path:
    :param encoding:
    :param buffer_size:
    """
    
    self.file_path = file_path
    self.encoding = encoding
    self.buffer_size = buffer_size
    self.file = open(self.file_path,'rb')
    self.index = []
    self.iter_num = -1 # current line-number
    self.loaded = False
    pass
```



### 3.3 析构函数

当对象销毁时，自动关闭当前文件。

```python
def __del__(self):
    self.file.close()
```



### 3.4 加载每行文本在文件中的开始位置和每行长度

这里默认将每行的换行符为`\n`，没有只将`\r`作为换行符的情况，也不支持自定义换行符。如果有需要自定义行分隔符，修改`0x0A`的位置即可。

```python
def __load(self):
        """
        Load the binary offset and length(we call it the 'line index') of all lines of the file into list object.
        :return: None
        """
        # if the line index has been loaded, return immediately
        if self.loaded:
            return
        # read the line index
        buff_size = self.buffer_size
        offset, length = 0, 0
        f = self.file
        buff = f.read(buff_size)
        while len(buff) > 0:
            # find newline character '\n'
            start = 0
            pos = buff.find(0x0A, start)
            while pos >= 0:
                # founded! push the line start-position and line length to the list object
                length += (pos - start + 1)
                self.index.append([offset, length])
                offset += length
                length = 0
                start = pos + 1
                # find again
                pos = buff.find(0x0A, start)
            else:
                # there has no more newline charachter, the length must be updated
                length += (len(buff) - start)
            buff = f.read(buff_size)
        # the last line of file maybe have not newline character '\n', you need to remember them.
        if length > 0:
            self.index.append([offset, length])
        # Update flag, indicating that the row index has been loaded
        self.loaded = True
        pass
```



### 3.5 支持len操作，返回文件总行数

通过`__len__` 方法，支持`len()`函数操作，返回文件总行数。

```python
def __len__(self):
    """
    Return the total number of lines in the file
    :return:
    """
    self.__load()
    return len(self.index)
```



### 3.6 支持下标取值操作、支持切片操作

通过`__getitem__`方法支持下标取值操作，还考虑到切片取值情况。

```python
def __getitem__(self,item):
    self.__load()
    if isinstance(item,int):
        return self.__getitem(item)
    elif isinstance(item, slice):
        text_list = []
        start,stop,step = item.start, item.stop, item.step
        start = 0 if start is None else start
        step = 1 if step is None else step
        stop = len(self.index) if stop is None else stop
        for n in range(start, stop, step):
            text = self.__getitem(n)
            text_list.append(text)
        return text_list
    
def __getitem(self,n):
    if n>= len(self.index) or n < -len(self.index):
        raise IndexError('index out of range')
    [offset,length] = self.index[n]
    self.file.seek(offset)
    b = self.file.read(length)
    text = b.decode(encoding=self.encoding, errors="strict")
    return text
```



### 3.7 支持迭代操作，支持遍历文本文件

通过`__iter__` 方法返回迭代器，通过`__next__` 方法支持迭代获取值

```python
def __iter__(self):
    self.iter_num = -1
    return self

def __next__(self):
    self.__load()
    pass
    self.iter_num += 1
    if self.iter_num >= len(self.index)
        raise StopIteration
    return self.__getitem__(self.iter_num)
```



## 4.使用

### 4.1 创建对象

```python
# create new object
fl = FileContentList('data.txt')
```

### 4.2 获取文件总行数

```python
# print line count
count = len(fl)
print('line count:%d' % count)
```

### 4.3 获取指定行的文本，通过切片获取指定行的文本

```python
# slice
print(fl[:5])
print(fl[-1])

# print top 10 lines
for i in range(10):
    print(fl[i])
```

### 4.4 迭代返回文本

```python
# enumerate top 10 lines
for line,n in enumerate(fl):
    if n >= 10:
        break
    print(line)
```



## 5.性能

使用文件大小分别为1.91G和6.53G的两个文件进行测试，测试内容主要是：进行初始化操作和随机返回1000行文本，buffer_size分别取10K、100K、1000K进行测试。

电脑配置：

CPU：Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz 1.80 GHz

内存：16G

硬盘：SSD 512G

| 缓冲区大小 | 初始化(ms) 文件 1.91G | 初始化(ms) 文件 6.53G |
| ---------- | --------------------- | --------------------- |
| 10K        | 3081                  | 37446                 |
| 100K       | 2154                  | 8727                  |
| 1M         | 2079                  | 8266                  |
| 2M         | 2541                  | 9813                  |

| 次数  | 随机返回1000行文本(ms) 文件 1.91G | 随机返回1000行文本(ms) 文件 6.53G |
| ----- | --------------------------------- | --------------------------------- |
| 第1次 | 15.79                             | 15.99                             |
| 第2次 | 15.04                             | 15.62                             |
| 第3次 | 15.60                             | 15.99                             |

可以看到，初始化的性能与文件大小、buffer_size 有关，buffe_size=1M 时速度最快；随机返回文本与文件大小并无关系。