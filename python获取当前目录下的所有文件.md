os模块下有两个函数：

## os.walk()

`os.walk()` 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。

`os.walk()` 方法是一个简单易用的文件、目录遍历器，可以帮助我们高效的处理文件、目录方面的事情。

在Unix，Windows中有效。

```python
os.walk(top[,topdown=True[,onerror=None[, followlinks=False]]])
```

- top

  要遍历的目录地址，返回的是一个三元组：root,dirs,files

  - root 指的是当前遍历的这个文件夹本身的地址。
  - dirs是一个list，内容是该文件夹中所有的目录的名字（不包括子目录）
  - files同样是list，内容是该文件夹中所有的文件（不包括子目录）

- **topdown**，可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。如果 topdown 参数为 True，walk 会遍历top文件夹，与top 文件夹中每一个子目录。

- **onerror** -- 可选，需要一个 callable 对象，当 walk 需要异常时，会调用。

- **followlinks** -- 可选，如果为 True，则会遍历目录下的快捷方式(linux 下是软连接 symbolic link )实际所指的目录(默认关闭)，如果为 False，则优先遍历 top 的子目录。

```python
import os
def file_name(file_dir):
    for root,dirs,files in os.walk(file_dir):
        print(root) # 当前目录路径
        print(dirs) # 当前路径下所有子目录
        print(files) # 当前路径下所有非目录子文件
```

注意：以上代码会遍历所有子文件夹，如果只是想看当前目录一层则需要加上break

```python
a = os.walk(top="/home/zhaosheng",topdown=True,followlinks=False)
for i in a:
    print(i)
    break
```



```python
import os
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpeg':
                L.append(os.path.join(root,file))
    return L
```



递归输出当前路径下所有非目录子文件

```python
import os
def listdir(path,list_name): #传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
```



## os.listdir()

`os.listdir()` 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。

它不包括 **.** 和 **..** 即使它在文件夹中。

只支持在 Unix, Windows 下使用。

```python
os.listdir(path)
```

