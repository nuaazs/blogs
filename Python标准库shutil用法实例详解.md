## 1. 文件夹与文件操作

### 1.1 `copyfileobj(fsrc,fdst,length=16*1024)`

将`fsrc`文件内容复制到`fdst`文件，`length`为`fsrc`每次读取的长度，用作缓冲区大小。

- `fsrc`： 源文件
- `fdst`： 复制至fdst文件
- `length`： 缓冲区大小，即fsrc每次读取的长度

```python
import shutil
f1 = open("file.txt","r")
f2 = open("file_copy.txt","a+")
shutil.copyfileobj(f1,f2,length=1024)
```



### 1.2 `copyfile(src,dst)`

将`src`文件内容复制至`dst`文件

- `src`： 源文件路径
- `dst`： 复制至dst文件，若dst文件不存在，将会生成一个dst文件；若存在将会被覆盖
- `follow_symlinks`：设置为True时，若src为软连接，则当成文件复制；如果设置为False，复制软连接。默认为True。

```python
import shutil
shutil.copyfile("file.txt","file_copy.txt")
```



### 1.3 `copymode(src,dst)`

将`src`文件权限复制至`dst`文件。文件内容，所有者和组不受影响。

- `src`： 源文件路径
- `dst`： 将权限复制至dst文件，dst路径必须是真实的路径，并且文件必须存在，否则将会报文件找不到错误
- `follow_symlinks`：设置为False时，src, dst皆为软连接，可以复制软连接权限，如果设置为True，则当成普通文件复制权限。默认为True。

```python
import shutil
shutil.copymode("file.txt","file_copy.txt")
```





### 1.4 `copystat(src,dst)`

将权限，上次访问时间，上次修改时间以及`src`的标志复制到`dst`。文件内容，所有者和组不受影响

- `src`： 源文件路径
- `dst`： 将权限复制至dst文件，dst路径必须是真实的路径，并且文件必须存在，否则将会报文件找不到错误
- `follow_symlinks`：设置为False时，src, dst皆为软连接，可以复制软连接权限、上次访问时间，上次修改时间以及src的标志，如果设置为True，则当成普通文件复制权限。默认为True。

```python
import shutil
shutil.copystat("file.txt","file_copy.txt")
```



### 1.5 `copy(src, dst)`

 将文件`src`复制至`dst`。

`dst`可以是个目录，会在该目录下创建与`src`同名的文件，若该目录下存在同名文件，将会报错提示已经存在同名文件。权限会被一并复制。本质是先后调用了`copyfile`与`copymode`而已。

- `src`： 源文件路径
- `dst`： 将权限复制至dst文件，dst路径必须是真实的路径，并且文件必须存在，否则将会报文件找不到错误
- `follow_symlinks`：设置为False时，src, dst皆为软连接，可以复制软连接权限、上次访问时间，上次修改时间以及src的标志，如果设置为True，则当成普通文件复制权限。默认为True。

```python
improt shutil,os
shutil.copy2("file.txt","file_copy.txt")
# 或者
shutil.copy2("file.txt",os.path.join(os.getcwd(),"copy"))
##  os.getcwd() 方法用于返回当前工作目录
```



### 1.6 `copy2(src, dst)`

 将文件`src`复制至`dst`。

`dst`可以是个目录，会在该目录下创建与`src`同名的文件，若该目录下存在同名文件，将会报错提示已经存在同名文件。权限、上次访问时间、上次修改时间和`src`的标志会一并复制至`dst`。本质是先后调用了`copyfile`与`copymode`而已。

- `src`： 源文件路径
- `dst`： 将权限复制至dst文件，dst路径必须是真实的路径，并且文件必须存在，否则将会报文件找不到错误
- `follow_symlinks`：设置为False时，src, dst皆为软连接，可以复制软连接权限、上次访问时间，上次修改时间以及src的标志，如果设置为True，则当成普通文件复制权限。默认为True。

```python
improt shutil,os
shutil.copy2("file.txt","file_copy.txt")
# 或者
shutil.copy2("file.txt",os.path.join(os.getcwd(),"copy"))
```



### 1.7 `ignore_patterns(*patterns)`

忽略模式，用于配合copytree()方法，传递文件将会被忽略，不会被拷贝。

- `patterns`:文件名称，元组



### 1.8 `copytree(src,dst,symlinks=False,ignore=None)`

- `src`：源文件夹
- `dst`：复制至`dst`文件夹，该文件夹会自动创建，需保证此文件夹不存在，否则将报错
- `symlinks`：是否复制软连接，True复制软连接，False不复制，软连接会被当成文件复制过来，默认False
- `ignore`：忽略模式，可传入`ignore_patterns()`
- `copy_function`：拷贝文件的方式，可以传入一个可执行的处理函数，默认为`copy2`
- `ignore_dangling_symlinks`：`sysmlinks`设置为False时，拷贝指向文件已删除的软连接时，将会报错，如果想消除这个异常，可以设置此值为True。默认为False

```python
import shutil,os
folder1 = os.path.join(os.getcwd(),"aaa")

# bbb和ccc文件夹都可以不存在，会自动创建
folder2 = os.path.join(os.getcwd(),"bbb","ccc")

# 将"abc.txt","bcd.txt"忽略不复制
shutil.copytree(folder1,folder2,ignore=shutil.jgnore_patterns("abc.txt","bcd.txt"))
```



### 1.9 `retree(path,ignore_errors=False,onerror=None)`

移除文件树，将文件夹目录删除。

- `ignore_errors`：是否忽略错误，默认False
- `onerror`：定义错误处理函数，需传递一个可执行的处理函数，该处理函数接收三个参数：函数、路径和`excinfo`

```python
import shutil,os
folder1 = os.path.join(os.getcwd(),"aaa")
shutil.rmtree(folder1)
```



### 1.10 `move(src,dst)`

将src移动至dst目录下，若dst目录不存在，则效果等同于src改名为dst。若dst目录存在，将会把src文件夹的所有内容移动至该目录下面。

- `src`：源文件夹或文件
- `dst`：移动至`dst`文件夹，或将文件改名为`dst`文件。如果`src`为文件夹，而`dst`为文件将会报错
- `copy_function`：拷贝文件的方式，可以传入一个可执行的处理函数。默认为copy2。

```python
import shutil,os
# 示例一，将src文件夹移动至dst文件夹下面，如果bbb文件夹不存在，则变成了重命名操作
folder1 = os.path.join(os.getcwd(),"aaa")
folder2 = os.path.join(os.getcwd(),"bbb")
shutil.move(folder1, folder2)
# 示例二，将src文件移动至dst文件夹下面，如果bbb文件夹不存在，则变成了重命名操作
file1 = os.path.join(os.getcwd(),"aaa.txt")
folder2 = os.path.join(os.getcwd(),"bbb")
shutil.move(file1, folder2)
# 示例三，将src文件重命名为dst文件(dst文件存在，将会覆盖)
file1 = os.path.join(os.getcwd(),"aaa.txt")
file2 = os.path.join(os.getcwd(),"bbb.txt")
shutil.move(file1, file2)
```



### 1.11 `disk_usage(path)`

获取当前目录所在硬盘的使用情况。

- `path`：文件夹或文件路径。Windows中必须是文件夹路径，在linux中可以是文件路径和文件夹路径

```python
import shutil,os
path = os.path.join(os.getcwd(),"aaa")
info = shutil.disk_usage(path)
print(info)
# usage(total=95089164288, used=7953104896, free=87136059392)
```



### 1.12 `chown(path,user=None,group=None)`

修改路径指向的文件或文件夹的所有者或分组。

- `path`：路径
- `user`：所有者，传递user的值必须是真实的，否则将报错no such user
- `group`：分组，传递group的值必须是真实的，否则将报错no such group

```python
import shutil,os
path = os.path.join(os.getcwd(),"file.txt")
shutil.chown(path,user="root",group="root")
```



### 1.13 `which(cmd,mode=os.F_OK|os.X_OK,path=None)`

获取给定的cmd命令的可执行文件路径。

```python
import shutil
info = shutil.which("python3")
print(info)
# /usr/bin/python3
```



## 2. 归档操作

shutil还提供了创建和读取压缩和存档文件的高级使用程序。内部实现主要依靠的是`zipfile`和`tarfile`模块

### 2.1 `make_archive(base_name,format,root_dir, ...)`

生成压缩文件。

- `base_name`：压缩文件的文件名，不允许有扩展名，因为会根据压缩格式生成相应的扩展名
- `format`：压缩格式
- `root_dir`：将指定文件夹进行压缩

```python
import shutil,os
base_name = os.path.join(os.getcwd(),"aaa")
format = "zip"
root_dir = os.path.join(os.getcwd(),"aaa")

# 将会root_dir文件夹下的内容进行压缩，生成一个aaa.zip文件。
shutil.make_archive(base_name,format,root_dir)
```



### 2.2 `get_archive_formats()`

获取支持的压缩文件格式。目前支持的有：`tar`、`zip`、`gztar`、`bztar`。在Python3还多支持一种格式`xztar`



### 2.3 `unpack_archive(filename,extract_dir=None,format=None)`

解压操作。

- `filename`：文件路径。
- `extract_dir`：解压至的文件夹路径。文件夹可以不存在，会自动创建。
- `format`：解压格式，默认为None，会根据扩展名自动选择解压格式。

```python
import shutil,os
zip_path = os.path.join(os.getcwd(),"aaa.zip")
extract_dir = os.path.join(os.getcwd(),"aaa")
shutil.unpack_archive(zip_path,extract_dir)
```



### 2.4 `get_unpack_formats()`

 获取支持的解压文件格式。目前支持的有：`tar`、`zip`、`gztar`、`bztar`和`xztar`。
