# Python-import导入上级目录文件

假设有如下目录结构：

```text
-- dir0
　　| file1.py
　　| file2.py
　　| dir3
　　　| file3.py
　　| dir4
　　　| file4.py
```

dir0文件夹下有file1.py、file2.py两个文件和dir3、dir4两个子文件夹，dir3中有file3.py文件，dir4中有file4.py文件。

## 1.导入同级模块

python导入同级模块（在同一个文件夹中的py文件）直接导入即可。

```python
import xxx
```

如在file1.py中想导入file2.py，注意无需加后缀".py"：

```python
import file2
# 使用file2中函数时需加上前缀"file2."，即：
# file2.fuction_name()
```

## 2.导入下级模块

导入下级目录模块也很容易，需在下级目录中新建一个空白的__init__.py文件再导入：

```python
from dirname import xxx
```

如在file1.py中想导入dir3下的file3.py，首先要在dir3中新建一个空白的__init*__*.py文件。

```text
-- dir0
　　| file1.py
　　| file2.py
　　| dir3
　　　| __init__.py
　　　| file3.py
　　| dir4
　　　| file4.py
```

再使用如下语句：

```python
# plan A
from dir3 import file3
```

或是

```python
# plan B
import dir3.file3
# import dir3.file3 as df3
```

但使用第二种方式则下文需要一直带着路径dir3书写，较为累赘，建议可以另起一个别名。

## 3.导入上级模块

要导入上级目录下模块，可以使用`sys.path`： 　

```python
import sys 
sys.path.append("..") 
import xxx　
```

如在file4.py中想引入import上级目录下的file1.py：

```python
import sys 
sys.path.append("..") 
import file1
```

**sys.path的作用：**当使用import语句导入模块时，解释器会搜索当前模块所在目录以及sys.path指定的路径去找需要import的模块，所以这里是直接把上级目录加到了sys.path里。

**“..”的含义：**等同于linux里的‘..’，表示当前工作目录的上级目录。实际上python中的‘.’也和linux中一致，表示当前目录。

## 4.导入隔壁文件夹下的模块

如在file4.py中想引入import在dir3目录下的file3.py。

这其实是前面两个操作的组合，其思路本质上是将上级目录加到sys.path里，再按照对下级目录模块的方式导入。

同样需要被引文件夹也就是dir3下有空的__init__.py文件。

```text
-- dir
　　| file1.py
　　| file2.py
　　| dir3
　　　| __init__.py
　　　| file3.py
　　| dir4
　　　| file4.py
```

同时也要将上级目录加到sys.path里：

```python
import sys
sys.path.append("..")
from dir3 import file3
```

## 5.常见错误及import原理：

在使用直接从上级目录引入模块的操作时：

```python
from .. import xxx
```

经常会报错:

```python3
ValueError: attempted relative import beyond top-level package
```

这是由于相对导入时，文件夹实质上充当的是package，也就是包的角色（比如我们常用的numpy、pandas都是包）。如果python解释器没有认同该文件夹是package，那么这就是一个普通的文件夹，无法实现相对导入。

文件夹作为package需要满足如下两个条件：

1. 文件夹中必须存在有__init__.py文件，可以为空。
2. 不能作为顶层模块来执行该文件夹中的py文件。