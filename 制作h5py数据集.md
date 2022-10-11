python的h5py包处理数据集非常方便，导入数据时，并不会占据内存空间。

## 安装

```python
pip install h5py
conda install h5py
```



## 创建h5py文件

```python
import h5py
import numpy as np
f = h5py.File("test.hdf5","w")
```

`create_dataset`创建一个给定形状和dtype的数据集

```python
dset = f.create_dataset("mydataset",(100,),dtype='i')
```

也可以

```python
import h5py
import numpy as np
with h5py.File("test.hdf5","w") as f:
    dset = f.create_dataset("mydataset",(100,),dtype='i')
```

## Groups和分层组织

"HDF"代表"分层数据格式"。HDF5文件中的每个对象都有一个名称，他们使用`/`separatpr排列在POSIX样式的层次结构中：

```python
dset.name
# u'/mydataset'
```

此系统中的"folders"称为Groups。我们创建的File对象本身就是一个Groups，在本例中是根Groups，名为`/`:

```python
f.name
# u'/'
```

创建子组是通过适当命名的`create_group`完成的。但是我们需要先在"append"模式下打开文件（如果存在则读/写，否则创建）

```python

```

