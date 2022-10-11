本文代码基于PyTorch 1.0版本，需要用到以下包：

```python
import collections
import os
import shutil
import tqdm
import numpy as np
import PIL.Image
import torch
import torchvision
```



## 〇、基本概念

### 0. 数据转换

![Snipaste_2020-09-22_16-04-30](C:\Users\nuaazs\Desktop\Snipaste_2020-09-22_16-04-30.png)

### 1.numpy array 和 Tensor(CPU & GPU)

```python
a = np.ones(5)  # array([1., 1., 1., 1., 1.])
b = torch.from_numpy(a)     # numpy array-> CPU Tensor
# tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
y = b.cuda()     # CPU Tensor -> GPU Tensor
# tensor([1., 1., 1., 1., 1.], device='cuda:0', dtype=torch.float64)
y = y.cpu()  # GPU Tensor-> CPU Tensor
# tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
y = y.numpy()  # CPU Tensor -> numpy array
# array([1., 1., 1., 1., 1.])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y = torch.from_numpy(y)
y.to(device) # 这里 x.to(device) 等价于 x.cuda()
tensor([1., 1., 1., 1., 1.], device='cuda:0', dtype=torch.float64)
```

索引、view是不会开辟新内存的，而像y=x+y这样的运算会新开内存，然后将y指向新内存。



### 2.Variable 和 Tensor（require_grad = True)

PyTorch 0.4 之前的模式为：Tensor没有梯度计算，加上梯度更新等操作后可以变为`Variable` 。PyTorch 0.4 将Tensor和Variable合并。默认Tensor的require_grad为false，可以通过修改requires_grad来为其添加梯度更新操作。

```python
>>> y
tensor([1., 1., 1., 1., 1.], dtype=torch.float64)  
>>> y.requires_grad
False
>>> y.requires_grad = True
>>> y
tensor([1., 1., 1., 1., 1.], dtype=torch.float64, requires_grad=True)
```



### 3.detch和with torch.no_grad()

一个比较好的`detach` 和`torch.no_grad` 区别的解释：

> `detach()` :detaches the output from the computational graph. So no gradient will be backproped along this variable.
>
> `torch.no_grad` :says that no operation should build the graph.
>
> The difference is that one refers to only a given variable on which it's called. The other affects all operations taking place within the with statement.

`detach()` 将一个变量从计算图中分离出来，也没有了相关的梯度更新。`torch.no_grad()` 只是说明该操作没有必要建图。不同之处在于，前者只能指定一个给定的变量，后者则会影响在with语句中发生的所有操作。



### 4. model.eval() 和 torch.no_grad()

> These two have different goals:
>
> - `model.eval()` will notify all your layers that you are in eval mode,that way, batchnorm or dropout layers will work in eval mode instead of training mode.
> - `torch.no_grad()` impacts the autograd engine and deactive it. It will reduce memory usage and speed up computations but you won't be able to backprop(which you don't want in an eval script).

model.eval() 和 torch.no_grad() 的区别在于，`model.eval()` 是将网络切换为测试状态，例如BN和随机失活(dropout)在训练和测试阶段使用不同的计算方法。`torch.no_grad()` 是关闭Pytorch张量的自动求导机制，以减少存储使用和加速计算，得到的结果无法进行`loss.backward()` 。



### 5. xx.data 和 xx.detach()

在 0.4.0 版本之前，`.data` 的语义是获取Variable的内部Tensor，在0.4.0 版本将`Variable` 和 `Tensor` merge之后，`.data` 和之前有类似的语义，也是内部的Tensor的概念。`x.data`与`x.detach()` 返回的tensor有相同的地方，也有不同的地方

**相同:**

- 都和 x 共享同一块数据
- 都和 x 的 计算历史无关
- requires_grad = False

**不同:**

- y= x.data 在某些情况下不安全, 某些情况, 指的就是上述 inplace operation 的第二种情况, 所以, release note 中指出, 如果想要 detach 的效果的话, 还是 detach() 安全一些.

```python
import torch
x = torch.FloatTensor([[1.,2.]])
w1 = torch.FloatTensor([[2.], [1.]])
w2 = torch.FloatTensor([3.])
w1.requires_grad = True
w2.requires_grad = True
d = torch.matul(x,w1)
d_ = d.data
f = torch.matul(d,w2)
d_[:] = 1
f.backward()
```

**如果需要获取其值，可以使用 xx.cpu().numpy() 或者 xx.cpu().detach().numpy() 然后进行操作，不建议再使用 volatile和 xx.data操作。**



### 6.ToTensor & ToPILImage 各自都做了什么？

ToTensor：

- 取值范围：[0,255] -> [0,1.0]
- NHWC -> NCHW
- PILImage -> FloatTensor

```python
# PIL.Image -> torch.Tensor.
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2, 0, 1).float() / 255
#　等价于
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path)) 
```



ToPILImage:

- 取值范围: [0, 1.0] --> [0, 255]
- NCHW --> NHWC
- 类型: FloatTensor -> numpy Uint8 -> PILImage

```python
# torch.Tensor -> PIL.Image.
image = PIL.Image.fromarray(torch.clamp(tensor * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
# 等价于
image = torchvision.transforms.functional.to_pil_image(tensor)
```



### 7.torch.nn.xxx 与torch.nn.function.xxx

建议统一`torch.nn.xxx`。`torch.functional.xxx` 可能会在下一个版本中去掉。

`torch.nn`　模块和　`torch.nn.functional`　的区别在于，`torch.nn`　模块在计算时底层调用了`torch.nn.functional`，但　`torch.nn`　模块包括该层参数，还可以应对**训练**和**测试**两种网络状态。使用　`torch.nn.functional`　时要注意网络状态，如:

```python
def forward(self, x):
    ...
    x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
```















## 一、基础配置

### 1. 检查Pytorch的版本

```python
torch.version
torch.version.cuda
torch.backends.cudnn.version()
torch.cuda.get_device_name(0)    # GPU type
```

### 2. 更新PyTorch

```python
conda update pytorch torchvision -c pytorch
```

### 3. 固定随机种子

```python
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```

### 4.指定程序运行在特定GPU卡上

在命令行指定环境变量

```python
CUDA_VISIBLE_DEVICES=0,1 python3 train.py
```

或在代码中指定

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```

### 5.判断是否有CUDA支持

```python
torch.cuda.is_available()
torch.set_default_tensor_type('torch.cuda.FloatTensor')
os.environ['CUDA_LAUNCH_BLOCKING']='1'
```

### 6.设置为cuDNN benchmark模式

Benchmark模式会提升计算速度，但是由于计算中的随机性，每次网络前馈结果略有差异。

```python
torch.backends.cudnn.benchmark = True
```

如果想要避免这种波动，设置

```python
torch.backends.cudnn.deterministic = True
```

### 7.清除GPU存储

有时Control-C终止运行后GPU存储没有及时释放，需要手动清空。在PyTorch内部可以

```python
torch.cuda.empty_cache()
```

或者在命令行可以先使用ps找到程序的PID，再使用kill结束该进程

```shell
ps aux | grep python
kill -9 [pid]
```

或者直接重置没有被清空的GPU

```shell
nvidia-smi --gpu-reset -i [gpu_id]
```

## 二、张量操作

### 1.张量基本信息 和 转换

Pytorch 给出了 9 种 CPU Tensor 类型和 9 种 GPU Tensor 类型。Pytorch 中默认的数据类型是 `torch.FloatTensor` , 即 `torch.Tensor`  等同于 `torch.FloatTensor` 。

| Data type                | dtype                         | CPU tensor         | GPU tensor              |
| ------------------------ | ----------------------------- | ------------------ | ----------------------- |
| 32-bit floating point    | torch.float32 or torch.float  | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64-bit floating point    | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16-bit floating point    | torch.float16 or torch.half   | torch.HalfTensor   | torch.cuda.HalfTensor   |
| 8-bit integer (unsigned) | torch.uint8                   | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8-bit integer (signed)   | torch.int8                    | torch.CharTensor   | torch.cuda.CharTensor   |
| 16-bit integer (signed)  | torch.int16 or torch.short    | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32-bit integer (signed)  | torch.int32 or torch.int      | torch.IntTensor    | torch.cuda.IntTensor    |
| 64-bit integer (signed)  | torch.int64 or torch.long     | torch.LongTensor   | torch.cuda.LongTensor   |
| Boolean                  | torch.bool                    | torch.BoolTensor   | torch.cuda.BoolTensor   |

```python
tensor.type() #Data type
tensor.size() #Shape of the tensor. It is a subclass of Python tuple
tensor.dim()  #Number of dimensions
t.numel() / t.nelement()  # 两者等价, 返回 tensor 中元素总个数
```

##### 最基本的Tensor创建方式

```python
troch.Tensor(2, 2) # 会使用默认的类型创建 Tensor, 
                   # 可以通过 torch.set_default_tensor_type('torch.DoubleTensor') 进行修改
torch.DoubleTensor(2, 2) # 指定类型创建 Tensor

torch.Tensor([[1, 2], [3, 4]])  # 通过 list 创建 Tensor
                                # 将 Tensor转换为list可以使用: t.tolist()
torch.from_numpy(np.array([2, 3.3]) ) # 通过 numpy array 创建 tensor
```

##### 确定初始值的方式创建

```python
torch.ones(sizes)  # 全 1 Tensor     
torch.zeros(sizes)  # 全 0 Tensor
torch.eye(sizes)  # 对角线为1，不要求行列一致
torch.full(sizes, value) # 指定 value
```

##### 分布

```python
torch.rand(sizes)  # 均匀分布   
torch.randn(sizes)   # 标准分布
# 正态分布: 返回一个张量，包含从给定参数 means, std 的离散正态分布中抽取随机数。 
# 均值 means 是一个张量，包含每个输出元素相关的正态分布的均值 -> 以此张量的均值作为均值
# 标准差 std 是一个张量，包含每个输出元素相关的正态分布的标准差 -> 以此张量的标准差作为标准差。 
# 均值和标准差的形状不须匹配，但每个张量的元素个数须相同
torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
tensor([-0.1987,  3.1957,  3.5459,  2.8150,  5.5398,  5.6116,  7.5512,  7.8650,
         9.3151, 10.1827])
torch.uniform(from,to) # 均匀分布 

torch.arange(s, e, steps)  # 从 s 到 e，步长为 step
torch.linspace(s, e, num)   # 从 s 到 e, 均匀切分为 num 份
# ! 注意linespace和arange的区别，前者的最后一个参数是生成的Tensor中元素的数量，而后者的最后一个参数是步长。
torch.randperm(m) # 0 到 m-1 的随机序列
# ! shuffle 操作
tensor[torch.randperm(tensor.size(0))] 
```

##### 复制

Pytorch 有几种不同的复制方式，注意区分

| Operation             | New/Shared memory | Still in computation graph |
| --------------------- | ----------------- | -------------------------- |
| tensor.clone()        | New               | Yes                        |
| tensor.detach()       | Shared            | No                         |
| tensor.detach.clone() | New               | No                         |



### 2.Tensor数据类型

```python
# Set default tensor type. Float in Pytorch is much faskter than double.
torch.set_default_tensor_type(torch.FloatTensor)

# Type convertions.
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()

# CPU/GPU 互转
# CPU Tensor 和 GPU Tensor 的区别在于， 前者存储在内存中，而后者存储在显存中。两者之间的转换可以通过 .cpu()、.cuda()和 .to(device) 来完成

```

判断Tensor类型几种方式：

```python
>>> a
tensor([[0.6065, 0.0122, 0.4473],
        [0.5937, 0.5530, 0.4663]], device='cuda:0')
>>> a.is_cuda  # 可以显示是否在显存中
True
>>> a.dtype   # Tensor 内部data的类型
torch.float32
>>> a.type()
'torch.cuda.FloatTensor'  # 可以直接显示 Tensor 类型 = is_cuda + dtype
```

##### 类型转换

```python
>>> a
tensor([[0.6065, 0.0122, 0.4473],
        [0.5937, 0.5530, 0.4663]], device='cuda:0')
>>> a.type(torch.DoubleTensor)   # 使用 type() 函数进行转换
tensor([[0.6065, 0.0122, 0.4473],
        [0.5937, 0.5530, 0.4663]], dtype=torch.float64)
>>> a = a.double()  # 直接使用 int()、long() 、float() 、和 double() 等直接进行数据类型转换进行
tensor([[0.6065, 0.0122, 0.4473],
        [0.5937, 0.5530, 0.4663]], device='cuda:0', dtype=torch.float64)
>>> b = torch.randn(4,5)
>>> b.type_as(a)  # 使用 type_as 函数, 并不需要明确具体是哪种类型
tensor([[ 0.2129,  0.1877, -0.0626,  0.4607, -1.0375],
        [ 0.7222, -0.3502,  0.1288,  0.6786,  0.5062],
        [-0.4956, -0.0793,  0.7590, -1.0932, -0.1084],
        [-2.2198,  0.3827,  0.2735,  0.5642,  0.6771]], device='cuda:0',
       dtype=torch.float64)
```







### 3.torch.Tensor与np.ndarray转换

```python
# torch.Tensor -> np.ndarray
ndarray = tensor.cpu().numpy()

# np.ndarray -> torch.Tensor
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float() # If ndarray has negative stride
```

### 4.torch.Tensor与PIL.Image转换

Pytorch中的张量默认采用`N`,`D` ,`H`,`W` 的顺序，并且数据范围在[0,1]，需要进行转置和规范化。

```python
# torch.Tensor -> PIL.Image.
image = PIL.Image.fromarray(torch.clamp(tensor*255,min=0,max=255).byte().permute(1,2,0).cup().numpy())
# Equivalently way
image = torchvision.transforms.functional.to_pil_image(tensor)

# PIL.Image -> torch.Tensor
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float()/255
# Equivalently way
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path))

```

### 5.np.ndarray与PIL.Image转换

```python
# np.ndarray -> PIL.Image
image = PIL.Image.fromarray(ndarray.astype(np.uint8))
# PIL.Image -> np.ndarray
nparray = np.asarray(PIL.Image.open(path))
```

### 6.从只包含一个元素的张量中提取值

这在训练时统计loss的变化过程中特别有用。否则浙江累计计算图，使GPU存储占用量越来越大。

```python
value = tensor.item()
```

### 7.张量形变

张量形变常常需要将卷积层特征输入全连接层的情形。相比`torch.view` ,`torch.reshape` 可以自动处理**张量不连续**的情况。

```python
tensor = torch.reshape(tensor,shape)
```

### 8.打乱顺序

```python
# Shuffle the first dimension
tensor = tensor[torch.randperm(tensor.size(0))]
```

### 9.水平翻转

PyTorch不支持`tensor[::-1]` 这样的负步长操作，水平翻转可以用张量索引实现。

```python
# Assume tensor has shape N*D*H*W.
tensor = tensor[:,:,:,torch.arange(tensor.size(3)-1,-1,-1).long()]
```

### 10.复制张量

三种复制方式，对应不同的需求。

```python
# Operation                 |  New/Shared memory | Still in computation graph |
tensor.clone()            # |        New         |          Yes               |
tensor.detach()           # |      Shared        |          No                |
tensor.detach.clone()()   # |        New         |          No                |
```

### 11.拼接张量

`torch.cat` 和`torch.stack` 的区别在于`torch.cat` 沿着给定的维度拼接，而`torch.stack`会新增

一维。例如当参数是3个10×5的张量，`torch.cat` 的结果是30×5的张量，而`torch.stack` 的结果是3×10×5的张量。

```python
tensor = torch.cat(list_of_tensors, dim=0)
tensor = torch.stack(list_of_tensors, dim=0)
```

### 12.将整数标记转换为独热(one hot)码

PyTorch中默认从0开始。

```
N = tensor.size(0)
one_hot = torch.zeros(N, num_classes).long()
one_hot.scatter_(dim=1,index = torch.unsqueeze(tensor, dim=1),src =torch.ones(N, num_classes).long())
```

### 13.得到非零/零元素

```python
torch.nonzero(tensor)		#Index of non-zero elements
torch.nonzero(tensor == 0)  #Index of zero elements
torch.nonzero(tensor).size(0) #Number of non-zero elements
torch.nonzero(tensor == 0).size(0) #Number of zero elements
```

### 14.判断两个张量相等

```python
torch.allclose(tensor1, tensor2)  # float tensor
torch.equal(tensor1, tensor2)  # int tensor
```

### 15.张量扩展

```python
# Expand tensor of shape 64*512 to shape 64*512*7*7
torch.reshape(tensor,(64,512,1,1)).expand(64,512,7,7)
```

### 16.矩阵乘法

```python
# Matrix multiplication:(m*n) * (n*p) -> (m * p)
result = torch.mm(tensor1,tensor2)

# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p).
result = torch.bmm(tensor1,tensor2)

# element-wise multiplication
result = tensor1 * tensor2
```

### 17.计算两组数据之间的两两欧氏距离

```python 
# X1 is of shape m*d ,X2 is of shape n*d
dist = torch.sqrt(torch.sum((X1[:,None,:]-X2)**2,dim=2))
```



### 18.索引、比较、排序

#### 索引操作

```python
a.item() #　从只包含一个元素的张量中提取值

a[row, column]   # row 行， cloumn 列
a[index]   # 第index 行
a[:,index]   # 第 index 列

a[0, -1]  # 第零行， 最后一个元素
a[:index]  # 前 index 行
a[:row, 0:1]  # 前 row 行， 0和1列

a[a>1]  # 选择 a > 1的元素， 等价于 a.masked_select(a>1)
torch.nonzero(a) # 选择非零元素的坐标，并返回
a.clamp(x, y)  # 对 Tensor 元素进行限制， 小于x用x代替， 大于y用y代替
torch.where(condition, x, y)  # 满足condition 的位置输出x， 否则输出y
>>> a
tensor([[ 6., -2.],
        [ 8.,  0.]])
>>> torch.where(a > 1, torch.full_like(a, 1), a)  # 大于1 的部分直接用1代替， 其他保留原值
tensor([[ 1., -2.],
        [ 1.,  0.]])

#　得到非零元素
torch.nonzero(tensor)               # 非零元素的索引
torch.nonzero(tensor == 0)          # 零元素的索引
torch.nonzero(tensor).size(0)       # 非零元素的个数
torch.nonzero(tensor == 0).size(0)  # 零元素的个数
```

#### 比较操作

```python
gt >    lt <     ge >=     le <=   eq ==    ne != 
topk(input, k) -> (Tensor, LongTensor)
sort(input) -> (Tensor, LongTensor)
max/min => max(tensor)      max(tensor, dim)    max(tensor1, tensor2)
```

sort 函数接受两个参数, 其中 参数 0 为按照行排序、1为按照列排序: True 为降序， False 为升序， 返回值有两个， 第一个是排序结果， 第二个是排序序号

```python
>>> import torch
>>> a = torch.randn(3, 3)
>>> a
tensor([[-1.8500, -0.2005,  1.4475],
        [-1.7795, -0.4968, -1.8965],
        [ 0.5798, -0.1554,  1.6395]])
>>> a.sort(0, True)[0] 
tensor([[ 0.5798, -0.1554,  1.6395],
        [-1.7795, -0.2005,  1.4475],
        [-1.8500, -0.4968, -1.8965]])
>>> a.sort(0, True)[1]
tensor([[2, 2, 2],
        [1, 0, 0],
        [0, 1, 1]])
>>> a.sort(1, True)[1]
tensor([[2, 1, 0],
        [1, 0, 2],
        [2, 0, 1]])
>>> a.sort(1, True)[0]
tensor([[ 1.4475, -0.2005, -1.8500],
        [-0.4968, -1.7795, -1.8965],
        [ 1.6395,  0.5798, -0.1554]])
```



### 19.Element-wise 和 归并操作

Element-wise:输出的Tensor 形状与原始的形状一致

```python
abs / sqrt / div / exp / fmod / log / pow...
cos / sin / asin / atan2 / cosh...
ceil / round / floor / trunc
clamp(input, min, max)
sigmoid / tanh...
```

归并操作:输出的Tensor形状小于原始的Tensor形状

```python
mean/sum/median/mode   # 均值/和/ 中位数/众数
norm/dist  # 范数/距离
std/var  # 标准差/方差
cumsum/cumprd # 累加/累乘
```



### 20.变形操作

#### view/resize/reshape调整Tensor的形状

- 元素总数必须相同
- view 和 reshape 可以使用 -1 自动计算维度
- 共享内存

`view()` 操作是需要 Tensor 在内存中连续的， 这种情况下需要使用 `contiguous()` 操作先将内存变为连续。 对于reshape 操作， 可以看做是 `Tensor.contiguous().view()`.

```python
a = torch.Tensor(2,2)
# tensor([[6.0000e+00, 8.0000e+00],
#        [1.0000e+00, 1.8367e-40]])
a.resize(4, 1)
#tensor([[6.0000e+00],
#        [8.0000e+00],
#        [1.0000e+00],
#        [1.8367e-40]])
```

#### transpose / permute 各维度之间的变换，

`transpose`  可以将指定的两个维度的元素进行**转置**， `permute`  则可以按照指定的维度进行**维度变换**。

```python
>>> x
tensor([[[-0.9699, -0.3375, -0.0178]],
        [[ 1.4260, -0.2305, -0.2883]]])

>>> x.shape
torch.Size([2, 1, 3])
>>> x.transpose(0, 1) # shape => torch.Size([1, 2, 3])
tensor([[[-0.9699, -0.3375, -0.0178],
         [ 1.4260, -0.2305, -0.2883]]])
>>> x.permute(1, 0, 2) # shape => torch.Size([1, 2, 3])
tensor([[[-0.9699, -0.3375, -0.0178],
         [ 1.4260, -0.2305, -0.2883]]])
>>> 
```

#### squeeze(dim) / unsquence(dim)

处理size为1的维度， 前者用于去除size为1的维度， 而后者则是将指定的维度的size变为1

```python
>>> a = torch.arange(1, 4)
>>> a
tensor([1, 2, 3]) # shape => torch.Size([3])
>>> a.unsqueeze(0) # shape => torch.Size([1, 3])
>>> a.unqueeze(0).squeeze(0) # shape => torch.Size([3])
```

#### expand / expand_as / repeat复制元素来扩展维度

有时需要采用复制的形式来扩展 Tensor 的维度， 这时可以使用 `expand`， `expand()` 函数将 size 为 1的维度复制扩展为指定大小， 也可以用 `expand_as() `函数指定为 示例 Tensor 的维度。

!! `expand` 扩大 tensor 不需要分配新内存，只是仅仅新建一个 tensor 的视图，其中通过将 stride 设为0，一维将会扩展位更高维。

`repeat` 沿着指定的维度重复 tensor。 不同于 `expand()`，复制的是 tensor 中的数据。

```python
>>> a = torch.rand(2, 2, 1)
>>> a
tensor([[[0.3094],
         [0.4812]],

        [[0.0950],
         [0.8652]]])
>>> a.expand(2, 2, 3) # 将第2维的维度由1变为3， 则复制该维的元素，并扩展为3
tensor([[[0.3094, 0.3094, 0.3094],
         [0.4812, 0.4812, 0.4812]],

        [[0.0950, 0.0950, 0.0950],
         [0.8652, 0.8652, 0.8652]]])

>>> a.repeat(1, 2, 1) # 将第二位复制一次
tensor([[[0.3094],
         [0.4812],
         [0.3094],
         [0.4812]],

        [[0.0950],
         [0.8652],
         [0.0950],
         [0.8652]]])
```

#### 使用切片操作扩展多个维度

```python
b = a[:,None, None,:] # None 处的维度为１
```



### 21.组合与分块

#### 组合

将不同的 Tensor 叠加起来。 主要有 `cat()` 和 `torch.stack()` 两个函数，cat 即 concatenate 的意思， 是指沿着已有的数据的某一维度进行拼接， 操作后的数据的总维数不变， 在进行拼接时， 除了拼接的维度之外， 其他维度必须相同。 而` torch. stack()` 函数会新增一个维度， 并按照指定的维度进行叠加。

```python
torch.cat(list_of_tensors, dim=0)　  # k 个 (m,n) -> (k*m, n)
torch.stack(list_of_tensors, dim=0)   # k 个 (m,n) -> (k*m*n)
```

#### 分块

指将 Tensor 分割成不同的子 Tensor，主要有 `torch.chunk()` 与 `torch.split()` 两个函数，前者需要指定分块的数量，而后者则需要指定每一块的大小，以整形或者list来表示。

```python
>>> a = torch.Tensor([[1,2,3], [4,5,6]])
>>> torch.chunk(a, 2, 0)
(tensor([[1., 2., 3.]]), tensor([[4., 5., 6.]]))
>>> torch.chunk(a, 2, 1)
(tensor([[1., 2.],
        [4., 5.]]), tensor([[3.],
        [6.]]))
>>> torch.split(a, 2, 0)
(tensor([[1., 2., 3.],
        [4., 5., 6.]]),)
>>> torch.split(a, [1, 2], 1)
(tensor([[1.],
        [4.]]), tensor([[2., 3.],
        [5., 6.]]))
```



### 22.Linear algebra

```python
diag  # 对角线元素
triu/tril  # 矩阵的上三角/下三角
addmm/addbmm/addmv/addr/badbmm...  # 矩阵运算
t # 转置
dor/cross # 内积/外积
inverse # 矩阵求逆
svd  # 奇异值分解

torch.mm(tensor1, tensor2)   # 矩阵乘法  (m*n) * (n*p) -> (m*p)
torch.bmm(tensor1, tensor2) # batch的矩阵乘法: (b*m*n) * (b*n*p) -> (b*m*p).
torch.mv(tensor, vec) #　矩阵向量乘法 (m*n) * (n) = (m)
tensor1 * tensor2 # Element-wise multiplication.
```



### 23.基本机制

#### 广播机制

不同形状的 Tensor 进行计算时， 可以自动扩展到较大的相同形状再进行计算。 广播机制的前提是一个 Tensor 至少有一个维度，且从尾部遍历 Tensor 时，两者维度必须相等， 其中七个要么是1， 要么不存在

#### 向量化操作

可以在同一时间进行批量地并行计算，例如矩阵运算，以达到更高的计算效率的一种方式

#### 共享内存机制

(1) 直接通过 Tensor 来初始化另一个 Tensor， 或者通过 Tensor 的组合、分块、索引、变形来初始化另一个Tensor， 则这两个 Tensor 共享内存:

```python
>>> a = torch.randn(2,3)
>>> b = a
>>> c = a.view(6)
>>> b[0, 0] = 0
>>> c[3] = 4
>>> a
tensor([[ 0.0000,  0.3898, -0.7641],
        [ 4.0000,  0.6859, -1.5179]])
```

(2) 对于一些操作通过加后缀 “\_” 实现 inplace 操作， 如 `add_()` 和 `resize_()` 等， 这样操作只要被执行， 本身的 Tensor 就会被改变。

```python
>>> a
tensor([[ 0.0000,  0.3898, -0.7641],
        [ 4.0000,  0.6859, -1.5179]])
>>> a.add_(a)
tensor([[ 0.0000,  0.7796, -1.5283],
        [ 8.0000,  1.3719, -3.0358]])
```

(3) Tensor与 Numpy 可以高效的完成转换， 并且转换前后的变量共享内存。在进行 Pytorch 不支持的操作的时候， 甚至可以曲线救国， 将 Tensor 转换为 Numpy 类型，操作后再转化为 Tensor

```python
# tensor <--> numpy
b = a.numpy() # tensor -> numpy
a = torch.from_numpy(a) # numpy -> tensor
```

!!! 需要注意的是，`torch.tensor()` 总是会进行数据拷贝，新 tensor 和原来的数据不再共享内存。所以如果你想共享内存的话，建议使用 `torch.from_numpy()` 或者 `tensor.detach()` 来新建一个 tensor, 二者共享内存。





### 24.nn

```python
from torch import nn
import torch.nn.functional as F
```

#### pad填充

```python
nn.ConstantPad2d(padding, value)
```

#### 卷积和反卷积

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True,dilation=1)
```

```python
# 最常用的两种卷积层设计 3x3 & 1x1
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
```

#### 池化层

```python
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
nn.AdaptiveMaxPool2d(output_size, return_indices=False)
nn.AdaptiveAvgPool2d(output_size) # global avg pool:output_size=1
nn.MaxUnpool2d(kernel_size,stride=None,padding=0)
```

#### 全连接层

```python
nn.Linear(in_features, out_features, bias=True)
```

#### 防止过拟合相关层

```python
nn.Dropout2d(p=0.5,inplace=False)
nn.AlphaDropout(p=0.5)
nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True)
```

#### 激活函数

```python
nn.Softplus(beta=1, threshold=20)
nn.Tanh()
nn.ReLU(inplace=False)
nn.ReLU6(inplace=False)
nn.LeakyReLU(negative_slope=0.01, inplace=False)
nn.PReLU(num_parameters=1, init=0.25)
nn.SELU(inplace=False)
nn.ELU(alpha=1.0,inplace=False)
```

#### RNN

```python
nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
nn.RNN(*args, **kwargs)
nn.LSTMCell(input_size, hidden_size, bias=True)
nn.LSTM(*args, **kwargs)
nn.GRUCell(input_size, hidden_size, bias=True)
nn.GRU(*args, **kwargs)
```

#### Embedding

```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None)
```

#### Sequential

```python
nn.Sequential(*args)
```

#### loss function

```python
nn.BCELoss(weight=None, size_average=True, reduce=True)
nn.CrossEntropyLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)
# CrossEntropyLoss 等价于 log_softmax + NLLLoss
nn.L1Loss(size_average=True, reduce=True)
nn.KLDivLoss(size_average=True, reduce=True)
nn.MSELoss(size_average=True, reduce=True)
nn.NLLLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)
nn.NLLLoss2d(weight=None, size_average=True, ignore_index=-100, reduce=True)
nn.SmoothL1Loss(size_average=True, reduce=True)
nn.SoftMarginLoss(size_average=True, reduce=True)
nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-06, swap=False, size_average=True, reduce=True)
nn.CosineEmbeddingLoss(margin=0, size_average=True, reduce=True)
```

#### functional

```python
nn.functional # nn中的大多数layer，在functional中都有一个与之相对应的函数。
              # nn.functional中的函数和nn.Module的主要区别在于，
              # 用nn.Module实现的layers是一个特殊的类，都是由 class layer(nn.Module)定义，
              # 会自动提取可学习的参数。而nn.functional中的函数更像是纯函数，
              # 由def function(input)定义。
```

#### init

```python
torch.nn.init.uniform
torch.nn.init.normal
torch.nn.init.kaiming_uniform
torch.nn.init.kaiming_normal
torch.nn.init.xavier_normal
torch.nn.init.xavier_uniform
torch.nn.init.sparse
```

#### net

```python
class net_name(nn.Module):
    def __init__(self):
        super(net_name, self).__init__()
        self.layer_name = xxxx
    def forward(self, x):
        x = self.layer_name(x)
        return x
net.parameters()   # 获取参数 
net.named_parameters  # 获取参数及名称
net.zero_grad()  # 网络所有梯度清零, grad 在反向传播过程中是累加的(accumulated)，
                 # 这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。
```



### 25.optim -> form torch import optim

```python
import torch.optim as optim

optim.SGD(params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
optim.Optimizer(params, defaults)

optimizer.zero_grad()  # 等价于 net.zero_grad() 
optimizer.step()
```



### 26.learning rate

```python
# Reduce learning rate when validation accuracy plateau.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',patience=5, verbose=True)

# Cosine annealing learning rate.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

#Reduce learning rate by 10 at given epochs.
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,70],gamma=0.1)

#Learning rate warmup by 10 epochs.
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t:t/10)
for t in range(0,10):
    scheduler.step()
    train(...);val(...)
```



### 27.torchvision

#### models

```python
from torchvision import models
resnet34 = models.resnet34(pretrained=True, num_classes=1000)
```

#### data augmentation

```python
from torchvision import transforms

# transforms.CenterCrop           transforms.Grayscale           transforms.ColorJitter          
# transforms.Lambda               transforms.Compose             transforms.LinearTransformation 
# transforms.FiveCrop             transforms.Normalize           transforms.functional           
# transforms.Pad                  transforms.RandomAffine        transforms.RandomHorizontalFlip  
# transforms.RandomApply          transforms.RandomOrder         transforms.RandomChoice         
# transforms.RandomResizedCrop    transforms.RandomCrop          transforms.RandomRotation        
# transforms.RandomGrayscale      transforms.RandomSizedCrop     transforms.RandomVerticalFlip   
# transforms.ToTensor             transforms.Resize              transforms.transforms                                           
# transforms.TenCrop              transforms.Scale               transforms.ToPILImage
```

#### 自定义dataset

```python
from torch.utils.data import Dataset
class my_data(Dataset):
    def __init__(self,image_path,annotation_path,transform=None):
    	# 初始化，读取数据集
    def __len__(self):
        # 获取数据集的总大小
    def __getitem__(self, id):
        # 对于制定的id，读取该数据并返回
```

#### datasets

```python
from torch.utils.data import Dataset,Dataloader
from torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # convert to Tensor
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))  # Normalization
])
dataset = ImageFloder(root, transform=transform, target_transform=None, loader=default_loader)
dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=1, shuffle=True)
for batch_datas, batch_labels in dataloader:
    ...
```

#### img process

```python
img = make_grid(next(dataiter)[0],4)
save_image(img,'a.png')
```

#### data Visualiztion

```python
from torchvision.transforms import ToPILImage

show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

(data, label) = trainset[100]
show((data + 1) / 2).resize((100, 100))  # 应该会自动乘以 255 的
```



### 28.Code Samples

```python
# torch.device object used throughout this script
device = torch.device("cuda" if use_cuda else "cpu")

model = MyRNN().to(device)

# train
total_loss = 0
for input, target in train_loader:
    input, target = input.to(device),target.to(device)
    hidden = input.new_zeros(*h_shape)  # has the same device & dtype as 'input'
    ... # get loss and optimize
    total_loss += loss.item()

# evaluate
with torch.no_grad():
    for input,target in test_loader:
        ...
```



### 29.jit & torchscript

```python
from torch.jit import script, trace
torch.jit.trace(model, torch.rand(1,3,224,224))  # export model
@torch.jit.script
```

```javascript
#include <torch/torch.h>
#include <torch/script.h>

# img blob -> img tensor
torch::Tensor img_tensor = torch::from_blob(img.data,{1,image.rows, image.cols,3},torch::kByte);
img_tensor = img_tensor.permute({0,3,1,2});
img_tensor = img_tensor.toType(torch::kFloat);
img_tensor = img_tensor.div(255);
# load model
std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("resnet.pt");
# forward
torch::Tensor output = module -> forward({img_tensor}).toTensor();
```



### 30.onnx

```python
torch.onnx.export(model, dummy data,xxx.proto) # exports an ONNX formatted
model = onnx.load("alexnet.proto") # load an onnx model
onnx.checker.check_model(model)  #check that the model
onnx.helper.printable_graph(model.graph) # print a human readable representation of the graph
```

### 31.Distributed Training

```python
import torch.distrubuted as dist  # distributed communication
from multiprocessing import Process # memory sharing processes
```





## 三、模型定义

### 1.卷积层

最常用卷积层配置是

```python
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1, bias = True)
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding=0,bias = True)
```

如果卷积层配置比较复杂，不方便计算输出大小时，可以利用如下可视乎工具辅助：

[卷积可视化辅助工具](https://ezyang.github.io/convolution-visualizer/index.html)





### 2.GAP(Global average pooling) 层

```python
gap = torch.nn.AdaptiveAvgPool2d(output_size=1)
```

### 3.双线性汇合(bilinear pooling)

```python
X = torch.reshape(N,D,H*W)    # Assume X has shape N*D*H*W
X = torch.bmm(X, torch.transpose(X,1,2))/(H*W)  # Bilinear pooling

assert X.size() == (N,D,D)
X = torch.reshape(X,(N,D*D))

# Signed-sqrt normaliztion
X = torch.sign(X)*torch.sqrt(torch.abs(X)+1e-5)
# L2 normaliztion
X = torch.nn.functional.normalize(X)
```

### 4.多卡同步BN(Batch normalization)

当使用`torch.nn.DataParallel` 将代码运行在多张GPU卡上时，PyTorch的BN层默认操作是各卡上数据独立的计算均值和标准差，同步BN使用所有卡上的数据一起计算BN层的均值和标准差，缓解了当批量大小(batch size)比较小时对均值和标准差估计不准的情况，是在目标检测等任务中一个有效的提升性能的技巧。

现在PyTorch官方已经支持同步BN操作

```python
sync_bn = torch.nn.SyncBatchNorm(num_features, eps=1e-05, momentum=0.1, affine=True,track_running_stats = True)
```

将已有网络的所有BN层改为同步BN层

```python
def convertBNtoSyncBN(module, process_group=None):
    '''Recursively replace all BN layers to SyncBN layer.

    Args:
        module[torch.nn.Module]. Network
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        sync_bn = torch.nn.SyncBatchNorm(module.num_features, module.eps, module.momentum, 
                                         module.affine, module.track_running_stats, process_group)
        sync_bn.running_mean = module.running_mean
        sync_bn.running_var = module.running_var
        if module.affine:
            sync_bn.weight = module.weight.clone().detach()
            sync_bn.bias = module.bias.clone().detach()
        return sync_bn
    else:
        for name, child_module in module.named_children():
            setattr(module, name) = convert_syncbn_model(child_module, process_group=process_group))
        return module
```



### 5.类似BN滑动平均

如果要实现类似BN滑动平均的操作，在forward函数中要使用原地(inplace)操作给滑动平均赋值。

```python
class BN(torch.nn.Module)
	def __init__(self):
        ...
        self.register_buffer('running_mean',torch.zeros(num_features))
    def forward(self, X):
        ...
        self.running_mean += momentum * (current -self.running_mean)
```

### 6.计算模型整体参数量

```python
num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
```

### 7.类似Keras的model.summary()输出模型信息

[PyTorch Summary](https://github.com/sksq96/pytorch-summary)

### 8.模型权值初始化

注意`model.modules()` 和 `model.childern()`的区别：`model.modules()` 会迭代地遍历模型的所有子层，而`model.children()` 只会遍历模型下的一层。

```python
# Common practise for initialization.
for layer in model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                      nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

# Initialization with given tensor.
layer.weight = torch.nn.Parameter(tensor)
```

### 9.部分层使用预训练模型

如果保存的模型是`torch.nn.DataParallel` ，则当前的模型也需要是`torch.nn.DataParallel` 。

torch.nn.Dataparallen(model).module == model.

```python
model.load_state_dict(torch.load('model,pth'),strict=False)
```



### 10.将GPU保存的模型加载到CPU

```python
model.load_state_dict(torch.load('model,pth',map_location='cpu'))
```



## 四、数据准备、特征提取与微调

### 1.图像分块打散(image shuffle) / 区域混淆机制 (region confusion mechanism, RCM)

```python
# X is torch.Tensor of size N*D*H*W.
# Shuffle rows
Q = (torch.unsqueeze(torch.arange(num_blocks), dim=1) * torch.ones(1, num_blocks).long()
     + torch.randint(low=-neighbour, high=neighbour, size=(num_blocks, num_blocks)))
Q = torch.argsort(Q, dim=0)
assert Q.size() == (num_blocks, num_blocks)

X = [torch.chunk(row, chunks=num_blocks, dim=2)
     for row in torch.chunk(X, chunks=num_blocks, dim=1)]
X = [[X[Q[i, j].item()][j] for j in range(num_blocks)]
     for i in range(num_blocks)]

# Shulle columns.
Q = (torch.ones(num_blocks, 1).long() * torch.unsqueeze(torch.arange(num_blocks), dim=0)
     + torch.randint(low=-neighbour, high=neighbour, size=(num_blocks, num_blocks)))
Q = torch.argsort(Q, dim=1)
assert Q.size() == (num_blocks, num_blocks)
X = [[X[i][Q[i, j].item()] for j in range(num_blocks)]
     for i in range(num_blocks)]

Y = torch.cat([torch.cat(row, dim=2) for row in X], dim=1)
```



### 2.得到视频数据基本信息

```python
import cv2
video = cv2.VideoCapture(mp4_path)
height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
video.release()
```

### 3.TSN每段（segment）采样一帧视频（[参考](https://zhuanlan.zhihu.com/p/106788199#ref_3)）

```text
K = self._num_segments
if is_train:
    if num_frames > K:
        # Random index for each segment.
        frame_indices = torch.randint(
            high=num_frames // K, size=(K,), dtype=torch.long)
        frame_indices += num_frames // K * torch.arange(K)
    else:
        frame_indices = torch.randint(
            high=num_frames, size=(K - num_frames,), dtype=torch.long)
        frame_indices = torch.sort(torch.cat((
            torch.arange(num_frames), frame_indices)))[0]
else:
    if num_frames > K:
        # Middle index for each segment.
        frame_indices = num_frames / K // 2
        frame_indices += num_frames // K * torch.arange(K)
    else:
        frame_indices = torch.sort(torch.cat((                              
            torch.arange(num_frames), torch.arange(K - num_frames))))[0]
assert frame_indices.size() == (K,)
return [frame_indices[i] for i in range(K)]
```

### 4.提取ImageNet预训练模型某层的卷积特征

```python
# VGG-16 relu5-3 feature.
model = torchvision.models.vgg16(pretrained=True).features[:-1]
# VGG-16 pool5 feature.
model = torchvision.models.vgg16(pretrained=True).features
# VGG-16 fc7 feature.
model = torchvision.models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])

#ResNet GAP feature
model = torchvision.models.resnet18(pretrained=True)
model = torch.nn.Sequential(collection.OrderedDict(list(model.named_children()[:-1])))

with torch.no_grad():
    model.eval()
    conv_representation= model(image)
```

### 5.提取ImageNet预训练模型多层的卷积特征

```python
class FeatureExtractor(torch.nn.Module):
    """
	Helper class to extract several convolution features from the given pre_trained model
	
	Attributes:
		_model,torch.nn.Module.
		_layers_to_extract,list<str> or set<str>
    Example:
    	>>> model = torchvision.models.resnet152(pretrained=True)
    	>>> model = torch.nn.Sequential(collections.OrdereDict(list(model.named_children())[:-1]))
    	>>> conv_representation = FeatureExtractor(pretrained_model = model,
    		layers_to_extract={'layer1','layer2','layer3','layer4'})(image)
"""
    def __init__(self,pretrained_model, layers_to_extract):
        torch.nn.Module.__init__(self)
        self._model = pretrained_model
        self._model.eval()
        self._layers_to_extract = set(layers_to_extract)
    def forward(self,x):
        with torch.no_grad():
            conv_representation = []
            for name, layer in self._model.named_children():
                x = layer(x)
                if name in self._layers_to_extract:
                    conv_representation.append(x)
                return conv_representaion
    
```

### 6.其他预训练模型

[预训练模型](https://github.com/Cadene/pretrained-models.pytorch)

### 7.微调全连接层

```python
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 100) # Replace the last fc layer
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
```

以较大学习率微调全连接层，较小学习率调卷积层

```python
model = torchvision.models.resnet18(pretrained=True)
finetuned_parameters = list(map(id,model.fc.parameters()))
conv_parameters = (p for p in model.parameters() if id(p) not in finetuned_parameters)
parameters = [{'params':conv_parameters, 'lr':1e-3},{'params':model.fc.parameters()}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
```





## 五、模型训练

### 1.常用训练和验证数据预处理

其中ToTensor操作会将`PIL.Image` 或形状为H*W\*D,数值范围为[0,255]的`np.ndarray` 转换为形状为D\*H\*W，数值范围为[0.0, 1.0]的torch.Tensor。

```python
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(size=224,
                                             scale=(0.08, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
 ])
 val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
])
```



### 3.训练代码基本框架

```python
for t in epoch(80):
    for images, labels in tqdm.tqdm(train_loader, desc='Epoch %3d' % (t+1)):
        images, labels = images.cuda(), labels.cuda()
        scores = model(images)
        loss = loss_function(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

将整数标记转换为独热吗

```python
N = tensor.size(0)
one_hot = torch.zeros(N, num_classes).long()
ont_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())
```



### 4.标记平滑(lagbel smoothing)

```python
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    N = labels.size(0)
    # C is the number of classes.
    smoothed_labels = torch.full(size=(N, C),fill_value=0.1 / (C-1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1),value-0.9)
    
    score = model(images)
    log_prob = torch.nn.functional.log_softmax(score, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels)/N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 5.Mixup([参考](https://zhuanlan.zhihu.com/p/106788199#ref_5))

```python
beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
for images,labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    # Mixup images.
    lambda_ = beat_distribution.sample([]).item()
    index = torch.randperm(images.size(0)).cuda()
    mixed_images = lambda_ * images + (1-lambda_) * images[index,:]
    
    # Mixup loss.
    scores = model(mixed_images)
    loss = (lambda_ * loss_function(scores,labels)
           + (1-lambda_) * loss_function(scores,labels[index]))
    
    optimizer.zero_grad()
    loss.backwrad()
    optimizer.step()
```

### 6.L1正则化

```python
l1_regularization = torch.nn.L1Loss(reduction='sum')
loss = ...  # Standard cross-entropy loss
for param in model.parameters():
    loss += lambda_ * torch.sum(torch.abs(param))
loss.backward()
```

### 7.不对偏置项进行L2正则化/权值衰减(weight decay)

```python
bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},                
              {'parameters': others_list}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
```

### 8.梯度裁剪(gradient clipping)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=20)
```

### 9.计算Softmax输出的准确率

```python
score = model(images)
prediction = torch.argmax(score, dim=1)
num_correct = torch.sum(prediction == labels).item()
accuruacy = num_correct / labels.size(0)
```

### 10.可视化模型前馈的计算图

https://link.zhihu.com/?target=https%3A//github.com/szagoruyko/pytorchviz

### 11.可视化学习曲线

有Facebook自己开发的`Visdom` 和`Tensorboard`两个选择。

[visdom](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/visdom)

[torch.utils.tensorboard](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/tensorboard.html)

```python
# Example using Visdom.
vis = visdom.Visdom(env='Learning curve', use_incoming_socket=False)
assert self._visdom.check_connection()
self._visdom.close()
options = collections.namedtuple('Options', ['loss', 'acc', 'lr'])(
    loss={'xlabel': 'Epoch', 'ylabel': 'Loss', 'showlegend': True},
    acc={'xlabel': 'Epoch', 'ylabel': 'Accuracy', 'showlegend': True},
    lr={'xlabel': 'Epoch', 'ylabel': 'Learning rate', 'showlegend': True})

for t in epoch(80):
    tran(...)
    val(...)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([train_loss]),
             name='train', win='Loss', update='append', opts=options.loss)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([val_loss]),
             name='val', win='Loss', update='append', opts=options.loss)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([train_acc]),
             name='train', win='Accuracy', update='append', opts=options.acc)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([val_acc]),
             name='val', win='Accuracy', update='append', opts=options.acc)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([lr]),
             win='Learning rate', update='append', opts=options.lr)
```

### 12.得到当前学习率

```python
# If there is one global learning rate(which is the common case).
lr = next(iter(optimizer.param_groups))['lr']

# If there are multiple learning rates for different layers.
all_lr = []
for param_group in optimizer.param_groups:
    all_lr.append(param_group['lr'])
```

### 13.学习率衰减

```python
# Reduce learning rate when validation accuarcy plateau.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
for t in range(0, 80):
    train(...); val(...)
    scheduler.step(val_acc)

# Cosine annealing learning rate.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
# Reduce learning rate by 10 at given epochs.
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
for t in range(0, 80):
    scheduler.step()    
    train(...); val(...)

# Learning rate warmup by 10 epochs.
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: t / 10)
for t in range(0, 10):
    scheduler.step()
    train(...); val(...)
```

### 14.保存与加载断点

注意为了能够恢复训练，我们需要同时保存模型和优化器的状态，以及当前的训练轮数。

```python
# Save checkpoint.
is_best = current_acc > best_acc
best_acc = max(best_acc,current_acc)
checkpoint = {
    'best_acc':best_acc,
    'epoch':t+1,
    'model':model.state_dict(),
    'optimizer':optimizer.state_dict(),
}
model_path = os.path.join('model', 'checkpoint.pth.tar')
torch.save(checkpoint,model_path)
if is_best:
    shutil.copy('checkpoint.pth.tar', model_path)

# Load checkpoint
if resume:
    model_path = os.path.join('model', 'checkpoint.pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Load checkpoint at epoch %d.' % start_epoch)
```

save and load model

```python
torch.save(model.state_dict(), 'xxx_params.pth')
mode.load_state_dict(t.load('xxx_params.pth'))

torch.save(model,'xxx.pth')
model.torch.load('xxxx.pth')

all_data = dict(
	optimizer=optimizer.state_dict(),
  	model = model.state_dict(),
    info = u'model and optim parameter'
)
t.save(all_data, 'xxx.pth')
all_data = t.load('xxx.pth')
all_data.keys()
```





### 15.计算准确率、查准率(precision)、查全率(recall)

```python
# data['label'] and data['prediction'] are groundtruth label and prediction 
# for each image, respectively.
accuracy = np.mean(data['label'] == data['prediction']) * 100

# Compute recision and recall for each class.
for c in range(len(num_classes)):
    tp = np.dot((data['label'] == c).astype(int),
                (data['prediction'] == c).astype(int))
    tp_fp = np.sum(data['prediction'] == c)
    tp_fn = np.sum(data['label'] == c)
    precision = tp / tp_fp * 100
    recall = tp / tp_fn * 100
```



### 16.计算模型参数量

```python
# Total parameters
num_params = sum(p.numel() for p in model.parameters())

# Trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```



### 17.模型权值

注意`model.modules()`和`model.children()`的区别：`model.modules()`会迭代地遍历模型的所有子层，而`model.children()`只会遍历模型下的一层。

```python
# Common paractise for initialization
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, val=0.0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
# Initializtion with given tensor.
m.weight = torch.nn.Parameter(tensor)
```



### 18.冻结参数

```python
if not requires_grad:
    for param in self.parameters():
        param.requires_grad = False
```





## 六、模型测试

### 1.计算每个类别的查准率(precision)、查全率(recall)、F1和总体指标

```python
import sklearn.metrics
all_label=[]
all_prediction=[]
for images, labels in tqdm.tqdm(data_loader):
    # Data.
    images, labels = images.cuda(),labels.cuda()
    # Forward pass.
    score = model(images)
    # Save label and predictions.
    prediction = torch.argmax(score,dim=1)
    all_label.append(labels.cpu().numpy())
    all_prediction.append(prediction.cpu().numpy())
# Compute RP and confusion matrix
all_label = np.concatenate(all_label)
assert len(all_label.shape) == 1
all_prediction = np.concatenate(all_prediction)
assert all_label.shape == all_prediction.shape

micro_p, micro_r, micro_f1, _ = sklearn.metrics.precision_recall_fscore_support(
     all_label, all_prediction, average='micro', labels=range(num_classes))
class_p, class_r, class_f1, class_occurence = sklearn.metrics.precision_recall_fscore_support(
     all_label, all_prediction, average=None, labels=range(num_classes))
# Ci,j = #{y=i and hat_y=j}
confusion_mat = sklearn.metrics.confusion_matrix(
     all_label, all_prediction, labels=range(num_classes))
assert confusion_mat.shape == (num_classes, num_classes)
```

### 2.将各类结果写入表格

```python
import csv

# Write results onto disk.
with open(os.path.join(path, filename), 'wt', encoding='utf-8') as f:
     f = csv.writer(f)
     f.writerow(['Class', 'Label', '# occurence', 'Precision', 'Recall', 'F1',
                 'Confused class 1', 'Confused class 2', 'Confused class 3',
                 'Confused 4', 'Confused class 5'])
     for c in range(num_classes):
         index = np.argsort(confusion_mat[:, c])[::-1][:5]
         f.writerow([
             label2class[c], c, class_occurence[c], '%4.3f' % class_p[c],
                 '%4.3f' % class_r[c], '%4.3f' % class_f1[c],
                 '%s:%d' % (label2class[index[0]], confusion_mat[index[0], c]),
                 '%s:%d' % (label2class[index[1]], confusion_mat[index[1], c]),
                 '%s:%d' % (label2class[index[2]], confusion_mat[index[2], c]),
                 '%s:%d' % (label2class[index[3]], confusion_mat[index[3], c]),
                 '%s:%d' % (label2class[index[4]], confusion_mat[index[4], c])])
         f.writerow(['All', '', np.sum(class_occurence), micro_p, micro_r, micro_f1, 
                     '', '', '', '', ''])
```

## 七、PyTorch其他注意事项

### 1.模型定义

建议有参数的层和汇合（pooling）层使用`torch.nn` 模块定义，激活函数直接使用`torch.nn.functional` 。`torch.nn` 模块和`torch.nn.functional` 的区别在于，torch.nn模块在计算时底层调用了torch.nn.functional，但torch.nn模块包括该层参数，还可以应对训练和测试两种网络状态。使用torch.nn.functional时要注意网络状态，如

```python
def forward(self,x):
    ...
    x = torch.nn.functional.dropout(x,p=0.5,training=self.training)
```

- model(x)前用`model.train()` 和 `model.eval()` 切换网络状态。
- 不需要计算梯度的代码块用`with torch.no_grad()` 包含起来。`model.eval()` 和 `torch.no_grad()` 的区别在于，`model.eval()` 是将网络切换为测试状态，例如BN和随机失活(dropout)在训练和测试阶段使用不同的计算方法。`torch.no_grad()` 是关闭PyTorch张量的自动求导机制，以减少存储使用和加速计算，得到的结果无法进行`loss.backward()`.
- `torch.nn.CrossEntropyLoss` 的输入不需要经过Softmax。`torch.nn.CrossEntropyLoss` 等价于`torch.nn.functional.log_softmax` + `torch.nn.NLLLoss` 。
- `loss.backward()` 前用`optimizer.zero_grad` 清除累计梯度。`optimizer.zero_grad()` 和 `model.zero_grad()` 效果一样。

### 2.PyTorch性能与测试

- `torch.utils.data.DataLoader` 中尽量设置`pin_memory=True` ，对特别小的数据集如MNIST设置pin_memory=False反而更快一些。num_workers的设置需要在实验中找到最快的取值。

- 用del及时删除不用的中间变量，节约GPU存储。

- 使用inplace操作可节约GPU存储，如

  ```python
  x = torch.nn.functional.relu(x, inplace=True)
  ```



此外，还可以通过`torch.utils.checkpoint` 前向传播时只保留一部分中间结果来节约GPU存储使用，在反向传播时需要的内容从最近中间结果中计算得到。

- 减少CPU和GPU之间的数据传输。例如如果你想知道一个epoch中每个`mini-batch` 的loss和准确率，先将它们累积在GPU中等一个epoch结束之后一起传输回CPU会比每个mini-batch都进行一次GPU到CPU的传输更快。
- 使用半精度浮点数`half()` 会有一定的速度提升，具体效率依赖于GPU型号。需要小心数值精度过低带来的稳定性问题。
- 时常使用assert tensor.size() == (N, D, H, W)作为调试手段，确保张量维度和你设想中一致。
- 除了标记y外，尽量少使用一维张量，使用n*1的二维张量代替，可以避免一些意想不到的一维张量计算结果。
- 统计代码各部分耗时

```python
with torch.autograd.profiler.profile(enable=True, use_cuda=False) as profile:
    ...
print(profile)
```

或者在命令行运行

```shell
python -m torch.utils.bottleneck main.py
```

