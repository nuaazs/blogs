**pytorch中的交叉熵**

pytorch的交叉熵`nn.CrossEntropyLoss`在训练阶段，里面是内置了`softmax`操作的，因此只需要喂入原始的数据结果即可，不需要在之前再添加`softmax`层。这个和tensorflow的`tf.softmax_cross_entropy_with_logits`如出一辙.[1][2]pytorch的交叉熵`nn.CrossEntropyLoss`在训练阶段，里面是内置了`softmax`操作的，因此只需要喂入原始的数据结果即可，不需要在之前再添加`softmax`层。这个和tensorflow的`tf.softmax_cross_entropy_with_logits`如出一辙.[1][2]

------

## **pytorch中的MSELoss和KLDivLoss**

在深度学习中，`MSELoss`均方差损失和`KLDivLoss`KL散度是经常使用的两种损失，在pytorch中，也有这两个函数，如:

```python
loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()
```

这个时候我们要注意到，我们的标签`target`是需要一个不能被训练的，也就是`requires_grad=False`的值，否则将会报错，出现如：

```python
AssertionError: nn criterions don’t compute the gradient w.r.t. targets - please mark these variables as volatile or not requiring gradients
```

我们注意到，其实不只是`MSELoss`，其他很多loss，比如交叉熵，KL散度等，其`target`都需要是一个不能被训练的值的，这个和TensorFlow中的`tf.nn.softmax_cross_entropy_with_logits_v2`不太一样，后者可以使用可训练的`target`，具体见[3]

## **在验证和测试阶段取消掉梯度（no_grad）**

一般来说，我们在进行模型训练的过程中，因为要监控模型的性能，在跑完若干个`epoch`训练之后，需要进行一次在**验证集**[4]上的性能验证。一般来说，在验证或者是测试阶段，因为只是需要跑个前向传播(forward)就足够了，因此不需要保存变量的梯度。保存梯度是需要额外显存或者内存进行保存的，占用了空间，有时候还会在验证阶段导致OOM(**O**ut **O**f **M**emory)错误，**因此我们在验证和测试阶段，最好显式地取消掉模型变量的梯度。** 在`pytroch 0.4`及其以后的版本中，用`torch.no_grad()`这个上下文管理器就可以了，例子如下：

```python
model.train()
# here train the model, just skip the codes
model.eval() # here we start to evaluate the model
with torch.no_grad():
 for each in eval_data:
  data, label = each
  logit = model(data)
  ... # here we just skip the codes
```

如上，我们只需要在加上上下文管理器就可以很方便的取消掉梯度。这个功能在`pytorch`以前的版本中，通过设置`volatile=True`生效，不过现在这个用法已经被抛弃了。

## **显式指定`model.train()`和`model.eval()`**

我们的模型中经常会有一些子模型，其在训练时候和测试时候的参数是不同的，比如`dropout`[6]中的丢弃率和`Batch Normalization`[5]中的![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma)和![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta)等，这个时候我们就需要显式地指定不同的阶段（训练或者测试），在`pytorch`中我们通过`model.train()`和`model.eval()`进行显式指定，具体如：

```python
model = CNNNet(params)
# here we start the training
model.train()
for each in train_data:
 data, label = each
 logit = model(data)
 loss = criterion(logit, label)
 ... # just skip
# here we start the evaluation

model.eval() 
with torch.no_grad(): # we dont need grad in eval phase
 for each in eval_data:
  data, label = each
  logit = model(data)
  loss = criterion(logit, label)
  ... # just skip
```

**注意，在模型中有BN层或者dropout层时，在训练阶段和测试阶段必须显式指定train()和eval()**。

## **关于`retain_graph`的使用**

在对一个损失进行反向传播时，在`pytorch`中调用`out.backward()`即可实现，给个小例子如：

```python
import torch
import torch.nn as nn
import numpy as np
class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,2)
        self.act = nn.ReLU()
    def forward(self,inputv):
        return self.act(self.fc1(inputv))
n = net()
opt = torch.optim.Adam(n.parameters(),lr=3e-4)
inputv = torch.tensor(np.random.normal(size=(4,10))).float()
output = n(inputv)
target = torch.tensor(np.ones((4,2))).float()
loss = nn.functional.mse_loss(output, target)
loss.backward() # here we calculate the gradient w.r.t the leaf
```

对`loss`进行反向传播就可以求得![[公式]](https://www.zhihu.com/equation?tex=%5Cdfrac%7B%5Cpartial%7B%5Cmathrm%7Bloss%7D%7D%7D%7B%5Cpartial%7Bw_%7Bi%2Cj%7D%7D%7D)，即是损失对于每个叶子节点的梯度。我们注意到，在`.backward()`这个API的文档中，有几个参数，如:

```python
backward(gradient=None, retain_graph=None, create_graph=False)
```

这里我们关注的是`retain_graph`这个参数，这个参数如果为`False`或者`None`则在反向传播完后，就释放掉构建出来的graph，如果为`True`则不对graph进行释放[7][8]。

我们这里就有个问题，我们既然已经计算忘了梯度了，为什么还要保存graph呢？直接释放掉等待下一个迭代不就好了吗，不释放掉不会白白浪费内存吗？我们这里根据[7]中的讨论，简要介绍下为什么在某些情况下需要保留graph。如下图所示，我们用代码构造出此graph:

```python
import torch
from torch.autograd import Variable
a = Variable(torch.rand(1, 4), requires_grad=True)
b = a**2
c = b*2
d = c.mean()
e = c.sum()
```



![img](https://pic1.zhimg.com/80/v2-9891a49d046d606313ef66c12b34ff08_720w.png)

如果我们第一次需要对末节点`d`进行求梯度，我们有:

```python
d.backward()
```

问题是在执行完反向传播之后，因为没有显式地要求它保留graph，系统对graph内存进行释放，如果下一步需要对节点`e`进行求梯度，那么将会因为没有这个graph而报错。因此有例子：

```python
d.backward(retain_graph=True) # fine
e.backward(retain_graph=True) # fine
d.backward() # also fine
e.backward() # error will occur!
```

利用这个性质在某些场景是有作用的，比如在对抗生成网络GAN中需要先对某个模块比如生成器进行训练，后对判别器进行训练，这个时候整个网络就会存在两个以上的`loss`，例子如:

```python
G_loss = ...
D_loss = ...

opt.zero_grad() # 对所有梯度清0
D_loss.backward(retain_graph=True) # 保存graph结构，后续还要用
opt.step() # 更新梯度，只更新D的，因为只有D的不为0

opt.zero_grad() # 对所有梯度清0
G_loss.backward(retain_graph=False) # 不保存graph结构了，可以释放graph，
# 下一个迭代中通过forward还可以build出来的
opt.step() # 更新梯度，只更新G的，因为只有G的不为0
```

这个时候就可以对网络中多个`loss`进行分步的训练了。

## **进行梯度累积，实现内存紧张情况下的大`batch_size`训练**

在上面讨论的`retain_graph`参数中，还可以用于**累积梯度**，在GPU显存紧张的情况下使用可以等价于用更大的`batch_size`进行训练。首先我们要明白，当调用`.backward()`时，其实是对损失到各个节点的梯度进行计算，计算结果将会保存在各个节点上，如果不用`opt.zero_grad()`对其进行清0，那么只要你一直调用`.backward()`梯度就会一直累积，相当于是在大的`batch_size`下进行的训练。我们给出几个例子阐述我们的观点。

```python
import torch
import torch.nn as nn
import numpy as np
class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,2)
        self.act = nn.ReLU()
    def forward(self,inputv):
        return self.act(self.fc1(inputv))
n = net()
inputv = torch.tensor(np.random.normal(size=(4,10))).float()
output = n(inputv)
target = torch.tensor(np.ones((4,2))).float()
loss = nn.functional.mse_loss(output, target)
loss.backward(retain_graph=True)
opt = torch.optim.Adam(n.parameters(),lr=0.01)
for each in n.parameters():
    print(each.grad)
```

第一次输出:

```python
tensor([[ 0.0493, -0.0581, -0.0451,  0.0485,  0.1147,  0.1413, -0.0712, -0.1459,
          0.1090, -0.0896],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000]])
tensor([-0.1192,  0.0000])
```

在运行一次`loss.backward(retain_graph=True)`，输出为:

```python
tensor([[ 0.0987, -0.1163, -0.0902,  0.0969,  0.2295,  0.2825, -0.1424, -0.2917,
          0.2180, -0.1792],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000]])
tensor([-0.2383,  0.0000])
```

同理，第三次：

```python
tensor([[ 0.1480, -0.1744, -0.1353,  0.1454,  0.3442,  0.4238, -0.2136, -0.4376,
          0.3271, -0.2688],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000]])
tensor([-0.3575,  0.0000])
```

运行一次`opt.zero_grad()`，输出为：

```text
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
tensor([0., 0.])
```

现在明白为什么我们一般在求梯度时要用`opt.zero_grad()`了吧，那是为什么不要这次的梯度结果被上一次给影响，但是在某些情况下这个‘影响’是可以利用的。

## **调皮的`dropout`**

这个在利用`torch.nn.functional.dropout`的时候，其参数为：

```python
torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)
```

注意这里有个`training`指明了是否是在训练阶段，是否需要对神经元输出进行随机丢弃，这个是需要自行指定的，即便是用了`model.train()`或者`model.eval()`都是如此，这个和`torch.nn.dropout`不同，因为后者是一个**层**(Layer)，而前者只是一个函数，不能纪录状态[9]。

## **嘿，检查自己，说你呢, `index_select`**

`torch.index_select()`是一个用于索引给定张量中某一个维度中元素的方法，其API手册如：

```python
torch.index_select(input, dim, index, out=None) → Tensor
Parameters: 
 input (Tensor) – 输入张量，需要被索引的张量
 dim (int) – 在某个维度被索引
 index (LongTensor) – 一维张量，用于提供索引信息
 out (Tensor, optional) – 输出张量，可以不填
```

其作用很简单，比如我现在的输入张量为`1000 * 10`的尺寸大小，其中`1000`为样本数量，`10`为特征数目，如果我现在需要指定的某些样本，比如第`1-100`,`300-400`等等样本，我可以用一个`index`进行索引，然后应用`torch.index_select()`就可以索引了，例子如：

```python
>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-0.4664,  0.2647, -0.1228, -1.1068],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> indices = torch.tensor([0, 2])
>>> torch.index_select(x, 0, indices) # 按行索引
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> torch.index_select(x, 1, indices) # 按列索引
tensor([[ 0.1427, -0.5414],
        [-0.4664, -0.1228],
        [-1.1734,  0.7230]])
```

然而有一个问题是，`pytorch`似乎在使用`GPU`的情况下，不检查`index`是否会越界，因此如果你的`index`越界了，但是报错的地方可能不在使用`index_select()`的地方，而是在后续的代码中，这个似乎就需要留意下你的`index`了。同时，`index`是一个`LongTensor`，这个也是要留意的。

------

## **悄悄地更新，BN层就是个小可爱**

在trainning状态下，BN层的统计参数`running_mean`和`running_var`是在调用`forward()`后就更新的，这个和一般的参数不同，容易造成疑惑，考虑到篇幅较长，请移步到[11]。

------

## **`F.interpolate`的问题**

我们经常需要对图像进行插值，而`pytorch`的确也是提供对以`tensor`形式表示的图像进行插值的功能，那就是函数`torch.nn.functional.interpolate`[12]，但是我们注意到这个插值函数有点特别，它是对以`batch`为单位的图像进行插值的，如果你想要用以下的代码去插值：

```python
image = torch.rand(3,112,112) # H = 112, W = 112, C = 3的图像
image = torch.nn.functional.interpolate(image, size=(224,224))
```

那么这样就会报错，因为此处的`size`只接受一个整数，其对W这个维度进行缩放，这里，`interpolate`会认为3是`batch_size`，因此如果需要对图像的H和W进行插值，那么我们应该如下操作：

```python
image = torch.rand(3,112,112) # H = 112, W = 112, C = 3的图像
image = image.unsqueeze(0) # shape become (1,3,112,112)
image = torch.nn.functional.interpolate(image, size=(224,224))
```

------

## **Reference**

[1]. **[Why does CrossEntropyLoss include the softmax function?](https://link.zhihu.com/?target=https%3A//discuss.pytorch.org/t/why-does-crossentropyloss-include-the-softmax-function/4420)**

[2]. **[Do I need to use softmax before nn.CrossEntropyLoss()?](https://link.zhihu.com/?target=https%3A//discuss.pytorch.org/t/do-i-need-to-use-softmax-before-nn-crossentropyloss/16739/2)**

[3]. **[tf.nn.softmax_cross_entropy_with_logits 将在未来弃用](https://link.zhihu.com/?target=https%3A//blog.csdn.net/LoseInVain/article/details/80932605)**

[4]. **[训练集，测试集，检验集的区别与交叉检验](https://link.zhihu.com/?target=https%3A//blog.csdn.net/LoseInVain/article/details/78108955)**

[5]. Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[J]. arXiv preprint arXiv:1502.03167, 2015.

[6]. Hinton G E, Srivastava N, Krizhevsky A, et al. Improving neural networks by preventing co-adaptation of feature detectors[J]. arXiv preprint arXiv:1207.0580, 2012. [7]. **[What does the parameter retain_graph mean in the Variable's backward() method?](https://link.zhihu.com/?target=https%3A//stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method)**

[8]. [https://pytorch.org/docs/stable/autograd.html?highlight=backward#torch.Tensor.backward](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/autograd.html%3Fhighlight%3Dbackward%23torch.Tensor.backward)

[9] [https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.functional.dropout](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%3Fhighlight%3Ddropout%23torch.nn.functional.dropout)

[10]. **[index_select doesnt return errors when out of bounds (GPU only) #571](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch/issues/571)**

[11]. **[Pytorch的BatchNorm层使用中容易出现的问题](https://link.zhihu.com/?target=https%3A//blog.csdn.net/LoseInVain/article/details/86476010)**

[12]. [https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.functional.html%23torch.nn.functional.interpolate)

