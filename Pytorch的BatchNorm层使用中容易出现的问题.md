# Batch Normalization，批规范化

**B**atch **N**ormalization（简称为BN）[2]，中文翻译成**批规范化**，是在深度学习中普遍使用的一种技术，通常用于解决多层神经网络中间层的**协方差偏移**(Internal Covariate Shift)问题，类似于网络输入进行零均值化和方差归一化的操作，不过是在中间层的输入中操作而已，具体原理不累述了，见[2-4]的描述即可。

在BN操作中，最重要的无非是这四个式子：

![image-20201124101955665](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201124101955665.png).

注意到这里的最后一步也称之为**仿射**(affine)，引入这一步的目的主要是**设计一个通道，使得输出output至少能够回到输入input的状态（$$\gamma=1,\beta=0$$时）使得BN的引入至少不至于降低模型的表现，这是深度网络设计的一个套路。**
整个过程见流程图，BN在输入后插入，BN的输出作为规范后的结果输入的后层网络中。

![image-20201124102111230](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201124102111230.png)

- $$\gamma,\beta$$：分别是仿射中的$$weight$$ 和 $$bias$$，在pytorch中用`weight`和`bias`表示。
- $$\mu_{B}$$和 $$\sigma_{B}^2$$：和上面的参数不同，这两个是根据输入的batch的统计特性计算的，严格来说不算是“学习”到的参数，不过对于整个计算是很重要的。在pytorch中，这两个统计参数，用`running_mean`和`running_var`表示[5]，这里的`running`指的就是当前的统计参数不一定只是由当前输入的batch决定，还可能和历史输入的batch有关，详情见以下的讨论，特别是参数`momentum`那部分。



以图片输入作为例子，在`pytorch`中即是`nn.BatchNorm2d()`，我们实际中的BN层一般是对于通道进行的，举个例子而言，我们现在的输入特征（可以视为之前讨论的batch中的其中一个样本的shape）为$\mathbf{x} \in \mathbb{R}^{C \times W \times H}$（其中C是通道数，W是width，H是height），那么我们的$\mu_{\mathcal{B}} \in \mathbb{R}^{C}$,而方差$\sigma_{B}^{2} \in \mathbb{R}^{C}$。而仿射中$$weight$$ ,$\gamma \in \mathbb{R}^{C}$以及$$bias$$, $\beta \in \mathbb{R}^{C}$。我们会发现，这些参数，无论是学习参数还是统计参数都会通道数有关，其实在`pytorch`中，通道数的另一个称呼是`num_features`，也即是特征数量，因为不同通道的特征信息通常很不相同，因此需要隔离开通道进行处理。

有些朋友可能会认为这里的$$weight$$应该是一个张量，而不应该是一个矢量，其实不是的，这里的$$weight$$其实应该看成是 **对输入特征图的每个通道得到的归一化后的$\hat{\mathbf{X}}$进行尺度放缩的结果**，因此对于一个通道数为$$C$$输入特征图，那么每个通道都需要一个尺度放缩因子，同理，$$bias$$也是对于每个通道而言的。这里切勿认为$y_{i} \leftarrow \gamma \hat{x}_{i}+\beta$这一步是一个全连接层，他其实只是一个尺度放缩而已。关于这些参数的形状，其实可以直接从`pytorch`源代码看出，这里截取了`_NormBase`层的部分初始代码，便可一见端倪。

```python
class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()
```



# 在Pytorch中使用

Pytorch中的BatchNorm的API主要有：

```python
torch.nn.BatchNorm1d(num_features, 
                     eps=1e-05, 
                     momentum=0.1, 
                     affine=True, 
                     track_running_stats=True)
12345
```

一般来说pytorch中的模型都是继承`nn.Module`类的，都有一个属性`trainning`指定是否是训练状态，训练状态与否将会影响到某些层的参数是否是固定的，比如BN层或者Dropout层。通常用`model.train()`指定当前模型`model`为训练状态,`model.eval()`指定当前模型为测试状态。
同时，BN的API中有几个参数需要比较关心的，一个是`affine`指定是否需要仿射，还有个是`track_running_stats`指定是否跟踪当前batch的统计特性。容易出现问题也正好是这三个参数：`trainning`，`affine`，`track_running_stats`。

- 其中的`affine`指定是否需要仿射，也就是是否需要上面算式的第四个，如果`affine=False`，则$\gamma=1, \beta=0$，并且不能学习被更新。一般都会设置成`affine=True`[10]
- `trainning`和`track_running_stats`，`track_running_stats=True`表示跟踪整个训练过程中的batch的统计特性，得到方差和均值，而不只是仅仅依赖与当前输入的batch的统计特性。相反的，如果`track_running_stats=False`那么就只是计算当前输入的batch的统计特性中的均值和方差了。当在推理阶段的时候，如果`track_running_stats=False`，此时如果`batch_size`比较小，那么其统计特性就会和全局统计特性有着较大偏差，可能导致糟糕的效果。

一般来说，`trainning`和`track_running_stats`有四种组合[7]

1. `trainning=True`, `track_running_stats=True`。这个是期望中的训练阶段的设置，此时BN将会跟踪整个训练过程中batch的统计特性。
2. `trainning=True`, `track_running_stats=False`。此时BN只会计算当前输入的训练batch的统计特性，可能没法很好地描述全局的数据统计特性。
3. `trainning=False`, `track_running_stats=True`。这个是期望中的测试阶段的设置，此时BN会用之前训练好的模型中的（假设已经保存下了）`running_mean`和`running_var`并且**不会对其进行更新**。一般来说，只需要设置`model.eval()`其中`model`中含有BN层，即可实现这个功能。[6,8]
4. `trainning=False`, `track_running_stats=False` 效果同(2)，只不过是位于测试状态，这个一般不采用，这个只是用测试输入的batch的统计特性，容易造成统计特性的偏移，导致糟糕效果。

同时，**我们要注意到**，BN层中的`running_mean`和`running_var`的更新是在`forward()`操作中进行的，而不是`optimizer.step()`中进行的，因此如果处于训练状态，就算你不进行手动`step()`，BN的统计特性也会变化的。如

```python
model.train()
for data, label in self.dataloader:
    pred = model(data)
    # 在这里会更新model中的BN的统计特性参数, running_mean, running_var
    loss = self.loss(pred, label)
    # 就算不要下列三行代码,BN的统计特性也会变化
    opt.zero_grad()
    loss.backward()
    opt.step()
```

这个时候要将`model.eval()`转到测试阶段，才能固定住`running_mean`和`running_var`。有时候如果是先预训练模型然后加载模型，重新跑测试的时候结果不同，有一点性能上的损失，这个时候十有八九是`trainning`和`track_running_stats`设置的不对，这里需要多注意。 [8]

假设一个场景，如下图所示：

![image-20201124104823694](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201124104823694.png)

此时为了收敛容易控制，先预训练好模型`model_A`，并且`model_A`内含有若干BN层，后续需要将`model_A`作为一个`inference`推理模型和`model_B`联合训练，此时就希望`model_A`中的BN的统计特性值`running_mean`和`running_var`不会乱变化，因此就必须将`model_A.eval()`设置到测试模式，否则在`trainning`模式下，就算是不去更新该模型的参数，其BN都会改变的，这个将会导致和预期不同的结果。





### **问题解答:**

> 即使将track_running_stats设置为False，如果momentum不为None的话，还是会用滑动平均来计算running_mean和running_var的，而非是仅仅使用本batch的数据情况。而且关于冻结bn层，有一些更好的方法。

这里的`momentum`的作用，按照文档，这个参数是在对统计参数进行更新过程中，进行指数平滑使用的，比如统计参数的更新策略将会变成：
$\hat{x}_{\text {new }}=(1-\text { momentum }) \times \hat{x}+\text { momentum } \times x_{t}$
其中的更新后的统计参数，$$\hat{x}_{\text {new }}$$是根据当前观察$${x}_{\text {new }}$$和历史观察$$\hat{x}$$进行加权平均得到的（差分的加权平均相当于历史序列的指数平滑），默认的`momentum=0.1`。然而跟踪历史信息并且更新的这个行为是基于`track_running_stats`为`true`并且`training=true`的情况同时成立的时候，才会进行的，当在`track_running_stats=true, training=false`时(在默认的`model.eval()`情况下，即是之前谈到的四种组合的第三个，既满足这种情况)，将不涉及到统计参数的指数滑动更新了。[12,13]

这里引用一个不错的BN层冻结的例子，如：[14]

```python
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from apex.fp16_utils import *

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

model = models.resnet50(pretrained=True)
model.cuda()
model=network(model)
model.train()
model.apply(fix_bn) # fix batchnorm
input = Variable(torch.FloatTensor(8,3,224,224).cuda())
output = model(input)
output_mean = torch.mean(output)
output_mean.backward()
```





> 为什么模型测试时的参数为trainning=False, track_running_stats=True啊？？测试不是用训练时的滑动平均值吗？为什么track_running_stats=True呢？为啥要跟踪当前batch？？

我感觉这个问题问得挺好的，我们需要去翻下源码[15]，我们发现我们所有的`BatchNorm`层都有个共同的父类`_BatchNorm`，我们最需要关注的是`return F.batch_norm()`这一段，我们发现，其对`training`的判断逻辑是

```python
training=self.training or not self.track_running_stats
```

那么，其实其在`eval`阶段，这里的`track_running_stats`并不能设置为`False`，原因很简单，这样会使得上面谈到的`training=True`，导致最终的期望程序错误。至于设置了`track_running_stats=True`是不是会导致在`eval`阶段跟踪测试集的`batch`的统计参数呢？我觉得是不会的，我们追踪会发现[16]，整个流程的最后一步其实是调用了`torch.batch_norm()`，其是调用C++的底层函数，其参数列表可和`track_running_stats`一点关系都没有，只是由`training`控制，因此当`training=False`时，其不会跟踪统计参数的，只是会调用训练集训练得到的统计参数。（当然，时间有限，我也没有继续追到C++层次去看源码了）。

```python
class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
```

```python
def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], bool, float, float) -> Tensor  # noqa
    r"""Applies Batch Normalization for each channel across a batch of data.

    See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
    :class:`~torch.nn.BatchNorm3d` for details.
    """
    if not torch.jit.is_scripting():
        if type(input) is not Tensor and has_torch_function((input,)):
            return handle_torch_function(
                batch_norm, (input,), input, running_mean, running_var, weight=weight,
                bias=bias, training=training, momentum=momentum, eps=eps)
    if training:
        _verify_batch_size(input.size())

    return torch.batch_norm(
        input, weight, bias, running_mean, running_var,
        training, momentum, eps, torch.backends.cudnn.enabled
```





# Reference

[1]. [用pytorch踩过的坑](https://blog.csdn.net/LoseInVain/article/details/82916163)
[2]. Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[C]// International Conference on International Conference on Machine Learning. JMLR.org, 2015:448-456.
[3]. [<深度学习优化策略-1>Batch Normalization（BN）](https://zhuanlan.zhihu.com/p/26702482)
[4]. [详解深度学习中的Normalization，BN/LN/WN](https://zhuanlan.zhihu.com/p/33173246)
[5]. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L23-L24
[6]. https://discuss.pytorch.org/t/what-is-the-running-mean-of-batchnorm-if-gradients-are-accumulated/18870
[7]. [BatchNorm2d增加的参数track_running_stats如何理解？](https://www.zhihu.com/question/282672547)
[8]. [Why track_running_stats is not set to False during eval](https://discuss.pytorch.org/t/why-track-running-stats-is-not-set-to-false-during-eval/25412)
[9]. [How to train with frozen BatchNorm?](https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106)
[10]. [Proper way of fixing batchnorm layers during training](https://discuss.pytorch.org/t/proper-way-of-fixing-batchnorm-layers-during-training/13214)
[11]. [大白话《Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift》](https://zhuanlan.zhihu.com/p/33101420)
[12]. https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146/2
[13]. https://zhuanlan.zhihu.com/p/65439075
[14]. https://github.com/NVIDIA/apex/issues/122
[15]. https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d
[16]. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#batch_norm