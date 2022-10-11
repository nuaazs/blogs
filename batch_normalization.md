|batch normalization|deep learning|mechine learning|

**1. What** is BN?
顾名思义，batch normalization嘛，就是“批规范化”咯。Google在ICML文中描述的非常清晰，即在每次SGD时，通过mini-batch来对相应的activation做规范化操作，**使得结果（输出信号各个维度）的均值为0，方差为1**. 

最后的“**scale and shift**”操作则是为了让因训练所需而“刻意”加入的BN能够有可能还原最初的输入（即当![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma%5E%7B%28k%29%7D%3D%5Csqrt%7BVar%5Bx%5E%7B%28k%29%7D%5D%7D%2C+%5Cbeta%5E%7B%28k%29%7D%3DE%5Bx%5E%7B%28k%29%7D%5D)），从而保证整个network的capacity。（有关capacity的解释：实际上BN可以看作是在原模型上加入的“新操作”，这个新操作很大可能会改变某层原来的输入。当然也可能不改变，不改变的时候就是“还原原来输入”。如此一来，既可以改变同时也可以保持原输入，那么模型的容纳能力（capacity）就提升了。）

![img](https://pic4.zhimg.com/50/9ad70be49c408d464c71b8e9a006d141_hd.jpg?source=1940ef5c)![img](https://pic4.zhimg.com/80/9ad70be49c408d464c71b8e9a006d141_720w.jpg?source=1940ef5c)

关于DNN中的normalization，大家都知道白化（whitening），只是在模型训练过程中进行白化操作会带来过高的计算代价和运算时间。

因此本文提出两种简化方式：

1. 直接对输入信号的每个维度做规范化（“normalize each scalar feature independently”）
2. 在每个mini-batch中计算得到mini-batch mean和variance来替代整体训练集的mean和variance. 这便是Algorithm 1.



**2. How** to Batch Normalize?
怎样学BN的参数在此就不赘述了，就是经典的chain rule：

![img](https://pic2.zhimg.com/50/beb44145200caafe24fe88e7480e9730_hd.jpg?source=1940ef5c)![img](https://pic2.zhimg.com/80/beb44145200caafe24fe88e7480e9730_720w.jpg?source=1940ef5c)



**3. Where** to use BN?
BN可以**应用于网络中任意的activation set**。文中还特别指出在CNN中，BN应作用在非线性映射前，即对![[公式]](https://www.zhihu.com/equation?tex=x%3DWu%2Bb)做规范化。另外对CNN的“权值共享”策略，BN还有其对应的做法（详见文中3.2节）。

**4. Why** BN?
好了，现在才是重头戏－－为什么要用BN？BN work的原因是什么？
说到底，BN的提出还是为了克服深度神经网络难以训练的弊病。其实BN背后的insight非常简单，只是在文章中被Google复杂化了。
首先来说说“Internal Covariate Shift”。文章的title除了BN这样一个关键词，还有一个便是“ICS”。大家都知道在统计机器学习中的一个经典假设是“源空间（source domain）和目标空间（target domain）的数据分布（distribution）是一致的”。如果不一致，那么就出现了新的机器学习问题，如，transfer learning/domain adaptation等。而covariate shift就是分布不一致假设之下的一个分支问题，它是指源空间和目标空间的条件概率是一致的，但是其边缘概率不同，即：对所有![[公式]](https://www.zhihu.com/equation?tex=x%5Cin+%5Cmathcal%7BX%7D),![[公式]](https://www.zhihu.com/equation?tex=P_s%28Y%7CX%3Dx%29%3DP_t%28Y%7CX%3Dx%29)，但是![[公式]](https://www.zhihu.com/equation?tex=P_s%28X%29%5Cne+P_t%28X%29). 大家细想便会发现，的确，对于神经网络的各层输出，由于它们经过了层内操作作用，其分布显然与各层对应的输入信号分布不同，而且差异会随着网络深度增大而增大，可是它们所能“指示”的样本标记（label）仍然是不变的，这便符合了covariate shift的定义。由于是对层间信号的分析，也即是“internal”的来由。
那么好，为什么前面我说Google将其复杂化了。其实如果严格按照解决covariate shift的路子来做的话，大概就是上“importance weight”（[ref](https://link.zhihu.com/?target=http%3A//120.52.72.36/www.jmlr.org/c3pr90ntcsf0/papers/volume8/sugiyama07a/sugiyama07a.pdf)）之类的机器学习方法。可是这里Google仅仅说“通过mini-batch来规范化某些层/所有层的输入，从而可以固定每层输入信号的均值与方差”就可以解决问题。如果covariate shift可以用这么简单的方法解决，那前人对其的研究也真真是白做了。此外，试想，均值方差一致的分布就是同样的分布吗？当然不是。显然，ICS只是这个问题的“包装纸”嘛，仅仅是一种high-level demonstration。
那BN到底是什么原理呢？说到底还是**为了防止“梯度弥散”**。关于梯度弥散，大家都知道一个简单的栗子：![[公式]](https://www.zhihu.com/equation?tex=0.9%5E%7B30%7D%5Capprox+0.04)。在BN中，是通过将activation规范为均值和方差一致的手段使得原本会减小的activation的scale变大。可以说是一种更有效的local response normalization方法（见4.2.1节）。

**5. When** to use BN?
OK，说完BN的优势，自然可以知道什么时候用BN比较好。例如，在神经网络训练时遇到收敛速度很慢，或梯度爆炸等无法训练的状况时可以尝试BN来解决。另外，在一般使用情况下也可以加入BN来加快训练速度，提高模型精度。

