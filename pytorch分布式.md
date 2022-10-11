## torch.distributed 包支持

`Pytorch` 中通过 `torch.distributed` 包提供分布式支持，包括 `GPU` 和 `CPU` 的分布式训练支持。`Pytorch` 分布式目前只支持 `Linux`。

在此之前，`torch.nn.DataParallel` 已经提供数据并行的支持，但是其不支持多机分布式训练，且底层实现相较于 `distributed` 的接口，有些许不足。

`torch.distributed` 的优势如下：

1. **每个进程对应一个独立的训练过程，且只对梯度等少量数据进行信息交换。**

在每次迭代中，每个进程具有自己的 `optimizer` ，并独立完成所有的优化步骤，进程内与一般的训练无异。

在各进程梯度计算完成之后，各进程需要将梯度进行汇总平均，然后再由 `rank=0` 的进程，将其 `broadcast` 到所有进程。之后，各进程用该梯度来更新参数。

由于各进程中的模型，初始参数一致 (初始时刻进行一次 `broadcast`)，而每次用于更新参数的梯度也一致，因此，各进程的模型参数始终保持一致。

而在 `DataParallel` 中，全程维护一个 `optimizer`，对各 `GPU` 上梯度进行求和，而在主 `GPU` 进行参数更新，之后再将模型参数 `broadcast` 到其他 `GPU`。

相较于 `DataParallel`，`torch.distributed` 传输的数据量更少，因此速度更快，效率更高。

1. **每个进程包含独立的解释器和 GIL**。

由于每个进程拥有独立的解释器和 `GIL`，消除了来自单个 `Python` 进程中的多个执行线程，模型副本或 `GPU` 的额外解释器开销和 `GIL-thrashing` ，因此可以减少解释器和 `GIL` 使用冲突。这对于严重依赖 `Python runtime` 的 `models` 而言，比如说包含 `RNN` 层或大量小组件的 `models` 而言，这尤为重要。

## RingAllReduce VS TreeAllReduce

`Pytorch 1.x` 的多机多卡计算模型并没有采用主流的 `Parameter Server` 结构，而是直接用了`Uber Horovod` 的形式，也是百度开源的 `RingAllReduce` 算法。

采用 `PS` 计算模型的分布式，通常会遇到网络的问题，随着 `worker` 数量的增加，其加速比会迅速的恶化，需要借助其他辅助技术。

由于某一个 `GPU` 需要接收其他所有 `GPU` 的梯度，并求平均以及 `broadcast` 回去，若 `GPU` 数量越大时，通信成本也就越高。其基本架构如下图所示。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/rO9lz.jpg)

而 `Uber` 的 `Horovod`，采用的 `RingAllReduce` 的计算方案，其特点是网络单次通信量不随着 `worker(GPU)` 的增加而增加，是一个恒定值。

在 `RingALll` 中，`GPU` 集群被组织成一个逻辑环，每个 `GPU` 只从左邻居接受数据、并发送数据给右邻居，即每次同步每个 `gpu` 只获得部分梯度更新，等一个完整的 `Ring` 完成，每个 `GPU` 都获得了完整的参数。

与 `TreeAllReduce` 不同，`RingAllreduce` 算法的每次通信成本是恒定的，与系统中 `gpu` 的数量无关，完全由系统中 `gpu` 之间最慢的连接决定。

其基本的结构和算法原理如下所示：

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/iWD1n.jpg)

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/4tTNM.jpg)

如上图所示的结构，在 `5` 次迭代之后，所有的 `GPU` 更新为了值之和。





## Pytorch 分布式使用流程

## 基本概念

下面是分布式系统中常用的一些概念：

- **`group`**：

即进程组。默认情况下，只有一个组，一个 `job` 即为一个组，也即一个 `world`。

当需要进行更加精细的通信时，可以通过 `new_group` 接口，使用 `word` 的子集，创建新组，用于集体通信等。

- **`world size`** ：

表示全局进程个数。

- **`rank`**：

表示进程序号，用于进程间通讯，表征进程优先级。`rank = 0` 的主机为 `master` 节点。

- **`local_rank`**：

进程内，`GPU` 编号，非显式参数，由 `torch.distributed.launch` 内部指定。比方说， `rank = 3，local_rank = 0` 表示第 `3` 个进程内的第 `1` 块 `GPU`。

## 基本使用流程

`Pytorch` 中分布式的基本使用流程如下：

1. 在使用 `distributed` 包的任何其他函数之前，需要使用 `init_process_group` 初始化进程组，同时初始化 `distributed` 包。
2. 如果需要进行小组内集体通信，用 `new_group` 创建子分组
3. 创建分布式并行模型 `DDP(model, device_ids=device_ids)`
4. 为数据集创建 `Sampler`
5. 使用启动工具 `torch.distributed.launch` 在每个主机上执行一次脚本，开始训练
6. 使用 `destory_process_group()` 销毁进程组

## 使用模板

下面以 `TCP` 初始化方式为例，共 `3` 太主机，每台主机 `2` 块 `GPU`，进行分布式训练。

### TCP 初始化方式

### 代码

```python
import torch.distributed as dist
import torch.utils.data.distributed

# ......
parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
parser.add_argument('--rank', default=0,
                    help='rank of current process')
parser.add_argument('--word_size', default=2,
                    help="word size")
parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',
                    help="init-method")
args = parser.parse_args()

# ......
dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.word_size)

# ......
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

# ......
net = Net()
net = net.cuda()
net = torch.nn.parallel.DistributedDataParallel(net)
```

### 说明

1. 在 `TCP` 方式中，在 `init_process_group` 中必须手动指定以下参数
2. `rank` 为当前进程的进程号
3. `word_size` 为当前 `job` 的总进程数
4. `init_method` 内指定 `tcp` 模式，且所有进程的 `ip:port` 必须一致，设定为主进程的 `ip:port`
5. 必须在 `rank==0` 的进程内保存参数。
6. 若程序内未根据 `rank` 设定当前进程使用的 `GPUs`，则默认使用全部 `GPU`，且以数据并行的方式使用。
7. 每条命令表示一个进程。若已开启的进程未达到 `word_size` 的数量，则所有进程会一直等待
8. 每台主机上可以开启多个进程。但是，若未为每个进程分配合适的 `GPU`，则同机不同进程可能会共用 `GPU`，应该坚决避免这种情况。
9. 使用 `gloo` 后端进行 `GPU` 训练时，会报错。
10. 若每个进程负责多块 `GPU`，可以利用多 `GPU` 进行模型并行。如下所示：

```python
class ToyMpModel(nn.Module):
    def init(self, dev0, dev1):
        super(ToyMpModel, self).init()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10,10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10,5).to(dev1)
    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
dev0 = rank * 2
dev1 = rank * 2 + 1
mp_model = ToyMpModel(dev0, dev1)
ddp_mp_model = DDP(mp_model)
    
```

### Env 初始化方式

### 代码

```python
import torch.distributed as dist
import torch.utils.data.distributed

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

dist.init_process_group(backend='nccl',init_method='env://')

trainset = torchvision.datasets.CIFAR10(root='./data',train=True, download=download,transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.Dataloader(trainset, batch_size=batch_size, sampler=train_sampler)

net = Net()
device = torch.device('cuda',args.local_rank)
net = net.to(device)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],output_device=args.local_rank)
```

### 执行方式

```python
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=3 --node_rank=0 --master_addr="192.168.1.201" --master_port=23456 env_init.py

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=3 --node_rank=1 --master_addr="192.168.1.201" --master_port=23456 env_init.py

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=3 --node_rank=2 --master_addr="192.168.1.201" --master_port=23456 env_init.py
```



### 说明

1. 在 `Env` 方式中，在 `init_process_group` 中，无需指定任何参数
2. 必须在 `rank==0` 的进程内保存参数。
3. 该方式下，使用 `torch.distributed.launch` 在每台主机上，为其创建多进程，其中:
   1. `nproc_per_node` 参数指定为当前主机创建的进程数。一般设定为当前主机的 `GPU` 数量
   2. `nnodes` 参数指定当前 `job` 包含多少个节点
   3. `node_rank` 指定当前节点的优先级
   4. `master_addr` 和 `master_port` 分别指定 `master` 节点的 `ip:port`
4. 若没有为每个进程合理分配 `GPU`，则默认使用当前主机上所有的 `GPU`。即使一台主机上有多个进程，也会共用 `GPU`。
5. 使用 `torch.distributed.launch` 工具时，将会为当前主机创建 `nproc_per_node` 个进程，每个进程独立执行训练脚本。同时，它还会为每个进程分配一个 `local_rank` 参数，表示当前进程在当前主机上的编号。例如：`rank=2, local_rank=0` 表示第 `3` 个节点上的第 `1` 个进程。
6. 需要合理利用 `local_rank` 参数，来合理分配本地的 `GPU` 资源
7. 每条命令表示一个进程。若已开启的进程未达到 `word_size` 的数量，则所有进程会一直等待







## 进程组

## 初始化进程组

### init_process_group

### 函数原型

```python
torch.distributed.init_process_group(backend,
                                    init_method=None,
                                    timeout=datetime.timedelta(0,1800),
                                    world_size=-1,
                                    rank=-1,
                                    store=None)
```

### 函数作用

该函数需要在**每个进程中**进行调用，用于**初始化该进程**。在使用分布式时，该函数必须在 `distributed` 内所有相关函数之前使用。

### 参数详解

- **backend** ：**指定当前进程要使用的通信后端**

小写字符串，支持的通信后端有 `gloo`，`mpi`，`nccl` 。建议用 `nccl`。

- **init_method** ： **指定当前进程组初始化方式**

可选参数，字符串形式。如果未指定 `init_method` 及 `store`，则默认为 `env://`，表示使用读取环境变量的方式进行初始化。该参数与 `store` 互斥。

- **rank** ： **指定当前进程的优先级**

`int` 值。表示当前进程的编号，即优先级。如果指定 `store` 参数，则必须指定该参数。

`rank=0` 的为主进程，即 `master` 节点。

- **world_size** ：

该 `job` 中的总进程数。如果指定 `store` 参数，则需要指定该参数。

- **timeout** ： **指定每个进程的超时时间**

可选参数，`datetime.timedelta` 对象，默认为 `30` 分钟。该参数仅用于 `Gloo` 后端。

- **store**

所有 `worker` 可访问的 `key` / `value`，用于交换连接 / 地址信息。与 `init_method` 互斥。





### new_group

### 函数声明

```python
torch.distributed.new_group(ranks=None, 
                            timeout=datetime.timedelta(0, 1800), 
                            backend=None)
```

### 函数作用

`new_group()` 函数可用于使用所有进程的任意子集来创建新组。其返回一个分组句柄，可作为 `collectives` 相关函数的 `group` 参数 。`collectives` 是分布式函数，用于特定编程模式中的信息交换。

### 参数详解

- **ranks**：**指定新分组内的成员的 `ranks` 列表**

`list` ，其中每个元素为 `int` 型

- **timeout**：**指定该分组进程组内的操作的超时时间**

可选参数，`datetime.timedelta` 对象，默认为 `30` 分钟。该参数仅用于 `Gloo` 后端。

- **backend**：**指定要使用的通信后端**

小写字符串，支持的通信后端有 `gloo`，`nccl` ，必须与 `init_process_group()` 中一致。



## 获取进程组属性

### get_backend

**原型**

```python
torch.distributed.get_backend(group=<object>)
```

**函数说明**

返回给定进程组的 `backend`。

**参数**

- **group**：**要获取信息的进程组**。

进程组对象，默认为主进程组。如果指定另一个进程组，则调用该函数的进程必须为所指定的进程组的进程。

**返回**

- 给定进程组的后端，以小写字符串的形式给出

### get_rank

**函数原型**

```python
torch.distributed.get_rank(group=<object>)
```

**函数说明**

返回当前进程的 `rank`。

`rank` 是赋值给一个分布式进程组组内的每个进程的唯一识别。一般而言，`rank` 均为从 `0` 到 `world_size` 的整数。

**参数**

- **group**

要获取信息的进程组对象，默认为主进程组。如果指定另一个进程组，则调用该函数的进程必须为所指定的进程组的进程。

### get_world_size

**函数原型**

```python
torch.distributed.get_world_size(group=<object>)
```

**函数说明**

返回当前进程组内的进程数。

**参数**

- **group**

要获取信息的进程组对象，默认为主进程组。如果指定另一个进程组，则调用该函数的进程必须为所指定的进程组的进程。

### is_initialized

**函数原型**

```python
torch.distributed.is_initialized()
```

**函数说明**

检查默认进程组是否被初始化。

### is_mpi_available

**函数原型**

```python
torch.distributed.is_mpi_available()
```

**函数作用**

检查 `MPI` 后端是否可用。

### is_nccl_available

**函数原型**

```python
torch.distributed.is_nccl_available()
```

**函数原型**

检查 `NCCL` 后端是否可用。

## 通信后端

## 概述

使用分布式时，在梯度汇总求平均的过程中，各主机之间需要进行通信。因此，需要指定通信的协议架构等。`torch.distributed` 对其进行了封装。

`torch.distributed` 支持 `3` 种后端，分别为 `NCCL`，`Gloo`，`MPI`。各后端对 `CPU / GPU` 的支持如下所示：

![img](https://pic4.zhimg.com/80/v2-54b2efac8658c14f72104c2101a81ecf_720w.jpg)

## 各种后端

### **Gloo 后端**

`gloo` 后端支持 `CPU` 和 `GPU`，其支持集体通信（`collective Communication`），并对其进行了优化。

由于 `GPU` 之间可以直接进行数据交换，而无需经过 `CPU` 和内存，因此，在 `GPU` 上使用 `gloo` 后端速度更快。

`torch.distributed` 对 `gloo` 提供原生支持，无需进行额外操作。

### NCCL 后端

`NCCL` 的全称为 `Nvidia` 聚合通信库（`NVIDIA Collective Communications Library`），是一个可以实现多个 `GPU`、多个结点间聚合通信的库，在 `PCIe、Nvlink、InfiniBand` 上可以实现较高的通信速度。

`NCCL` 高度优化和兼容了 `MPI`，并且可以感知 `GPU` 的拓扑，促进多 `GPU` 多节点的加速，最大化 `GPU` 内的带宽利用率，所以深度学习框架的研究员可以利用 `NCCL` 的这个优势，在多个结点内或者跨界点间可以充分利用所有可利用的 `GPU`。

`NCCL` 对 `CPU` 和 `GPU` 均有较好支持，且 `torch.distributed` 对其也提供了原生支持。

对于每台主机均使用多进程的情况，使用 `NCCL` 可以获得最大化的性能。每个进程内，不许对其使用的 `GPUs` 具有独占权。若进程之间共享 `GPUs` 资源，则可能导致 `deadlocks`。

### MPI 后端

`MPI` 即消息传递接口（`Message Passing Interface`），是一个来自于高性能计算领域的标准的工具。它支持点对点通信以及集体通信，并且是 `torch.distributed` 的 `API` 的灵感来源。使用 `MPI` 后端的优势在于，在大型计算机集群上，`MPI` 应用广泛，且高度优化。

但是，`torch.distributed` 对 `MPI` 并不提供原生支持。因此，要使用 `MPI`，必须从源码编译 `Pytorch`。是否支持 `GPU`，视安装的 `MPI` 版本而定。

### 编译步骤

1. 创建并激活 `Anaconda` 环境，安装 [the guide](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch%23from-source) 指定的依赖包，但是此时还不能运行 `python setup.py install`
2. 选择并安装偏好的 `MPI` 实现。需要注意的是，开启 `CUDA-aware MPI` 可能需要一些额外的步骤。可以使用不提供 `GPU` 支持的 `Open-MPI`：`conda install -c conda-forgeopenmpi`
3. 进入 `Pytorch` 源码，执行 `python setup.py install`

### 使用实例

**源码**

```python
# filename 'ptdist.py'
import torch
import torch.distributed as dist

def main(rank, world):
    if rank == 0:
        x = torch.tensor([1., -1.]) # Tensor of interest
        dist.send(x, dst=1)
        print('Rank-0 has sent the following tensor to Rank-1')
        print(x)
    else:
        z = torch.tensor([0., 0.]) # A holder for recieving the tensor
        dist.recv(z, src=0)
        print('Rank-1 has recieved the following tensor from Rank-0')
        print(z)

if __name__ == '__main__':
    dist.init_process_group(backend='mpi')
    main(dist.get_rank(), dist.get_world_size())
```

**执行**

```text
$ mpiexec -n 2 -ppn 1 -hosts miriad2a,miriad2b python ptdist.py
```

**结果**

```text
Rank-0 has sent the following tensor to Rank-1
 tensor([ 1., -1.])
 Rank-1 has recieved the following tensor from Rank-0
 tensor([ 1., -1.])
```

## 如何选择

> **强烈建议：**
> `NCCL` 是目前最快的后端，且对多进程分布式（`Multi-Process Single-GPU`）支持极好，可用于单节点以及多节点的分布式训练。
> 节点即主机。即使是单节点，由于底层机制不同，`distributed` 也比 `DataParallel` 方式要高效。

**基本原则：**

- 用 `NCCL` 进行分布式 `GPU` 训练
- 用 `Gloo` 进行分布式 `CPU` 训练

**无限带宽互联的 GPU 集群**

- 使用 `NCCL`，因为它是目前唯一支持 `InfiniBand` 和 `GPUDirect` 的后端

**无限带宽和 GPU 直连**

- 使用 `NCCL`，因为其目前提供最佳的分布式 `GPU` 训练性能。尤其是 `multiprocess single-node` 或 `multi-node distributed` 训练。
- 如果用 `NCCL` 训练有问题，再考虑使用 `Cloo`。(当前，`Gloo` 在 `GPU` 分布式上，相较于 `NCCL` 慢)

**无限带宽互联的 CPU 集群**

- 如果 `InfiniBand` 对 `IB` 启用 `IP`，请使用 `Gloo`，否则使使用 `MPI`。
- 在未来将添加 `infiniBand` 对 `Gloo` 的支持

**以太网互联的 CPU 集群**

- 使用 `Gloo`，除非有特别的原因使用 `MPI`。

## 初始化方式

分布式任务中，各节点之间需要进行协作，比如说控制数据同步等。因此，需要进行初始化，指定协作方式，同步规则等。

`torch.distributed` 提供了 `3` 种初始化方式，分别为 `tcp`、`共享文件` 和 `环境变量初始化` 等。

## TCP 初始化

**代码**

`TCP` 方式初始化，需要指定进程 `0` 的 `ip` 和 `port`。这种方式需要手动为每个进程指定进程号。

```python
import torch.distributed as dist

# Use address of one of the machines
dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                        rank=args.rank, world_size=4)
```

**说明**

不同进程内，均使用主进程的 `ip` 地址和 `port`，确保每个进程能够通过一个 `master` 进行协作。该 `ip` 一般为主进程所在的主机的 `ip`，端口号应该未被其他应用占用。

实际使用时，在每个进程内运行代码，并需要为每一个进程手动指定一个 `rank`，进程可以分布与相同或不同主机上。

多个进程之间，同步进行。若其中一个出现问题，其他的也马上停止。

**使用**

```
Node 1
python mnsit.py --init-method tcp://192.168.54.179:22225 --rank 0 --world-size 2
Node 2
python mnsit.py --init-method tcp://192.168.54.179:22225 --rank 1 --world-size 2
```

**底层实现**

在深入探讨初始化算法之前，先从 `C/C++` 层面，大致浏览一下 `init_process_group` 背后发生了什么。

1. 解析并验证参数
2. 后端通过 `name2channel.at()` 函数进行解析，返回一个 `channel` 类，将用于执行数据传输
3. 丢弃 `GIL`，并调用 `THDProcessGroupInit()` 函数，其实例化该 `channel`，并添加 `master` 节点的地址
4. `rank 0` 对应的进程将会执行 `master` 过程，而其他的进程则作为 `workers`
5. `master`
6. 为所有的 `worker` 创建 `sockets`
7. 等待所有的 `worker` 连接
8. 发送给他们所有其他进程的位置
9. 每一个 `worker`
10. 创建连接 `master` 的 `sockets`
11. 发送自己的位置信息
12. 接受其他 `workers` 的信息
13. 打开一个新的 `socket`，并与其他 `wokers` 进行握手信号
14. 初始化结束，所有的进程之间相互连接

## 共享文件系统初始化

该初始化方式，要求共享的文件对于组内所有进程可见！

**代码**

设置方式如下：

```python
import torch.distributed as dist

# rank should always be specified
dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                        world_size=4, rank=args.rank)
```

**说明**

其中，以 `file://` 为前缀，表示文件系统各式初始化。`/mnt/nfs/sharedfile` 表示共享的文件，各个进程在共享文件系统中通过该文件进行同步或异步。因此，所有进程必须对该文件具有读写权限。

每一个进程将会打开这个文件，写入自己的信息，并等待直到其他所有进程完成该操作。在此之后，所有的请求信息将会被所有的进程可访问，为了避免 `race conditions`，文件系统必须支持通过 `fcntl` 锁定（大多数的 `local` 系统和 `NFS` 均支持该特性）。

**说明**：若指定为同一文件，则每次训练开始之前，该文件必须手动删除，但是文件所在路径必须存在！

> **与 `tcp` 初始化方式一样，也需要为每一个进程手动指定 `rank`。**

**使用**

在主机 `01` 上：

```text
python mnsit.py --init-method file://PathToShareFile/MultiNode --rank 0 --world-size 2
```

在主机 `02` 上：

```text
python mnsit.py --init-method file://PathToShareFile/MultiNode --rank 1 --world-size 2
```

这里相比于 `TCP` 的方式麻烦一点的是**运行完一次必须更换共享的文件名，或者删除之前的共享文件**，不然第二次运行会报错。

## 环境变量初始化

默认情况下使用的都是环境变量来进行分布式通信，也就是指定 `init_method="env://"`。通过在所有机器上设置如下四个环境变量，所有的进程将会适当的连接到 `master`，获取其他进程的信息，并最终与它们握手(信号)。

- `MASTER_PORT`: 必须指定，表示 `rank0`上机器的一个空闲端口（必须设置）
- `MASTER_ADDR`: 必须指定，除了 `rank0` 主机，表示主进程 `rank0` 机器的地址（必须设置）
- `WORLD_SIZE`: 可选，总进程数，可以这里指定，在 `init` 函数中也可以指定
- `RANK`: 可选，当前进程的 `rank`，也可以在 `init` 函数中指定

配合 `torch.distribution.launch` 使用。

**使用实例**

```
Node 1: (IP: 192.168.1.1, and has a free port: 1234)
>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
           --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
           and all other arguments of your training script)
Node 2
>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
           --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
           and all other arguments of your training script)
```

## Distributed Modules

## DistributedDataParallel

### 原型

```python
torch.nn.parallel.DistributedDataParallel(module, 
                                          device_ids=None, 
                                          output_device=None, 
                                          dim=0, 
                                          broadcast_buffers=True, 
                                          process_group=None, 
                                          bucket_cap_mb=25, 
                                          find_unused_parameters=False, 
                                          check_reduction=False)
```

### 功能

将给定的 `module` 进行分布式封装， 其将输入在 `batch` 维度上进行划分，并分配到指定的 `devices` 上。

`module` 会被复制到每台机器的每个 `GPU` 上，每一个模型的副本处理输入的一部分。

在反向传播阶段，每个机器的每个 `GPU` 上的梯度进行汇总并求平均。与 `DataParallel` 类似，`batch size` 应该大于 `GPU` 总数。

### 参数解析

- **module**

要进行分布式并行的 `module`，一般为完整的 `model`

- **device_ids**

`int` 列表或 `torch.device` 对象，用于指定要并行的设备。**参考 DataParallel**。

对于数据并行，即完整模型放置于一个 `GPU` 上（`single-device module`）时，需要提供该参数，表示将模型副本拷贝到哪些 `GPU` 上。

对于模型并行的情况，即一个模型，分散于多个 `GPU` 上的情况（`multi-device module`），以及 `CPU` 模型，该参数比必须为 `None`，或者为空列表。

与单机并行一样，输入数据及中间数据，必须放置于对应的，正确的 `GPU` 上。

- **output_device**

`int` 或者 `torch.device`，**参考 DataParallel**。

对于 `single-device` 的模型，表示结果输出的位置。

对于 `multi-device module` 和 `GPU` 模型，该参数必须为 `None` 或空列表。

- **broadcast_buffers**

```
bool` 值，默认为 `True
```

表示在 `forward()` 函数开始时，对模型的 `buffer` 进行同步 (`broadcast`)

- **process_group**

对分布式数据（主要是梯度）进行 `all-reduction` 的进程组。

默认为 `None`，表示使用由 `torch.distributed.init_process_group` 创建的默认进程组 (`process group`)。

- **bucket_cap_mb**

```
DistributedDataParallel will bucket parameters into multiple buckets so that gradient reduction of each bucket can potentially overlap with backward computation.
bucket_cap_mb controls the bucket size in MegaBytes (MB) (default: 25)
```

- **find_unused_parameters**

`bool` 值。

`Traverse the autograd graph of all tensors contained in the return value of the wrapped module’s forward function`.

`Parameters that don’t receive gradients as part of this graph are preemptively marked as being ready to be reduced`.

`Note that all forward outputs that are derived from module parameters must participate in calculating loss and later the gradient computation`.

`If they don’t, this wrapper will hang waiting for autograd to produce gradients for those parameters`.

`Any outputs derived from module parameters that are otherwise unused can be detached from the autograd graph using torch.Tensor.detach`. (default: `False`)

- **check_reduction**

`when setting to True, it enables DistributedDataParallel to automatically check if the previous iteration’s backward reductions were successfully issued at the beginning of every iteration’s forward function`.

```
You normally don’t need this option enabled unless you are observing weird behaviors such as different ranks are getting different gradients, which should not happen if DistributedDataParallel is correctly used. (default: False)
```

### 注意

1. 要使用该 `class`，需要先对 `torch.distributed` 进行初进程组始化，可以通过 `torch.distributed.init_process_group()` 实现。
2. 该 `module` 仅在 `gloo` 和 `nccl` 后端上可用。
3. 根据分布式原理，`Constructor` 和 `differentiation of the output` (或 `a function of the output of this module`) 是一个分布式同步点。在不同的进程执行不同的代码时，需要考虑这一点。
4. 该 `module` 假设，所有的参数在其创建时，在模型中已经注册，之后没有新的参数加入或者参数移除。对于 `buffers` 也是一样。(这也是由分布式原理决定)
5. 该 `module` 假设，所有的参数在每个分布式进程模型中注册的顺序一致。该 `module` 自身将会按照该模型中参数注册的相反顺序执行梯度的 `all-reduction`。换言之，用户应该保证，每个分布式进程模型一样，且参数注册顺序一致。(这也是由分布式原理决定)
6. 如果计划使用该 `module`，并用 `NCCL` 后端或 `Gloo` 后端 (使用 `infiniband`)，需要与多 `workers` 的 `Dataloader` 一同使用，请修改多进程启动算法为 `forkserver` (`python 3 only`) 或 `spawn` 。不幸的是，`Gloo` (使用 `infiniband`) 和 `NCCL2 fork` 并不安全，并且如果不改变配置时，很可能会 `deadlocks`。
7. 在 `module` 上定义的前向传播和反向传播 `hooks` 和其子 `modules` 将不会涉及，除非 `hooks` 在 `forward` 中进行了初始化。
8. 在使用 `DistributedDataParallel` 封装 `model` 后，不应该再修改模型的参数。也就是说，当使用 `DistributedDataParallel` 打包 `model` 时，`DistributedDataParallel` 的 `constructor` 将会在模型上注册额外的归约函数，该函数作用于模型的所有参数。

如果在构建 `DistributedDataParallel` 之后，改变模型的参数，这是不被允许的，并且可能会导致不可预期的后果，因为部分参数的梯度归约函数可能不会被调用。

1. 在进程之间，参数永远不会进行 `broadcast`。该 `module` 对梯度执行一个 `all-reduce` 步骤，并假设在所有进程中，可以被 `optimizer` 以相同的方式进行更改。 在每一次迭代中，`Buffers` (`BatchNorm stats` 等) 是进行 `broadcast` 的，从 `rank 0` 的进程中的 `module` 进行广播，广播到系统的其他副本中。

### 使用

`DistributedDataParallel` 可以通过如下两种方式进行使用：

1. `Single-Process Multi-GPU`

在这种情况下，每个主机上将用单进程，每个进程使用所在主机的所有的 `GPUs`。这种方式下，代码如下所示：

\```python

> torch.distributed.init_process_group(backend="nccl") model = DistributedDataParallel(model) # device_ids will include all GPU devices by default ```

1. `Multi-Process Single-GPU`

**我们强烈建议用该方式来使用 DistributedDataParallel**，使用多进程，每个进程使用一个 `GPU`。这是目前 `Pytorch` 中，无论是单节点还是多节点，进行数据并行训练**最快**的方式。

并且实验证明，在单节点多 `GPU` 上进行训练，该方式比 `torch.nn.DataParallel` 更快。这是因为分布是并行不需要 `broadcast` 参数。

假设每个主机有 `N` 个 `GPUs`，那么需要使用 `N` 个进程，并保证每个进程单独处理一个 `GPU`。因此，需要保证训练代码在单个 `GPU` 上进行操作，可以用如下代码进行实现：

```
python torch.cuda.set_device(i)
```

其中，`i` 应该为 `0` 到 `N - 1` 之间。在每一个进程中，应该通过如下方式构建模型：

\```python

> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...') model = DistributedDataParallel(model, device_ids=[i], output_device=i]) ```

为了在每个主机（`node`）上使用多进程，可以使用 `torch.distributed.launch` 或 `torch.multiprocessing.spawn` 来实现。

## DistributedDataParallelCPU

### 原型

```python
torch.nn.parallel.DistributedDataParallelCPU(module)
```

### **参数**

- **module** – `module to be parallelized`

### **说明**

- 在 `module` 级别上利用 `CPU` 实现分布式数据并行。
- 该 `module` 支持 `mpi` 和 `gloo` 后端。
- 该 `container` 通过在 `batch` 维度上，对输入进行分割，并分配到特定的设备上，实现模型的并行。将该 `module` 复制到每一台机器上，每一个副本处理输入的一部分。在反向传播阶段，每各节点的梯度求平均。
- 该 `module` 应该与 `DistributedSampler` 一起使用。`DistributedSampler` 将会为每个节点加载一个原始数据集的子集，每个子集的 `batchsize` 相同。因此，总 `bs` 的缩放如下所示：



- n = 1, batch size = 8
- n = 2, batch size = 16
- n = 4, batch size = 32
- n = 8, batch size = 64

该 `class` 的创建，需要该 `distributed package` 以 `process group` 模式进行初始化。(`torch.distributed.init_process_group()`)。

### **警告**

- `constructor`， `forward method` 和 `differentiation of the output` (或 `a function of the output of this module`) 是一个分布式同步点。在不同节点可能执行不同代码的情况下，需要考虑这一点。
- 该 `module` 假设，所有的参数在其创建时，在模型中已经注册，之后没有新的参数加入或者参数移除。对于 `buffers` 也是一样。
- 该 `module` 假设所有的 `buffers` 和梯度都是密集型的。
- 在 `module` 上定义的前向传播和反向传播 `hooks` 和其子 `modules` 将不会涉及，除非 `hooks` 在 `forward` 中进行了初始化。
- 参数在 `__init__()` 函数中，在不同节点之间进行 `broadcast`。该 `module` 在梯度上执行一个 `all-reduce` 步骤，并假设它们将会被 `optimizer` 在所有节点上以相同的方式进行更改。

## DistributedSampler

### 原型

```python
torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None)
```

### 参数

- **dataset**

进行采样的数据集

- **num_replicas**

分布式训练中，参与训练的进程数

- **rank**

当前进程的 `rank` 序号（必须位于分布式训练中）

### 说明

对数据集进行采样，使之划分为几个子集，不同 `GPU` 读取的数据应该是不一样的。

一般与 `DistributedDataParallel` 配合使用。此时，每个进程可以传递一个 `DistributedSampler` 实例作为一个 `Dataloader sampler`，并加载原始数据集的一个子集作为该进程的输入。

在 `Dataparallel` 中，数据被直接划分到多个 `GPU` 上，数据传输会极大的影响效率。相比之下，在 `DistributedDataParallel` 使用 `sampler` 可以为每个进程划分一部分数据集，并避免不同进程之间数据重复。

**注意**：在 `DataParallel` 中，`batch size` 设置必须为单卡的 `n` 倍，但是在 `DistributedDataParallel` 内，`batch size` 设置于单卡一样即可。

### 使用实例

```python
# 分布式训练示例
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

dataset = your_dataset()
datasampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size_per_gpu, sampler=datasampler)
model = your_model()
model = DistributedDataPrallel(model, device_ids=[local_rank], output_device=local_rank)
```

## 通信方式

`torch.distributed` 支持 `Collective Communication` 和 `Point-to-Point Communication`，前者为默认通信方式。

## Point-to-Point Communication

### 基本概念

点对点通信，使用同一个 `IP` 和端口。

![img](https://pic1.zhimg.com/80/v2-cf379fd23a7e7ef9fd6d813320de96fc_720w.jpg)

点对点通信指的是，数据从一个进程转移到另一个进程。这通过 `send` 和 `recv` 函数实现，也可以用对应的即时版本，`isend` 和 `irevc` 进行实现。

当想要对进程之间的通信有一个 `fine-grained` 控制的时候，点对点通信是很有用的。

### 阻塞式 blocking

### send

**函数原型**

```python
torch.distributed.send(tensor, dst, group=<object>, tag=0)
```

**函数作用**

同步发送一个 `tensor`。

**参数**

- **tensor** ：要发送的 `tensor`
- **dst**：目标 `rank`，整数
- **group**：工作的进程组
- **tag** ：用于匹配当前 `send` 与远程 `recv` 的标记

### recv

**函数原型**

```python
torch.distributed.recv(tensor, src=None, group=<object>, tag=0)
```

**函数作用**

同步接收一个 `tensor`。

**参数**：

- **tensor**：要接收的 `tensor`
- **src**：源 `rank`，如果未指定，则可以从任意进程接收数据
- **group**：工作的进程组
- **tag**：用于匹配当前 `recv` 与远程 `send` 的标记

**返回值**

```
Sender rank -1, if not part of the group
```

### 非阻塞式 Non-blocking

### isend

**函数原型**

```python
torch.distributed.isend(tensor, dst, group=<object>, tag=0)
```

**函数作用**

异步发送一个 `tensor`。

**参数**

- **tensor** ：要发送的 `tensor`
- **dst**：目标 `rank`，整数
- **group**：工作的进程组
- **tag** ：用于匹配当前 `send` 与远程 `recv` 的标记

**返回**

```
A distributed request object. None, if not part of the group
```

### irecv

**函数原型**

```python
torch.distributed.irecv(tensor, src=None, group=<object>, tag=0)
```

**函数作用**

异步接收一个 `tensor`。

**参数**：

- **tensor**：要接收的 `tensor`
- **src**：源 `rank`，如果未指定，则可以从任意进程接收数据
- **group**：工作的进程组
- **tag**：用于匹配当前 `recv` 与远程 `send` 的标记

**返回值**

```
A distributed request object. None, if not part of the group
```

### 实例

### 阻塞式

在阻塞式通信中，进程间通信必须等待数据传输完成才会进行下一步。

```python
"""Blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
```

在上面的例子中，每一个进程都是以 `0` 开始的，随后进程 `0` 增加了该 `tensor`，然后将其发送给进程 `1`，因此两个进程中均更新为 `1`。

### 非阻塞式

在非阻塞通信中，进程间通信无需等待。该方法返回一个 `DistributedRequest` 对象，该对象支持如下两个方法：

1. `is_completed()`

如果通信完成，则返回 `True`。

1. `wait()`

对进程上锁，等待通信结束。在 `req.wait()` 执行之后，我们可以保证通信已经结束。

```python
"""Non-blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
```

我们需要小心的处理发送和接受的 `tensor`。由于我们不知道什么时候数据将会与其他进程进行通信，因此在 `req.wait()` 结束之前，我们不应该更改发送的 `tensor` 或者访问接受的 `tensor`。

## Collective Communication

### 集体通信的概念

每个 `collective operations` ，也就是群体操作，支持同步和异步的方式。

- 同步操作（默认）

当 `async_op` 设置为 `False` 时，为同步操作。

当函数返回时，可以保证 `collective` 操作执行完毕(由于所有的 `CUDA` 操作都是异步的，因此当为 `CUDA` 操作时，操作不一定已完成)，同时任何进一步的函数调用取决于 `collective` 操作能够调用的数据。在同步模式下，`collective` 函数不返回任何值。

- 异步操作

当 `async_op` 设置为 `True` 时，为此模式。

此时，`collective` 操作函数返回一个分布式请求对象。通常来说，无需手动创建该对象，且该对象支持两个操作：

- `is_completed()` ：判断是否执行完毕，若是则返回 `True`
- `wait()`：使用这个方法来阻塞这个进程，直到调用的 `collective function` 执行完毕

与点对点通信相反，集体通信支持同组内的所有进程之间的通信。

要创建一个组，可以传递一个 `rank` 的列表给 `dist.new_group(group)`。

默认情况下，集体操作是执行在所有的进程上的，也被称之为 `world`（所有的进程）。

### 单 GPU 集体操作

### broadcast

**函数原型**

```python
torch.distributed.broadcast(tensor, src, group=<object>, async_op=False)
```

**函数功能**

广播该 `tensor` 到整个 `group`。

该 `tensor` 在该组内所有对应的 `tensor` 必须尺寸一致。

![img](https://pic2.zhimg.com/80/v2-9b1092c046e04f06732977c0db2e9589_720w.jpg)

**参数**

- **tensor**

若 `src` 为当前进程，则 `tensor` 为要发送的数据；若不为当前进程，则 `tensor` 为要接收的数据。

- **src**

源进程。

- **group**

可选参数，指定该操作所在的组。

- **async_op**

可选参数，决定是同步还是异步

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### scatter

**函数原型**

```text
torch.distributed.scatter(tensor, scatter_list, src, group=<object>, async_op=False)
```

**函数功能**

分发 `tensor` 到组内所有进程，注意与 `broadcast` 的区别。

![img](https://pic1.zhimg.com/80/v2-b5e59f7297c513b80a040af5a6c40880_720w.jpg)

**参数**

- **tensor**

输出 `Tensor`，对应于上图的 `Rank 0` 内的值

- **scatter_list**

要分发的目标 `tensor` 列表。仅在发送数据的进程中需要设定。

- **src**

源 `rank`。除了发送数据的进程，其与所有进程均需要设定该参数。

- **group**

指定该操作所在的组。

- **async_op**

可选参数，决定是同步还是异步

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### barrier

**函数原型**

```python
torch.distributed.barrier(group=<object>, async_op=False)
```

**函数功能**

同步所有进程。

若 `async_op` 为 `False`，或 `async` 进程是在 `wait()` 中调用的，则该操作将封锁进程，直到整个组进入该函数。

**参数**

- **group**

指定该操作所在的组。

- **async_op**

可选参数，决定是同步还是异步

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### gather

**函数原型**

```python
torch.distributed.gather(tensor, gather_list, dst, group=<object>, async_op=False)
```

**函数功能**

将一组 `tensor` 聚集于一个进程。

![img](https://pic3.zhimg.com/80/v2-0abc1e3c45f2e250c51900cc44f745da_720w.jpg)

**参数**

- **tensor**

输入 `tensor`

- **gather_list**

仅在接收数据的进程中需要设定，为一个尺寸合适的 `tensor`。

- **dst**

目标 `rank`。在所有发送数据的进程中，均需要设定该参数

- **group**

指定该操作所在的组。

- **async_op**

可选参数，决定是同步还是异步

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### all_gather

**函数原型**

```python
torch.distributed.all_gather(tensor_list, tensor, group=<object>, async_op=False)
```

**函数功能**

将 `group` 中的 `tensor` 集中到 `tensor_list` 中。

![img](https://pic2.zhimg.com/80/v2-f0f5834ce004382f935b73fcfaf645bd_720w.jpg)

**参数**

- **tensor_list**

合适尺寸的输出 `tensor` 列表。

- **tensor**

当前进程需要 `broadcast` 的 `tensor`。

- **group**

指定该操作所在的组。

- **async_op**

可选参数，决定是同步还是异步

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### reduce

**函数原型**

```python
torch.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=<object>, async_op=False)
```

**函数功能**

对所有进程内的数据进行归约。但是结果只存储于 `dst` 进程。

![img](https://pic2.zhimg.com/80/v2-d914e7609410bfe7591634bc01f9a239_720w.jpg)

**参数**

- **tensor**

`collective` 输入输出，该操作是 `inplace` 的

- **dst**

目标 `rank`

- **op**

指定归约操作的类型，为 `torch.distributed.ReduceOp` 枚举类型。支持：

1. `torch.distributed.ReduceOp.SUM`
2. `torch.distributed.ReduceOp.PRODUCT`
3. `torch.distributed.ReduceOp.MIN`
4. `torch.distributed.ReduceOp.MAX`
5. **group**

指定该操作所在的组

- **async_op**

可选参数，决定是同步还是异步

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### all_reduce

**函数原型**

```python
torch.distributed.torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=<object>, async_op=False)
```

**函数功能**

与 `reduce` 一致，区别在于，所有进程都获取最终结果，`inplace` 操作。

![img](https://pic1.zhimg.com/80/v2-0f1278761ae7de210aff5d4e0899cf50_720w.jpg)

**参数**

- **tensor**

`collective` 输入输出，该操作是 `inplace` 的

- **op**

指定归约操作的类型，为 `torch.distributed.ReduceOp` 枚举类型。支持：

1. `torch.distributed.ReduceOp.SUM`
2. `torch.distributed.ReduceOp.PRODUCT`
3. `torch.distributed.ReduceOp.MIN`
4. `torch.distributed.ReduceOp.MAX`
5. **group**

指定该操作所在的组

- **async_op**

可选参数，决定是同步还是异步

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### 多 GPU 集体操作

### 说明

当使用 `NCCL` 和 `Gloo` 后端时，如果每个节点上拥有多个 `GPU`，支持每个节点内的多 `GPUs` 之间的分布式 `collective` 操作。要注意到每个进程上的 `tensor list` 长度都必须相同。

该操作可能潜在的改善整个分布式训练性能，并易于通过传递一个 `tensor` 列表来使用。函数调用时，在传递的列表中的每个 `tensor`，需要在主机的一个单独的 `GPU` 上。

### 实例

例如，假设用于训练的系统包含 `2` 个节点(`node`)，也就是主机，每个节点有 `8` 个 `GPU`。在这 `16` 个 `GPU` 上的每一个中，有一个需要进行 `all_reduce` 的 `tensor`。那么可以参考如下代码：

**Code running on Node 0**：

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl",
                        init_method="file:///distributed_test",
                        world_size=2,
                        rank=0)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)
```

**Code running on Node 1**：

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl",
                        init_method="file:///distributed_test",
                        world_size=2,
                        rank=1)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)
```

在所有的调用之后，两个节点上，所有的 `GPU` 上的对应 `tensor` 均为 `16`。

### broadcast_multigpu

**函数原型**

```python
torch.distributed.broadcast_multigpu(tensor_list, src, group=<object>, async_op=False, src_tensor=0)
```

**函数功能**

将 `tensors` 在组内进行 `broadcast` 组内每个节点均有多个 `GPU tensor`。

参与到 `collective` 的所有的进程对应的所有 `GPU` 内的指定 `tensor`，必须具有相同的元素数量，且 `tensor_list` 内的每一个 `tensor` 必须处于不同的 `GPU` 上。

当前仅有 `nccl` 和 `gloo` 后端支持，且必须为 `GPU tensors`。

**参数**

- **tensor_list**

参与到该 `collective` 内的 `tensors` 列表。

如果 `src` 为当前进程，则 `src_tensor` 指定的元素（`tensor_list[src_tensor]`）将被 `broadcast` 到 `src` 进程内所有的其他 `tensors`（分别不同 `GPU` ），以及其他非 `src` 进程的 `tensor_list` 内的所有元素。

需要保证，对于所有调用该函数的分布式进程中，`tensor_list` 的长度是一样的。

- **src**

源进程

- **group**

指定该操作所在的组

- **async_op**

可选参数，决定是同步还是异步

- **src_tensor**

可选参数，是定 `tensor_list` 内的源 `tensor` 的索引。

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### all_gather_multigpu

**函数原型**

```python
torch.distributed.all_gather_multigpu(output_tensor_lists, input_tensor_list, group=<object>, async_op=False)
```

**函数功能**

从列表中的整个组收集张量。`tensor_list` 中的每个张量应该位于一个单独的 `GPU` 上目前只支持 `nccl` 后端张量，应该只支持 `GPU` 张量。

**参数**

- **output_tensor_lists**

输出列表。在每张 `GPU` 上，其应该包含合适的尺寸来接收 `collective` 的输出。例如，`output_tensor_lists[i]` 包含 `all_gather` 的结果，其存在于 `input_tensor_list[i]` 所属的 `GPU`。

需要注意，`output_tensor_lists` 内的每一个元素，尺寸均为 `world_size *len(input_tensor_list)`。

`input_tensor_list[j]` 中索引为 `k` 的值，对应于 `output_tensor_lists[i][k *world_size + j]`。

- **input_tensor_list**

当前进程中，需要进行 `broadcast` 的 `tensors` 的列表，每个 `tensors` 应该位于不同的 `GPU` 上。

要注意，所有调用该函数的分布式进程中，`input_tensor_list` 的长度应该一致。

- **group**

指定该操作所在的组。

- **async_op**

可选参数，决定是同步还是异步

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### reduce_multigpu

**函数原型**

```python
torch.distributed.reduce_multigpu(tensor_list, dst, op=ReduceOp.SUM, group=<object>, async_op=False, dst_tensor=0)
```

**函数功能**

对所有机器上的多个 `GPUs` 中的 `tensors` 进行归约。`tensor_list` 内的每个 `tensor` 应该位于独立的 `GPU` 上。

只有 `dst` 进程上的 `tensor_list[dst_tensor]` 对应的 `GPU` 会接收最后的结果。

当前只有 `nccl` 支持该操作，且必须全部为 `GPU tensors`。

**参数**

- **tensor_list**

`collective` 对应的输入输出 `GPU tensor`，该操作为 `inplace`。

需要确保所有调用该函数的分布式进程中的 `tensor_list` 长度一致。

- **dst**

目标进程

- **op**

可选参数，支持：

1. `torch.distributed.ReduceOp.SUM`
2. `torch.distributed.ReduceOp.PRODUCT`
3. `torch.distributed.ReduceOp.MIN`
4. `torch.distributed.ReduceOp.MAX`
5. **group**

指定该操作所在的组

- **async_op**

可选参数，决定是同步还是异步

- **dst_tensor**

可选参数，指定 `tensor_list` 内的 `tensor` 的索引。

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

### all_reduce_multigpu

**函数原型**

```python
torch.distributed.all_reduce_multigpu(tensor_list, op=ReduceOp.SUM, group=<object>, async_op=False)
```

**函数说明**

对所有机器上的多个 `GPUs` 中的 `tensors` 进行归约。`tensor_list` 内的每个 `tensor` 应该位于独立的 `GPU` 上。

所有的进程都将获得最终结果。

当前只有 `nccl` 支持该操作，且必须全部为 `GPU tensors`。

**参数**

- **tensor_list**

`collective` 对应的输入输出 `GPU tensor`，该操作为 `inplace`。

需要确保所有调用该函数的分布式进程中的 `tensor_list` 长度一致。

- **op**
- `torch.distributed.ReduceOp.SUM`
- `torch.distributed.ReduceOp.PRODUCT`
- `torch.distributed.ReduceOp.MIN`
- `torch.distributed.ReduceOp.MAX`
- **group**

指定该操作所在的组

- **async_op**

可选参数，决定是同步还是异步

**返回值**

若 `async_op` 设置为 `True`，则返回 `work` 句柄；否则返回 `None`。

## 启动工具 Launch utility

## 概述

`torch.distributed` 提供了一个启动工具，即 `torch.distributed.launch`，用于在每个单节点上启动多个分布式进程。其同时支持 `Python2` 和 `Python 3`。

`launch` 可用于单节点的分布式训练，支持 `CPU` 和 `GPU`。对于 `GPU` 而言，若每个进程对应一个 `GPU`，则训练将取得最大性能。可通过指定参数（`nproc_per_node`），让 `launch` 在单节点上创建指定数目的进程（不可大于该节点对应的 `GPU` 数目）。

该工具以及多进程分布式训练，目前只有在 `NCCL` 上才能发挥最好的性能，`NCCL` 也是被推荐用于分布式 `GPU` 训练的。

## 参数

- **training_script**

位置参数，单 `GPU` 训练脚本的完整路径，该工具将并行启动该脚本。

- **--nnodes**

指定用来分布式训练脚本的节点数

- **--node_rank**

多节点分布式训练时，指定当前节点的 `rank`。

- **--nproc_per_node**

指定当前节点上，使用 `GPU` 训练的进程数。建议将该参数设置为当前节点的 `GPU` 数量，这样每个进程都能单独控制一个 `GPU`，效率最高。

- **--master_addr**

`master` 节点（`rank` 为 `0`）的地址，应该为 `ip` 地址或者 `node 0` 的 `hostname`。对于单节点多进程训练的情况，该参数可以设置为 `127.0.0.1`。

- **--master_port**

指定分布式训练中，`master` 节点使用的端口号，必须与其他应用的端口号不冲突。

## 使用

### 单节点多进程分布式训练

```text
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of your training script)
```

### 多节点多进程分布式

**Node 1** (`192.168.1.1:1234`)

```text
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of your training script)
```

**Node 2**

```text
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of your training script)
```

### 查看帮助

```python
python -m torch.distributed.launch --help
```