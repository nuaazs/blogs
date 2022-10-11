## 主要贡献

- 在COCO数据集上的第一个实时实例分割模型
- 提出了比NMS算法更快的Fast NMS

## 网络结构

YOLACT 的框架图如下：

![image-20200403141950957](https://www.zdaiot.com/DeepLearningApplications/%E5%AE%9E%E4%BE%8B%E5%88%86%E5%89%B2/YOLACT%E8%AF%A6%E8%A7%A3/image-20200403141950957.png)

### 预处理

对上面图中左下方绿色的原图进行预处理，步骤包括：**将图片调整到合适的大小、转换 bbox 坐标和转换 mask 矩阵等**。

### Backbone

预处理后的图片输入进 backbone 网络，论文中使用了ResNet101，源代码中作者还实现了ResNet50 和 DarkNet53。为了下面描述方便，将ResNet 的网络结构放到下面：

![img](https://www.zdaiot.com/DeepLearningApplications/%E5%AE%9E%E4%BE%8B%E5%88%86%E5%89%B2/YOLACT%E8%AF%A6%E8%A7%A3/v2-94d5266cf80bb8553610a1fd14b84302_720w.jpg)

可以看到，它的卷积模块有五个，分别是conv1,conv2x,…,conv5x，**这五个模块的输出分别对应上图的 C1 到 C5。**与SSD相似，YOLACT也采用了多个尺度的特征图。这样做的好处是小的特征图卷积核感受视野比较大，且每一个单元格对应在原图中的尺寸比较大，所以可以设定**比较大的特征图来用来检测相对较小的目标，而小的特征图负责检测大目标**，如下图所示，8*8 的特征图相当于将原图分为 64 份，这样每个 anchor 比较小，而 4*4 的则相反：

![img](https://www.zdaiot.com/DeepLearningApplications/%E5%AE%9E%E4%BE%8B%E5%88%86%E5%89%B2/YOLACT%E8%AF%A6%E8%A7%A3/v2-669005e58c443af6d1254bd08a192641_720w.jpg)

### FPN

FPN全称为Feature pyramid Network。在上面的YOLACT框架图中，橙色的 P3-P7 是FPN网络。P5 是由 C5 经过一个卷积层得到的；接着对 P5 进行一次双线性插值将其放大，与经过卷积的 C4相加得到 P4；同样的方法得到 P3。此外，还对 P5 进行了卷积得到 P6，对 P6 进行卷积得到 P7。

论文中解释了使用 FPN 的原因：**注意到更深层的特征图能生成更鲁棒的 mask，而更大的 prototype mask 能确保最终的 mask 质量更高且更好地检测到小物体。又想特征深，又想特征图大（本文中的P3），那就得使用 FPN 了。**

接下来是并行的操作。P3 被送入 Protonet，P3-P7 也被同时送到 Prediction Head 中。

### Protonet

受Mask R-CNN结构的启发，本文设计了Protonet结构，当输入图像尺寸是550×550时，结构如下图所示。它的输入是P3（因为FPN结构，它的特征深，且特征图大），由多个3×3卷积层、上采样层和一个1×1的卷积层组成，输出的mask维度为138×138×32，即32 个 prototype mask，每个大小是 138×138。

![image-20200403144449496](https://www.zdaiot.com/DeepLearningApplications/%E5%AE%9E%E4%BE%8B%E5%88%86%E5%89%B2/YOLACT%E8%AF%A6%E8%A7%A3/image-20200403144449496.png)

该结构与FCN的区别在于，没有针对prototypes的特定损失。相反的，这些prototypes的所有监督信息均来自集成后的mask损失。

最后，论文中提到，Protonet的输出无界特别重要，因为这可以使网络为它非常有信心的原型（例如明显的背景）产生大的、压倒性的激活。因此，这里使用了Relu激活函数。

### Prediction Head

该分支的结构下面图中右半部分所示，这个分支的输入是 P3-P7 共五个特征图，**Prediction Head 也有五个共享参数的预测层与之一一对应。**和RetinaNet相比，这里的Head结构更浅，并添加了一个mask coefficient分支。图中c表示类别数；a表示每个特征图anchors的种类；k表示prototype的个数，对于不同的特征图，这里均等于32。下图中右半部分，首先是三个分支共享3×3的卷积层，然后是每一个分支有各自的3×3的卷积层。

![image-20200403144911479](https://www.zdaiot.com/DeepLearningApplications/%E5%AE%9E%E4%BE%8B%E5%88%86%E5%89%B2/YOLACT%E8%AF%A6%E8%A7%A3/image-20200403144911479.png)

输入的特征图先生成 anchor。每个像素点生成 3 个 anchor（a=3），比例是 1:1、1:2 和 2:1。五个特征图的 anchor 基本边长分别是 24、48、96、192 和 384。基本边长根据不同比例进行调整，确保 anchor 的面积相等（？）。

以P3为例，对该图进行解释，经过Head网络中前几个卷积层后，它的维度为 W3×H3×256，那么它的anchor数就是W3×H3×3。接下来 Prediction Head 为其生成三类输出：

- 全部类别置信度，COCO 中共有 81 类（包括背景，c=81），所以其维度为W3×H3×3×81；
- 全部位置偏移，维度为W3×H3×3×4；
- 全部 mask coefficient，维度为W3×H3×3×32

也就是，经典的anchor-based的目标检测器为每一个anchor产生4+c个系数，而本文产生4+c+k个系数。原文中还提到下面这段话，我目前理解还不是很深，所以先放在这里：然后对于非线性，我们发现能够从最终mask中减去prototype很重要。 因此，我们将 tanh （取值范围为-1到1）应用于k个mask coefficient，从而在没有非线性的情况下产生更稳定的输出。 这种设计选择的相关性在上面整体框架图中显而易见，因为在不允许减去的情况下，两种mask都无法构建。

### Mask Assembly

为了产生instance mask，我们将mask coefficient作为系数，将prototype分支的结果进行**线性组合**。将线性组合后结果经过sigmoid函数产生最后的masks。这些操作能够通过矩阵相乘和sigmoid函数得到：

M=σ(PCT)

其中，P是prototype masks，维度为h×w×k；c是mask coefficients，维度为n×k，n为经过NMS和得分阈值后的instance数量。当然，也可以使用更复杂的步骤，这里为了简单和快，只使用了线性组合。最终得到的M尺寸为h×w×n，也就是**预测出的n个mask。**

### 损失函数

这里使用三种损失训练模型：

1. 类别置信度损失 Lcls，计算方式SSD中相同，即softmax损失
2. box regression loss Lbox，计算方式SSD中相同，即smooth-L1损失
3. mask loss Lmask，计算集成后的masks M 和ground truth masks Mgt之间的二分类交叉熵损失：Lmask=BCE(M,Mgt)

这三种损失的权重系数为1,1.5,6.125。并且训练过程中，使用OHEM算法保证正负样本比例为3:1。

值得注意的是，其中 mask loss 在计算时，因为 mask 的大小是138×138，需要先将原图的 mask 数据通过双线性插值缩小到这一尺寸。

### Cropping Masks

在评估期间，我们使用**预测的边界框**裁剪最终的掩码。在训练过程中，我们改为使用真实边界框裁剪，并将Lmask除以真实边界框面积，以保留原型中的小对象。

### Fast NMS

在得到全部的位置偏移后，可以调整 anchor 的位置得到ROI位置。

未完待续