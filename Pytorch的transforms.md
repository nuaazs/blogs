变换是常见的图像变换。它们可以使用链接在一起`Compose`。此外，还有`torchvision.transforms.functional`模块。功能转换可以对转换进行细粒度控制。如果您必须构建更复杂的转换管道（例如，在分段任务的情况下），这将非常有用。



```python
torchvision.transforms.Compose(transforms)
```

```python
transforms.Compose([
    transforms.CenterCrop(10),
    transforms.ToTensor(),    
])
```



## 〇、常见的图像变换概述

1. **裁剪（Crop）**
   1. 中心裁剪：`transforms.CenterCrop`
   2. 随机裁剪：`transforms.RandomCrop`
   3. 随机长宽比裁剪：`transforms.RandomResizedCrop` 
   4. 上下左右中心裁剪：`transforms.FiveCrop`
   5. 上下左右中心裁剪后翻转： `transforms.TenCrop`

2. **翻转和旋转（Flip and Rotation）**
   1. 依概率p水平翻转：`transforms.RandomHorizontalFlip(p=0.5)`
   2. 依概率p垂直翻转：`transforms.RandomVerticalFlip(p=0.5)`
   3.  随机旋转：`transforms.RandomRotation`

3. **图像变换（resize）** 
   1. 大小变化：`transforms.Resize`
   2. 标准化：`transforms.Normalize`
   3. 转为tensor，并归一化至[0-1]：`transforms.ToTensor`
   4. 填充：`transforms.Pad`
   5. 修改亮度、对比度和饱和度：`transforms.ColorJitter`
   6. 转灰度图：`transforms.Grayscale`
   7. 线性变换：`transforms.LinearTransformation()`
   8. 仿射变换：`transforms.RandomAffine`
   9. 依概率p转为灰度图：`transforms.RandomGrayscale`
   10. 将数据转换为PILImage：`transforms.ToPILImage`
   11. transforms.Lambda：Apply a user-defined lambda as a transform.

4. **对transforms操作**
   1. `transforms.RandomChoice(transforms)`从给定的一系列transforms中选一个进行操作 
   2. `transforms.RandomApply(transforms, p=0.5)`给一个transform加上概率，依概率进行操作
   3. `transforms.RandomOrder`将transforms中的操作随机打乱





## 一、裁剪Crop

### 1.1 随机裁剪：`transforms.RandomCrop`

```python
torchvision.transforms.RandomCrop(size,padding=None,pad_if_needed=False,fill=0,padding_mode='constant')
```

- `size`（sequence 或int）：所需输出大小
- `padding`（int或sequence ，optional）： 图像每个边框上的可选填充。默认值为None，即无填充。如果提供长度为4的序列，则它用于分别填充左，上，右，下边界。如果提供长度为2的序列，则分别用于填充左/右，上/下边界。
- `pad_if_needed`（boolean） ：如果小于所需大小，它将填充图像以避免引发异常。由于在填充之后完成裁剪，因此填充似乎是在随机偏移处完成的。
- `fill`：恒定填充的像素填充值。默认值为0。如果长度为3的元组，则分别用于填充R,G,B通道。仅当`padding_mode`为常量时才使用此值。
- `padding_mode`：填充类型。应该是：常量，边缘，反射或对称。默认值是常量。
  - 常量：具有常量值的焊盘，该值用填充指定
  - edge：填充图像边缘的最后一个值
  - 反射：具有图像反射的垫（不重复边缘上的最后一个值）,填充[1,2,3,4]在反射模式下两侧有2个元素将导致[3,2,1,2,3,4,3,2]
  - 对称：具有图像反射的垫（重复边缘上的最后一个值）,填充[1,2,3,4]在对称模式下两侧有2个元素将导致[2,1,1,2,3,4,4,3]
    

### 1.2 中心裁剪：`transforms.CenterCrop`

```python
torchvision.transforms.CenterCrop(size)
```

- 依据给定的`size`从中心裁剪 参数： size (sequence or int)，若为sequence,则为`(h,w)`，若为int，则`(size,size)`



### 1.3 随机长宽比裁剪：`transforms.RandomResizedCrop`

```python
torchvision.transforms.RandomResizedCrop(size, scale=(0.08,1.0), ratio=(0.75,1.333333), interpolation=2)
```

将给定的PIL图像裁剪为**随机大小和宽高比**。将原始图像大小变成**随机大小**（默认值：是原始图像的0.08到1.0倍）和**随机宽高比**（默认值：3/4到4/3倍）。这种方法最终调整到适当的大小。这通常用于训练**Inception网络**。

- size：每条边的预期输出大小
- scale：裁剪的原始尺寸的大小范围
- ratio：裁剪的原始宽高比的宽高比范围
- interpolation：默认值：PIL.Image.BILINEAR



### 1.4 上下左右中心裁剪：`transforms.FiveCrop`

```python
torchvision.transforms.FiveCrop(size)
```

将给定的PIL图像裁剪为四个角和中央裁剪。此转换**返回图像元组**，并且数据集返回的输入和目标数量可能不匹配。

- 对图片进行上下左右以及中心裁剪，获得**5张图片**，返回一个4D-tensor 参数： `size`(sequence or int)，若为sequence,则为(h,w)，若为int，则(size,size)



### 1.5 上下左右中心裁剪后翻转：`transforms.TenCrop`

```python
torchvision.transforms.TenCrop(size, vertical_flip=False)
```

将给定的PIL图像裁剪为四个角，中央裁剪加上这些的翻转版本（默认使用水平翻转）。此转换返回图像元组，并且数据集返回的输入和目标数量可能不匹配。

- `size`（sequence 或int） -作物的所需输出大小。如果size是int而不是像（h，w）这样的序列，则进行正方形裁剪（大小，大小）。
- `vertical_flip`（bool） - 使用垂直翻转而不是水平翻转



## 二、翻转和旋转 Flip Rotation

### 2.1 依概率p水平翻转：`transforms.RandomHorizontalFlip`

```python
torchvision.transforms.RandomHoriziontalFlip(p=0.5)
```



### 2.2 依概率p垂直翻转：`transforms.RandomVerticalFlip`

```python
torchvision.transforms.RandomVerticalFlip(p=0.5)
```



### 2.3 随机旋转：`transforms.RandomRotation`

```python
torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)
```

- `degrees`（sequence 或float或int）：要选择的度数范围。如果degrees是一个数字而不是像`(min，max)`这样的序列，则度数范围将是`(-degrees, +degrees)`。
- `resample`（{PIL.Image.NEAREST ，PIL.Image.BILINEAR ，PIL.Image.BICUBIC} ，可选）： 可选的**重采样过滤器**。如果省略，或者图像具有模式“1”或“P”，则将其设置为`PIL.Image.NEAREST`。
- `expand`（bool，optional）：可选的扩展标志。如果为`True`，则展开输出以使其足够大以容纳整个旋转图像。如果为`False`或省略，则使输出图像与输入图像的大小相同。请注意，展开标志假定围绕中心旋转而不进行平移。
- `center`（2-tuple ，optional）： 可选的旋转中心。原点是左上角。默认值是图像的中心。




## 三、图像变换

### 3.1 resize：`transforms.Resize`

```python
torchvision.transforms.Resize(size, interpolation=2)
```

将输入PIL图像的大小调整为给定大小。

- `size`（sequence 或int）：所需的输出大小。如果size是类似（h，w）的序列，则输出大小将与此匹配。如果size是int，则图像的较小边缘将与此数字匹配。即，如果高度>宽度，则图像将重新缩放为（尺寸*高度/宽度，尺寸）
- `interpolation`（int，optional）：所需的插值。默认是 `PIL.Image.BILINEAR`



## 3.2 标准化：`transforms.Normalize`

```python
torchvision.transforms.Normalize(mean, std)
```

用**平均值**和**标准偏差**归一化张量图像。给定`mean`：(M1,…,Mn)和`std`：(S1,…,Sn)对于n通道，此变换将标准化输入的每个通道，torch.*Tensor即 `input[channel] = (input[channel] - mean[channel]) / std[channel]`

- `mean`（sequence）：每个通道的均值序列。
- `std`（sequence）：每个通道的标准偏差序列。



### 3.3 转为tensor：`transforms.ToTensor`

```python
torchvision.transforms.ToTensor
```

功能：将PIL Image或者 ndarray 转换为tensor，**并且归一化至`[0-1]`** 

注意事项：归一化至`[0-1]`是**直接除以255**，若自己的ndarray数据尺度有变化，则需要自行修改。



### 3.4 填充：`transforms.Pad`

```python
torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
```

使用给定的`pad`值在所有面上填充给定的PIL图像

- `padding`（int或tuple）：每个边框上的填充。如果提供单个int，则用于填充所有边框。如果提供长度为2的元组，则分别为左/右和上/下的填充。如果提供长度为4的元组，则分别为左，上，右和下边框的填充。
- `fill`（int或tuple）：常量填充的像素填充值。默认值为0。如果长度为3的元组，则分别用于填充R,G,B通道。仅`padding_mode`为常量时才使用此值
  。
- padding_mode（str）：填充类型。应该是：常量，边缘，反射或对称。默认值是常量。



### 3.5 修改亮度、对比度和饱和度：`transforms.ColorJitter`

```python
torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
```

随机更改图像的亮度，对比度和饱和度。



### 3.6 转灰度图：`transforms.Grayscale`

```python
torchvision.transforms.Grayscale(num_output_channels=1)
```

将图像转换为灰度。功能：将图片转换为灰度图 

`num_output_channels`(int)：当为1时，正常的灰度图。



### 3.7 线性变换：`transforms.LinearTransformation()`

```python
torchvision.transforms.LinearTransformation(transformation_matrix)
```



#### 3.8 仿射变换：`transforms.RandomAffine`

```python
torchvision.transforms.RandomAffine(degrees,translate=None,scale=None,shear=None,resample=False,fillcolor=0)
```

图像保持中心不变的随机仿射变换。



### 3.9 依概率p转为灰度图：`transforms.RandomGrayscale`

```python
torchvision.transforms.RandomGrayscale(p=0.1)
```



### 3.10 将数据转换为PILImage：`transforms.ToPILImage`

```python
torchvision.transforms.ToPILImage(mode=None)
```

功能：将tensor 或者 ndarray的数据转换为 PIL Image 类型数据 

`mode` 为None时，为1通道， mode=3通道默认转换为RGB，4通道默认转换为RGBA。



### 3.11 transforms.Lambda

```python
torchvision.transforms.Lambda(lambd)
```

将用户定义的lambda应用为变换。

- `lambd`（函数）：用于转换的Lambda /函数。



## 四、对transforms操作，使数据增强更灵活

### 4.1 `transforms.RandomChoice(transforms)`

```python
torchvision.transforms.RandomChoice(transforms)
```

从给定的一系列transforms中选一个进行操作，randomly picked from a list



### 4.2 `transforms.RandomApply(transforms,p=0.5)`

```
torchvision.transforms.RandomApply(transforms,p=0.5)
```

给一个transform加上概率，以一定的概率执行该操作

- `transforms`（列表或元组）：转换列表
- `p`（浮点数）：概率



#### 4.3 `transforms.RandomOrder`

```python
torchvision.transforms.RandomOrder(transforms)
```

将transforms中的操作顺序随机打乱。