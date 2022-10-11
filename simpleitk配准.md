## 背景

4dct生成工作（少时相ct ➡ 多时相ct），直接生成CT图像效果可能不好，需要结合配准工具，首先预处理数据集，获得所有病人的4dct数据的形变场。然后通过0时相CT直接预测9个形变场。

形变场可视化方法可以参考这篇[博客](https://blog.csdn.net/zuzhiang/article/details/107423465)，本文主要将如何通过simpleitk获取形变场。



## 什么是形变场

首先来介绍一下形变场，一个大小为[W,H]的二维图像对应的形变场的大小是[W,H,2]，其中第三个维度的大小为2，分别表示在x轴和y轴方向的位移。同理，一个大小为[D,W,H]的三维图像对应的形变场的大小是[D,W,H,3]，其中第三个维度的大小为3，分别表示在x轴、y轴和z轴方向的位移。下图是一个二维脑部图像配准后得到的形变场。

![](https://img-blog.csdnimg.cn/20200718101341681.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p1emhpYW5n,size_16,color_FFFFFF,t_70)



## 利用Demon Registration获取形变场

```python
#!/usr/bin/env python

import SimpleITK as sitk
import sys
import os

def command_iteration(filter):
    print(f"{filter.GetElapsedIterations():3}={filter.GetMetric():10.5f}")
   
if len(sys.argv) < 4:
    print(
        f"Usage: {sys.argv[0]} <fixedImageFilter> <movingImageFile> <outputTransformFile>")
    sys.exit(1)
fixed = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)
moving = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)
matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(1024)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
moving = matcher.Execute(moving, fixed)

# The basic Demons Registration Filter
# Note there is a whole family of Demons Registration algorithms included in
# SimpleITK
demons = sitk.DemonsRegistrationFilter()
demons.SetNumberOfIterations(200)
# Standard deviation for Gaussian smoothing of displacement field
demons.SetStandardDeviations(1.0)
demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
displacementField = demons.Execute(fixed, moving)

print("======")
print(f"Number Of Iterations:{demons.GetElapsedIterations()}")
print(f"RMS:{demons.GetRMSChange}")
outTx = sitk.DisplacementFieldTransform(displacementField)
sitk.WriteTransform(outTx, sys.argv[3])

if ("SITK_NOSHOW" not in os.environ):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    
    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed),sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    # Use the // floor division operator so that the pixel type is
    # the same for all three images which is the expectation for
    # the compose filter.
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    sitk.Show(cimg, "DeformableRegistration1 Composition")
```



## 借助Slicer可视化

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/VZaEn.jpg)
