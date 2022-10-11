## 1. CT图像的处理

在[这篇文章](https://iint.icu/mri-ct%e5%b7%a5%e4%bd%9c%e7%9a%84%e6%95%b0%e6%8d%ae%e9%a2%84%e5%a4%84%e7%90%86/)中的第四部分介绍了CT降噪的步骤。但是那样的简单处理还是不能将细小的杂物去除，因为那些东西医师并没有标注。

借鉴[这项工作](https://gitee.com/iint/MRI-to-CT-DCNN-TensorFlow)的处理步骤：还是利用`Simpleitk`，利用MRI图像，生成Mask,然后裁剪CT图像。

1. 先用以前生成Numpy文件的代码稍微改一下，生成png图片，然后直接用他的源码。

```python
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from PIL import Image
# numpy的归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range)

root = r"/home/zhaosheng/paper2/data/geng01_new/geng_data_01"

input_path =  os.path.join(root,"inputs")
target_path = os.path.join(root,"targets")
target_files = sorted(os.listdir(target_path))
input_files = sorted(os.listdir(input_path))

assert len(target_files) == len(input_files)
# 开始写png
i = 1
for patient_id in range(len(target_files)):
    target_filepath = os.path.join(target_path,target_files[patient_id])
    target_data = np.asanyarray(nib.load(target_filepath).dataobj)

    input_filepath = os.path.join(input_path,input_files[patient_id])
    input_data = np.asanyarray(nib.load(input_filepath).dataobj)

    assert input_data.shape== target_data.shape
    _a,_b,slices_num = input_data.shape
    print(target_filepath)
    print(input_filepath)
    os.makedirs("/home/zhaosheng/paper2/data/geng01_new/geng_data_01/png/", exist_ok=True)
    for slice in range(slices_num):

        target_numpy = np.array(target_data[:,:,slice])
        input_numpy = np.array(input_data[:,:,slice])
    
        # 删除空白图片
        if (np.max(target_numpy) == np.min(target_numpy)) or (np.max(input_numpy) == np.min(input_numpy)) :
            continue
        target_numpy_norm = normalization(target_numpy)
        input_numpy_norm = normalization(input_numpy)
 
        if np.any(np.isnan(target_numpy_norm)) or np.any(np.isnan(input_numpy_norm)) or np.mean(target_numpy_norm)<0.01:
            continue

        patient_id_pre = '%03d' % (patient_id+1)
        prefix='%03d' % i

        pic_array = np.hstack([target_numpy_norm*255,input_numpy_norm*255])
        img = Image.fromarray(pic_array)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((512, 256), Image.ANTIALIAS)

        img.save(f"/home/zhaosheng/paper2/data/geng01_new/geng_data_01/png/"+"geng01_"+patient_id_pre+"_"+prefix+".png")
        i = i+1
print(f"Done! Total:{i} pairs")
```



2. 此处暂时省略处理步骤。



3. 将处理后的png重新转换成numpy

   ```python
   root = "/home/zhaosheng/paper2/ipynb/Data/brain01/post/"
   files = os.listdir(root)
   files = [file for file in files if "geng" in file]
   print(len(files))
   
   for i,file in enumerate(files,1):
       img = Image.open(os.path.join(root,file))
       img_array = np.array(img)
       mri_array  = normalization(img_array[:,:256])
       ct_array = normalization(img_array[:,256:512])
       
       assert ct_array.shape== mri_array.shape
       
       prefix='%03d' % i
       
       file = file.split(".")[0]
       target_file_name = f"/home/zhaosheng/paper2/data/geng01_new/geng_data_01/numpy/targets/"+file+".npy"
       input_file_name = f"/home/zhaosheng/paper2/data/geng01_new/geng_data_01/numpy/inputs/"+file+".npy"
   
       img_input = Image.open(os.path.join("/home/zhaosheng/paper2/data/geng01_new/geng_data_01/png/",file)+".png")
       img_input_array = np.array(img_input)
       print(img_input_array.shape)
       
       img_input_array=normalization(img_input_array[:,256:,0])
       assert img_input_array.shape== mri_array.shape
   
       np.save(target_file_name, ct_array)
       np.save(input_file_name, img_input_array)
       print(prefix)
   ```



4. 和以前的数据集放一起，完成数据集制作。



## 2. 给网络添加K折交叉验证



## 3. 将CT图像的RT文件的每个mask，也向着MRI对齐配准，便于后续画DVH图像

```python
import os
import argparse
import SimpleITK as sitk

def coregister_mask(fixed_file_path, moving_mask_path, save_path):
    rmethod = sitk.ImageRegistrationMethod()
    rmethod.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    rmethod.SetMetricSamplingStrategy(rmethod.RANDOM)
    rmethod.SetMetricSamplingPercentage(.01)
    rmethod.SetInterpolator(sitk.sitkLinear)
    rmethod.SetOptimizerAsGradientDescent(learningRate=1.0,
                                        numberOfIterations=200,
                                        convergenceMinimumValue=1e-6,
                                        convergenceWindowSize=10)
    rmethod.SetOptimizerScalesFromPhysicalShift()
    rmethod.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    rmethod.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    rmethod.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    for file_name in os.listdir(moving_mask_path):
        try:
            print(f"Now moving:{file_name}")
            moving_path = os.path.join(moving_mask_path, file_name)

            fixed_image = sitk.ReadImage(fixed_file_path, sitk.sitkFloat32)
            moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)

            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, moving_image, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
            rmethod.SetInitialTransform(initial_transform, inPlace=False)
            final_transform = rmethod.Execute(fixed_image, moving_image)
            #print('-> coregistered')

            moving_image = sitk.Resample(
                moving_image, fixed_image, final_transform, sitk.sitkLinear, .0,
                moving_image.GetPixelID())
            moving_image = sitk.Cast(moving_image, sitk.sitkInt16)
            #print('-> resampled')

            os.makedirs(save_path, exist_ok=True)
            sitk.WriteImage(moving_image, os.path.join(save_path,file_name))
            #print('-> exported')
        except:
            print(f"Erro: {file_name}")
        
for i in range(1,19):
    patient_id = '%03d' % i
    
    
    files = sorted(os.listdir("/home/zhaosheng/paper2/data/geng01_new/geng_data_01/inputs/"))
    print(files)
    print("Now "+patient_id+"  "+ files[i-1])
    print("*"*30)
    coregister_mask(
        "/home/zhaosheng/paper2/data/geng01_new/geng_data_01/inputs/"+files[i-1],
        "/home/zhaosheng/paper2/test_output/real_data/"+patient_id+"/nii/",
        "/home/zhaosheng/paper2/test_output/real_data/"+patient_id+"/reg_nii/",
    )
```



## 未完

