MRI-CT工作的数据预处理过程全记录。

## 1. 数据分类

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/pC7ta.jpg)

压缩包解压后，有19个文件夹，每个文件夹名称都是6位数字，文件夹中包含了CT,MR,RT文件的Dicom序列文件。

### 1.1 思路及实现

首先利用`re`正则表达式包，提取不同病人文件夹，识别里面的CT/MR文件，保存在新的文件夹中。

```python
import os
from glob import glob
import shutil
import os
import re
import pydicom


ROOT_PATH = r"F:/MRI-CT/ct-mri-raw/CT&mRI/"
SAVE_PATH = r"F:/MRI-CT-new/ct-mri/CT&mRI/"
# 获取子文件夹
_, dirs, _ = next(os.walk(ROOT_PATH))

# 利用正则表达式筛选6为数字的文件夹名
dirs = [dir_name for dir_name in dirs if re.findall("\d{6}", dir_name)!=[]]
patient_num = len(dirs)
print(f"病例数量{patient_num}人：{dirs}")

# 遍历不同病人的文件夹
for patient_id,dir_name in enumerate(dirs,1):
    patient_id = '%03d' % patient_id
    fold = os.path.join(ROOT_PATH,dir_name)
    files = glob(fold+r"\*.*")
    new_ct_fold = os.path.join(SAVE_PATH,f"{patient_id}/CT")
    new_mri_fold = os.path.join(SAVE_PATH,f"{patient_id}/MRI")
    new_rt_fold = os.path.join(SAVE_PATH,f"{patient_id}/RT")
    os.makedirs(new_ct_fold, exist_ok=True)
    os.makedirs(new_mri_fold, exist_ok=True)
    os.makedirs(new_rt_fold, exist_ok=True)
    
    for file_path in files:
        # 读取文件名，不要后缀
        file_name = file_path.split("\\")[-1]

        # 将 CT 和 MRI 分开
        if ("CT" in file_name):
            shutil.move(file_path, os.path.join(new_ct_fold,file_name)) # 移动文件
        if ("MR" in file_name): 
            # 读取MRI,将不同的加权MRI分开保存
            dcm = pydicom.read_file(file_path)
            seriesUid,seriesName = dcm.SeriesInstanceUID,dcm.SeriesDescription
            save_path = os.path.join(new_mri_fold,seriesName)
            os.makedirs(save_path, exist_ok=True)
            shutil.move(file_path, os.path.join(save_path,file_name))
        if ("RS" in file_name):
            shutil.move(file_path, os.path.join(new_rt_fold,file_name))
print("*"*20)
print("Done")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/QTULi.jpg)

### 1.1 附加

要把MRI中不同的加权方式分开保存。

```python
try:
                # 读取MRI,将不同的加权MRI分开保存
                dcm = pydicom.read_file(file_path)
                seriesUid,seriesName = dcm.SeriesInstanceUID,dcm.SeriesDescription
                save_path = os.path.join(new_mri_fold,seriesName)
                os.makedirs(save_path, exist_ok=True)
                shutil.move(file_path, os.path.join(save_path,file_name))
            except:
                pass
            
```



### 1.2 Tips

1.2.1 Python获取某个目录下的子文件夹和文件：

```python
import os
def get_file():
    # 定义路径
    path='./'
    # 设置空列表
    file_list=[]
    # 使用os.walk获取文件路径及文件
    for home, dirs, files in os.walk(path):
        # 遍历文件名
        for filename in files:
            # 将文件路径包含文件一同加入列表
            file_list.append(os.path.join(home,filename))
    # 赋值
    return file_list

if  __name__ == '__main__':
    file_list = get_file()
```

1.2.2 Python快速新建文件夹：

```python
os.makedirs("images/a", exist_ok=True)
```

1.2.3 Python快速移动文件

```python
shutil.move(file_path, os.path.join(new_ct_fold,file_name)) # 移动文件
```





## 2. 生成nii文件

```python
from time import sleep
from tqdm import tqdm
import os
from glob import glob
import shutil
import os
import re

# 注意此时的ROOT_PATH 就是1.1中的SAVE_PATH
ROOT_PATH = r"F:/MRI-CT-new/ct-mri/CT&mRI/"

_, dirs, _ = next(os.walk(ROOT_PATH))
dirs = [dir_name for dir_name in dirs if re.findall("^\d{3}$", dir_name)!=[]]
print('sub_dirs:', dirs)

for patient_id,dir_name in enumerate(dirs,1):

    fold = os.path.join(ROOT_PATH,dir_name)

    # CT
    ct_path = os.path.join(fold,"CT")
    reader = sitk.ImageSeriesReader()
    dicoms = reader.GetGDCMSeriesFileNames(ct_path)
    reader.SetFileNames(dicoms)
    
    ct_img = reader.Execute()
    ct_size = ct_img.GetSize()
    ct_img = sitk.Cast(ct_img, sitk.sitkFloat32)
    sitk.WriteImage(ct_img , fold+"_ct.nii")
    
    # MRI
    _,dirs, _ = next(os.walk(os.path.join(fold,"MRI")))
    mri_dirs = [dir_name for dir_name in dirs]
    print(f"mri_dirs:{mri_dirs}")
    
    for mri_dir in mri_dirs:
        mri_path = fold+"\\MRI\\"+mri_dir
        mri_reader = sitk.ImageSeriesReader()
        mri_dicoms = mri_reader.GetGDCMSeriesFileNames(mri_path)
        mri_reader.SetFileNames(mri_dicoms)
        mri_img = mri_reader.Execute()
        mri_size = mri_img.GetSize()

        mri_img = sitk.Cast(mri_img, sitk.sitkFloat32)
        sitk.WriteImage(mri_img, fold+"_"+mri_dir+".nii")
```

对于生成的nii文件可以批量导入3DSlicer查看，手动筛选。最后保存至文件夹`geng_data_01`。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/NX9YO.jpg)



## 3. 配准

将MRI固定，移动CT已完成配准

```python
import os
import argparse
import SimpleITK as sitk


def _parse(rootdir):
  filenames = [f for f in os.listdir(rootdir) if f.endswith('.nii')]
  filenames.sort()
  filetree = {}

  for filename in filenames:
    subject, modality = filename.split('.').pop(0).split('_')[:2]

    if subject not in filetree:
      filetree[subject] = {}
    filetree[subject][modality] = filename

  return filetree


def coregister(rootdir, fixed_modality, moving_modality):
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

  filetree = _parse(rootdir)

  for subject, modalities in filetree.items():
    print(f'{subject}:')

    if fixed_modality not in modalities or moving_modality not in modalities:
      print('-> incomplete')
      continue

    fixed_path = os.path.join(rootdir, modalities[fixed_modality])
    moving_path = os.path.join(rootdir, modalities[moving_modality])

    fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    rmethod.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = rmethod.Execute(fixed_image, moving_image)
    print('-> coregistered')

    moving_image = sitk.Resample(
        moving_image, fixed_image, final_transform, sitk.sitkLinear, .0,
        moving_image.GetPixelID())
    moving_image = sitk.Cast(moving_image, sitk.sitkInt16)
    print('-> resampled')

    sitk.WriteImage(moving_image, moving_path)
    print('-> exported')
coregister(r"F:\MRI-CT-new\ct-mri\CT&mRI\geng_data_01", "t1", "ct")
```

完成后Plot出来检查一下，去除一些不好的数据。

```python
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# numpy的归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range)

root = r"F:\MRI-CT-new\ct-mri\CT&mRI\geng_data_01"


input_path =  os.path.join(root,"inputs")
target_path = os.path.join(root,"targets")

target_files = sorted(os.listdir(target_path))
input_files = sorted(os.listdir(input_path))

assert len(target_files) == len(input_files)


# 开始写入numpy
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
    for slice in range(slices_num):
        if slice%5 != 0:
            continue
        target_numpy = np.array(target_data[:,:,slice])
        input_numpy = np.array(input_data[:,:,slice])
    
        # 删除空白图片
        if (np.max(target_numpy) == np.min(target_numpy)) or (np.max(input_numpy) == np.min(input_numpy)) :
            continue

        target_numpy_norm = normalization(target_numpy)
        input_numpy_norm = normalization(input_numpy)
 
        if np.any(np.isnan(target_numpy_norm)) or np.any(np.isnan(input_numpy_norm)) or np.mean(target_numpy_norm)<0.01:
            continue
        plt.subplot(1,2,1)
        plt.imshow(input_numpy_norm)
        
        plt.subplot(1,2,2)
        plt.imshow(target_numpy_norm)
        plt.show()
        i = i+1
print(f"Done! Total:{i} pairs")
```

例如007和012就有问题，直接删除就好，不差这一个。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/86Bw1.jpg)

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/vhlMm.jpg)



## 4. CT 降噪(不用)

可以看到，上面的真值CT还是有一些固定装置、或者CT机的伪影，这些也是我们不需要的，直接利用`simpleitk`的`otsu`去除。

```python
image = sitk.ReadImage(r"F:\MRI-CT-new\ct-mri\CT&mRI\geng_data_01\targets\004_ct.nii",sitk.sitkFloat32)
image_otsu = sitk.OtsuThreshold(image, 0, 1, 10)
mask_array = sitk.GetArrayFromImage(image_otsu)
image_array = sitk.GetArrayFromImage(image)
image_array[mask_array<0.5]=-1000 # 不要的区域将CT值改为-1000就变成了空气
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/jxFWb.jpg)



这样也不一定能完全处理好，好在我们有RT Structure文件，所以可以直接利用RT分割好的文件，直接用mask作筛选。

把RT文件放在CT的dicom文件目录下：

```python
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
#import SimpleITK as sitk
import matplotlib.pyplot as plt
import pydicom
from pydicom import dcmread

file = f"./RS.1.2.246.352.71.4.844417004102.341962.20210730090935.dcm"

dataset = dcmread(file)
print(dataset)
```

利用[dcmrtstruct2nii](https://github.com/Sikerdebaard/dcmrtstruct2nii)将RT文件转换成nii格式：

```python
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
print(list_rt_structs(file))
#dcmrtstruct2nii('/path/to/dicom/rtstruct/file.dcm', '/path/to/original/extracted/dicom/files', '/output/path')
dcmrtstruct2nii(file, './', './output/')
```

读取并显示mask：

```python
import SimpleITK as sitk
from matplotlib import pyplot as plt

def showNii(img):
    #for i in range(img.shape[0]):
    for i in range(100,120):
        plt.imshow(img[i,:,:],cmap='gray')
        plt.colorbar()
        plt.show()

itk_img = sitk.ReadImage(r"G:\MRI-CT-new\ct-mri\CT&mRI\001\CT\output\mask_Eye_R.nii.gz")
img = sitk.GetArrayFromImage(itk_img)
print(img.shape)
showNii(img)
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/XzCjF.jpg)



对生产的mask的nii文件同样也向着MRI配准一下：

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
        print(f"Now moving:{file_name}")
        moving_path = os.path.join(moving_mask_path, file_name)

        fixed_image = sitk.ReadImage(fixed_file_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        rmethod.SetInitialTransform(initial_transform, inPlace=False)
        final_transform = rmethod.Execute(fixed_image, moving_image)
        print('-> coregistered')

        moving_image = sitk.Resample(
            moving_image, fixed_image, final_transform, sitk.sitkLinear, .0,
            moving_image.GetPixelID())
        moving_image = sitk.Cast(moving_image, sitk.sitkInt16)
        print('-> resampled')

        os.makedirs(save_path, exist_ok=True)
        sitk.WriteImage(moving_image, os.path.join(save_path,file_name))
        print('-> exported')
        

coregister_mask(
    "/home/zhaosheng/paper2/data/geng01_new/geng_data_01/inputs/001_t1.nii",
    "/home/zhaosheng/paper2/data/geng01_new/001/RT/nii/",
    "/home/zhaosheng/paper2/data/geng01_new/001/RT/reg_nii/",
)
```



一般带有couch,rail的nii文件就是CT床之类的东西，直接读取mask，在对于位置给CT图像赋值-1000（空气即可）





## 5. MRI的N4偏置场校正（可选）

诸如扫描仪中的患者位置，扫描仪本身以及许多未知问题等因素可导致MR图像上的亮度差异。 换句话说，强度值（从黑色到白色）可以在同一组织内变化。 这被称为偏置场。 这是一种低频平滑的不良信号，会破坏MR图像。 偏置场导致MRI机器的磁场中的不均匀性。 如果未校正偏置字段将导致所有成像处理算法（例如，分段（例如，Freesurfer）和分类）输出不正确的结果。 在进行分割或分类之前，需要预处理步骤来校正偏置场的影响。
如下图所示：

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/e5frG.jpg)

```python
image = sitk.ReadImage(r"F:\MRI-CT-new\ct-mri\CT&mRI\geng_data_01\inputs\004_t1.nii", sitk.sitkFloat32)
image_otsu = sitk.OtsuThreshold(image, 0, 1, 10)


# N4 bias field correction
num_fitting_levels = 4
num_iters = 200
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrector.SetMaximumNumberOfIterations([num_iters] * num_fitting_levels)
cor_img = corrector.Execute(image, image_otsu)
cor_img = sitk.GetArrayFromImage(cor_img)
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/f6LQr.jpg)

```python
#-*-coding:utf-8-*-
import os
import shutil
import SimpleITK as sitk
import warnings
import glob
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection
 
 
def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: nii文件的输入路径
    :param out_file: 校正后的文件保存路径名
    :return: 校正后的nii文件全路径名
    """
    # 使用N4BiasFieldCorrection校正MRI图像的偏置场
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)
 
def normalize_image(in_file, out_file, bias_correction=True):
    # bias_correction：是否需要校正
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file
```





## 6. 生成numpy和png文件完成数据集的制作

```python
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# numpy的归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range)

root = r"F:\MRI-CT-new\ct-mri\CT&mRI\geng_data_01"


input_path =  os.path.join(root,"inputs")
target_path = os.path.join(root,"targets")

target_files = sorted(os.listdir(target_path))
input_files = sorted(os.listdir(input_path))

assert len(target_files) == len(input_files)


# 开始写入numpy
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
        
        target_file_name = "F:/MRI-CT-new/ct-mri/CT&mRI/geng_data_01/numpy/targets/"+"geng01_"+patient_id_pre+"_"+prefix+".npy"
        input_file_name = "F:/MRI-CT-new/ct-mri/CT&mRI/geng_data_01/numpy/inputs/"+"geng01_"+patient_id_pre+"_"+prefix+".npy"

        np.save(target_file_name, target_numpy_norm)
        np.save(input_file_name, input_numpy_norm)
        i = i+1
print(f"Done! Total:{i} pairs")
```

生成png文件：

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



## 7.神经网络的搭建、预测与训练

详见这篇文章]()或直接访问[Github]()。



## 8.结果的预处理

假设生成图片的地址为`Root`。

首先要提取A,B,fake_B放在不同的list中。读取灰度图（读出来是0-255的取值范围）并归一化，因为之前数据处理时设置的CT值范围为`[-1000,1800]`所以直接反向处理，把归一化的结果变回CT值。最后保存为numpy文件和nii文件。

我认为这样直接处理是可行的，因为CT图像本就是通过空气和水标定，这样简化处理不影响结果的准确性，但是可以减小工作量。

详细代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import SimpleITK as sitk
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range)

ROOT = "/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/20211109_test01_ttt/test_1/images/"

all_files  = os.listdir(ROOT)
real_B_images = sorted([os.path.join(ROOT,file_name) for file_name in all_files if "real_B" in file_name])
real_A_images = sorted([os.path.join(ROOT,file_name) for file_name in all_files if "real_A" in file_name])
fake_B_images = sorted([os.path.join(ROOT,file_name) for file_name in all_files if "fake_B" in file_name])

for i in range(len(real_A_images)):
    real_A = np.array(Image.open(real_A_images[i]).convert("L"))/255.
    real_B = np.array(Image.open(real_B_images[i]).convert("L"))/255.
    fake_B = np.array(Image.open(fake_B_images[i]).convert("L"))/255.
    fake_B[fake_B<0.05] = real_B[fake_B<0.05] # 小数据量时暂时去除部分噪点
    fake_B[fake_B>0.95] = real_B[fake_B>0.95]
    file_name = real_B_images[i].split("/")[-1].split("real")[-2]
    fake_B_HU = fake_B*(1800+1000)-1000
    real_B_HU = real_B*(1800+1000)-1000
    error = (fake_B_HU - real_B_HU)
    os.makedirs(os.path.join(ROOT,"numpy"), exist_ok=True)
    np.save(os.path.join(ROOT,"numpy",file_name+"fake_B"),fake_B_HU)
    np.save(os.path.join(ROOT,"numpy",file_name+"real_A"),real_A)
    np.save(os.path.join(ROOT,"numpy",file_name+"real_B"),real_B_HU)
    np.save(os.path.join(ROOT,"numpy",file_name+"error"),error)

fake_B_list_all = [os.path.join(ROOT,"numpy",_file) for _file in sorted(os.listdir(os.path.join(ROOT,"numpy"))) if "fake_B" in _file]
real_B_list_all = [os.path.join(ROOT,"numpy",_file) for _file in sorted(os.listdir(os.path.join(ROOT,"numpy"))) if "real_B" in _file]
real_A_list_all = [os.path.join(ROOT,"numpy",_file) for _file in sorted(os.listdir(os.path.join(ROOT,"numpy"))) if "real_A" in _file]
error_list_all = [os.path.join(ROOT,"numpy",_file) for _file in sorted(os.listdir(os.path.join(ROOT,"numpy"))) if "error" in _file]
```

```python
import os
os.getcwd()   #取得当前工作目录
os.chdir(r"D:\other_results\unet_256")
#filename = "001_ct" # 001_ct 038_ct 052_ct
for filename in ["001_ct" ,"038_ct", "052_ct"]:
    fake_B_list = [_file for _file in fake_B_list_all if filename in _file]
    real_B_list = [_file for _file in real_B_list_all if filename in _file]
    real_A_list = [_file for _file in real_A_list_all if filename in _file]
    error_list = [_file for _file in error_list_all if filename in _file]
    real_A_array = []
    for file in real_A_list:
        array_ = np.load(file)
        real_A_array.append(array_)
    real_A_array = np.array(real_A_array)
    real_A_array = np.transpose(real_A_array,(1,2,0))

    fake_B_array = []
    for file in fake_B_list:
        array_ = np.load(file)
        fake_B_array.append(array_)
    fake_B_array = np.array(fake_B_array)
    fake_B_array = np.transpose(fake_B_array,(1,2,0))

    real_B_array = []
    for file in real_B_list:
        array_ = np.load(file)
        real_B_array.append(array_)
    real_B_array = np.array(real_B_array)
    real_B_array = np.transpose(real_B_array,(1,2,0))

    real_A_out = sitk.GetImageFromArray(real_A_array)
    fake_B_out = sitk.GetImageFromArray(fake_B_array)
    real_B_out = sitk.GetImageFromArray(real_B_array)
    sitk.WriteImage(real_A_out,filename+"_real_A_out.nii")
    sitk.WriteImage(fake_B_out,filename+"_fake_B_out.nii")
    sitk.WriteImage(real_B_out,filename+"_real_B_out.nii")
```





## 9.结果的可视化

画一下横断面、矢状面和冠状面的结果，以及误差。

```python
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from os import listdir
from os.path import splitext
import nibabel as nb
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.viewers import OrthoSlicer3D
import scipy.io as io
import matplotlib.ticker as ticker

COLOR_BAR_FONT_SIZE = 6
for MODE in ["sag","cor","axi"]:        # 三个视图
    for pat_num in ["001","038","052"]: # 三个病例
        n = 110 # 第几层，可修改
        fake_img = nib.load("./"+pat_num+"_fake.nii")
        fake_ct_array = np.asanyarray(fake_img.dataobj)
        fake_ct_array = np.array(fake_ct_array)

        real_img = nib.load("./"+pat_num+"_real.nii")
        real_ct_array = np.asanyarray(real_img.dataobj)
        real_ct_array = np.array(real_ct_array)

        assert fake_ct_array.shape == real_ct_array.shape
        a,b,c = fake_ct_array.shape

        if MODE == "sag" or MODE == "cor":
            ax_aspect = b/a
        else:
            ax_aspect = 1

        func = lambda x,pos: "{:g}HU".format(x)
        fmt = ticker.FuncFormatter(func)

        # plt.colorbar(..., format=fmt)

        plt.figure(dpi=400)

        a1 = plt.subplot(1,3,1)

        if MODE == "sag":
            fake_array = fake_ct_array[:,:,n]
            real_array = real_ct_array[:,:,n]
        elif MODE == "cor":
            fake_array = fake_ct_array[:,n,:]
            real_array = real_ct_array[:,n,:]
        else:
            fake_array = fake_ct_array[n,:,:]
            real_array = real_ct_array[n,:,:]

        im =plt.imshow(fake_array,cmap='gray')
        a1.set_aspect(ax_aspect)
        plt.axis('off')
        plt.rcParams['font.size'] = COLOR_BAR_FONT_SIZE
        #plt.colorbar()
        cb1 = plt.colorbar(im, fraction=0.03, pad=0.05, format=fmt)
        tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数
        cb1.locator = tick_locator
        cb1.update_ticks()


        a2 = plt.subplot(1,3,2)
        im2 =plt.imshow(real_array,cmap='gray')
        a2.set_aspect(ax_aspect)
        plt.axis('off')
        plt.rcParams['font.size'] = COLOR_BAR_FONT_SIZE
        cb1 = plt.colorbar(im2, fraction=0.03, pad=0.05, format=fmt)
        tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数
        cb1.locator = tick_locator
        cb1.update_ticks()


        a3 = plt.subplot(1,3,3)
        im3 =plt.imshow(fake_array-real_array,cmap='gray')
        a3.set_aspect(ax_aspect)
        plt.axis('off')
        plt.rcParams['font.size'] = COLOR_BAR_FONT_SIZE
        cb1 = plt.colorbar(im3, fraction=0.03, pad=0.05, format=fmt)
        tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数
        cb1.locator = tick_locator
        cb1.update_ticks()


        print(f"MAE : {np.abs(fake_ct_array-real_ct_array).mean()}")
        print(f"ME : {(fake_ct_array-real_ct_array).mean()}")
        plt.tight_layout()
        plt.savefig("./"+pat_num+"_"+MODE+".jpg")
        plt.show()
        
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/4L3AN.jpg)



## 10.mat文件生成以及剂量计算

```python
nii_files = sorted([file_ for file_ in os.listdir("./") if "nii" in file_])

for nii in nii_files:
    filename = nii.split(".")[-2]
    matpath = filename+".mat"
    print(f"{nii} -> {matpath}")
    nii_to_mat(nii,matpath)
```

剂量计算采用Topas/Geant4，点击下载[代码文件]()。



## 11. 器官勾画

利用3DSlicer完成器官勾画。

到处时序号将会按照器官的顺序从1开始编号。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/6XIly.jpg)

## 12.DVH图分析

首先把上一步勾画后导出的结果，不同器官的mask提取出来，存在的地方为1，其他地方0填充。

```python
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from os import listdir
from os.path import splitext
import nibabel as nb
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.viewers import OrthoSlicer3D
import scipy.io as io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import csv

def getmask(array,label):
#     array_copy = array.copy
    output=np.zeros(array.shape)
    output[array_==label] = 1
    return output

real_ct = np.asanyarray(nib.load("./001_real.nii").dataobj)
np.save("./real/real_ct.npy",real_ct)
fake_ct = np.asanyarray(nib.load("./001_fake.nii").dataobj)
np.save("./fake/fake_ct.npy",fake_ct)

array_ = np.asanyarray(nib.load("./001.nii").dataobj)
# 0 air
array_air = getmask(array_,0)
np.save("./seg/array_air.npy",array_air)
# 1 l-lens
array_lens_l = getmask(array_,1)
np.save("./seg/array_lens_l.npy",array_lens_l)
# 2 r-lens
array_lens_r = getmask(array_,2)
np.save("./seg/array_lens_r.npy",array_lens_r)
# 3 shishenjing-l
array_ssj_l =getmask(array_,3)
np.save("./seg/array_ssj_l.npy",array_ssj_l)
# 4 shishenjing-r
array_ssj_r= getmask(array_,4)
np.save("./seg/array_ssj_r.npy",array_ssj_r)
# 5 brain
array_brain = getmask(array_,5)
np.save("./seg/array_brain.npy",array_brain)
# 6 brainstem
array_brainstem =getmask(array_,6)
np.save("./seg/array_brainstem.npy",array_brainstem)
# 6 brainstem
array_gtv =getmask(array_,7)
np.save("./seg/array_gtv.npy",array_gtv)

# 7 skull
array_skull =getmask(array_,8)
np.save("./seg/array_skull.npy",array_skull)
```

画DVH图：

```python
def get_dose_npy(file):
    with open(file,encoding = 'utf-8') as f:
        data = np.loadtxt(f,delimiter = ",", skiprows = 8)
    #return data
    output = data[:,3].reshape(256,256,187).transpose(2,0,1)
    filename = file.split("/")[-1].split(".")[-2]
    if "fake" in file:
        np.save("./fake_out/"+filename+".npy",output)
    elif "real" in file:
        np.save("./real_out/"+filename+".npy",output)
    print(f"{filename} Done.")
    return 0

def get_array(fold,name):
    boron = [file_ for file_ in os.listdir(fold) if name in file_]
    #print(boron)
    boron_array = np.load(os.path.join(fold,boron[0]))
    return boron_array

def get_dvh(organ_array,dose_array,label):
    if len(organ_array[organ_array>0]) == 0:
        xx = np.linspace(0,1,5000)
        yy = np.linspace(0,1,5000)
        xx= xx.tolist()
        yy= yy.tolist()
        return xx,yy
    out_dose = np.zeros(organ_array.shape)
    out_dose[organ_array>0] = dose_array[organ_array>0]
    #print("skin_dose shape:{}".format(skin_dose.shape))
    max_dose = dose_array.max()
    xx = np.linspace(0,max_dose,101)
    xx = xx[:100]
    yy = []
    for x in xx:
        yy.append(len(out_dose[out_dose>x]) / len(organ_array[organ_array>0]))
    #yy = np.array(yy)
    """    
    xx = torch.from_numpy(xx)
    yy = torch.from_numpy(yy)
    xx=Variable(xx.cuda(),requires_grad=True)
    yy=Variable(yy.cuda(),requires_grad=True)
    """
    xx= xx.tolist()
    #yy= yy.tolist()
    yy[-1]=0
    return [xx,yy,label]

def get_dose_dvh(fold):
    r=36
    n=6.5
    xx = n*r*3.1415926
    t=50 #肿瘤
    s=25 #皮肤
    nt=18#NT硼浓度
    
    ct = get_array(fold,"ct")
    boron_array = get_array(fold,"boron")
    fast_array = get_array(fold,"fast")
    gamma_array = get_array(fold,"gamma")
    nitrogen_array = get_array(fold,"nitrogen")
    air = get_array("./seg/","air")
    lens_l = get_array("./seg/","lens_l")
    lens_r = get_array("./seg/","lens_r")
    ssj_l = get_array("./seg/","ssj_l")
    ssj_r = get_array("./seg/","ssj_r")
    brain = get_array("./seg/","brain")
    skull = get_array("./seg/","skull")
    brainstem = get_array("./seg/","brainstem")
    gtv = get_array("./seg/","gtv")
#     print(gtv.shape)
    total_dose= np.zeros(ct.shape)
    total_dose[ct<-999] = boron_array[ct<-999]*xx* \
                    + fast_array[ct<-999]*xx \
                    + gamma_array[ct<-999]*xx \
                    + nitrogen_array[ct<-999]*xx
    
    total_dose = boron_array*xx*nt \
                + fast_array*xx \
                + gamma_array*xx \
                + nitrogen_array*xx
    
    total_dose[gtv==1] = boron_array[gtv==1]*xx*t \
                + fast_array[gtv==1]*xx \
                + gamma_array[gtv==1]*xx \
                + nitrogen_array[gtv==1]*xx
    dvh_list = []
    dvh_list.append(get_dvh(lens_l,total_dose,"Left lens"))
    dvh_list.append(get_dvh(lens_r,total_dose,"Right lens"))
    dvh_list.append(get_dvh(ssj_l,total_dose,"Left optic nerve"))
    dvh_list.append(get_dvh(ssj_r,total_dose,"Right optic nerve"))
    dvh_list.append(get_dvh(brain,total_dose,"Brain"))
    dvh_list.append(get_dvh(brainstem,total_dose,"BrainStem"))
    dvh_list.append(get_dvh(gtv,total_dose,"GTV"))
    dvh_list.append(get_dvh(skull,total_dose,"Skull"))
    return total_dose,ct,dvh_list
```

生成各个剂量文件，这一步比较耗时。

```python
# get_dose_npy("./fake/boron10.csv")
# get_dose_npy("./fake/fast10.csv")
# get_dose_npy("./fake/gamma10.csv")
# get_dose_npy("./fake/nitrogen10.csv")

# get_dose_npy("./real/boron10.csv")
# get_dose_npy("./real/fast10.csv")
# get_dose_npy("./real/gamma10.csv")
# get_dose_npy("./real/nitrogen10.csv")
```

直接plot就行。

```python
fake_total_dose,fake_ct,fake_dvh_list = get_dose_dvh("./fake_out/")
real_total_dose,real_ct,real_dvh_list = get_dose_dvh("./real_out/")
plt.figure(dpi=150)
# for dvh in fake_dvh_list:
dvh = fake_dvh_list[-1]
plt.plot(dvh[0],dvh[1],'b-',label=dvh[2],linewidth=1)
# for dvh in real_dvh_list:
dvh2 = real_dvh_list[-1]
plt.plot(dvh2[0],dvh2[1],'r--',label=dvh2[2],linewidth=1)
plt.legend()
plt.show()
plt.figure(dpi=150)
for dvh in fake_dvh_list:

    plt.plot(dvh[0],dvh[1],'-',label=dvh[2],linewidth=1)
for dvh2 in real_dvh_list:

    plt.plot(dvh2[0],dvh2[1],'--',label=dvh2[2],linewidth=1)
plt.legend()
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/Fctu5.jpg)

## 13.画剂量分布和CT的叠加图

```python

```





## 14.特征图的可视化（对比注意力机制效果）





## 15. MAE





## 16. SSIM与PSNR

### ssim

```python
import numpy 
import numpy as np
import math
import cv2
import torch
import pytorch_ssim
from torch.autograd import Variable

def ssim(img1,img2):
    img1 = torch.from_numpy(np.rollaxis(img1,2)).float().unsqueeze(0)/255.0
    img2 = torch.from_numpy(np.rollaxis(img2,2)).float().unsqueeze(0)/255.0
    img1 = Variable(img1,requires_grad=False) # torch.Size([256,256,3])
    img2 = Variable(img2,requires_grad=False)
    ssim_value = pytorch_ssim(img1,img2).item()
    return ssim_value
```





### psnr

$$
 P S N R=10 \cdot \log 10\left(\frac{M A X_{I}^{2}}{M S E}\right)=20 \cdot \log 10\left(\frac{M A X_{I}}{\sqrt{M S E}}\right) 
$$

```python
def psnr(img1,img2):
    mse = np.mean((img1/255.-img2/255.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20*log10(PIXEL_MAX/math.sqrt(mse))
```



## 17.伽马通过率

```python
'''
Created on 2018-12-06

@author: Louis Archambault
'''

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt

# import skfmm # [scikit-fmm](https://github.com/scikit-fmm/scikit-fmm)

class gamma:
   """ Gamma calculation (global) based on Chen et al. 2009

       WARNING (for now): assume same sampling between ref and meas
       Compute one distance map then apply to multiple measurements
       see reading notes and jupyter_notebook/gamma
   """


   def __init__(self,dose_crit,dist_crit,vox_size,threshold = None,ref=None):
      """ initialization

          ref: reference dose distribution a 3D numpy array
          vox_size: voxel size [x,y,z] in mm
            - Note that image shape is infered from vox_size
          dose_crit: \Delta [percent of max]
          dist_crit: \delta [mm]
          threshold: minimum dose below which gamma value is not computed [percent of max]
      """

      self.dist_crit = dist_crit
      self.dose_crit = dose_crit
      self.min_threshold_pct = threshold
      self.set_voxel_size(vox_size)
      self.ndim = len(self.vox_size)
      # print(f'Image dimensions: {self.ndim}')
      self.max_dose_pct = 120 # [%], fraction of maximum to use as the upper dose bin

      # empty initialization, waiting for ref image
      self.ndbins = None # number of dose bins
      self.dbins = []
      self.ref_img = None
      self.dist_map = None
      self.delta = None # voxel sizes as fx of criterion

      if ref is not None:
         self.set_reference(ref)

   def set_voxel_size(self,vox_size):
      self.vox_size = np.array(vox_size) # calling np.array does nothing
                                         # if vox_size is already a ndarray

   def set_reference(self,ref):

      # dx -> x,y,z distance as fx of distance criterion
      dx = self.vox_size/self.dist_crit # voxel dimensions expressed as fx of criterion
      dd = np.mean(dx) # dimension along dose axis should be the same as space axis.
                       # Dose bins are set afterward
      # print(dx,dd)

      # absolute dose criterion = % of max
      max_dose =  np.max(ref)
      abs_dose_crit = max_dose*self.dose_crit/100.

      # absolute threshold criterion
      if self.min_threshold_pct:
         self.min_threshold = self.min_threshold_pct/100 * max_dose
      else:
         self.min_threshold = None

      # Set the number of dose bins to match the voxel dose length (i.e. dd)
      # dd = dose_vox_size/abs_dose_crit
      # dose_vox_size = max_dose/ndbins
      # max_dose = (maximum of ref)*max_dose_pct
      # -> ndbins = max_dose/(dd * abs_dose_crit)
      ndbins = int(np.ceil(max_dose*self.max_dose_pct/100/dd/abs_dose_crit))
      dbins = np.linspace(0,max_dose*self.max_dose_pct/100,ndbins+1)

      # assign important data to self
      self.ref_img = ref
      self.delta = np.append(dx,dd) # voxel sizes as fx of criterion
      self.ndbins = ndbins
      self.dbins = dbins

      self.compute_dist_map()

   def compute_dist_map(self):
      """ Compute distance map (can be time consuming) """
      # Build an hyper surface with three spatial dimensions + 1 dose dimension
      hypSurfDim = self.ref_img.shape + (self.ndbins,)
      hypSurf = np.ones( hypSurfDim )

      # Fill each layer of the dose axis
      # Dose points are set to 0

      lookup = np.digitize(self.ref_img,self.dbins) - 1 # lookup contains the index of dose bins

      for i in range(self.ndbins):
         dose_points = lookup == i
         if self.ndim == 3:
            hypSurf[:,:,:,i][dose_points] = 0
            # simple (naive) interpolation. See Fig. 2 au Chen 2009
            hypSurf = self._interp_dose_along_ax3(hypSurf,lookup,0)
            hypSurf = self._interp_dose_along_ax3(hypSurf,lookup,1)
            hypSurf = self._interp_dose_along_ax3(hypSurf,lookup,2)
            # Here, we could try to mask layer by layer all position of pixels below threshold
            # to speed up calculation (only w/ skfmm)
         elif self.ndim == 2:
            hypSurf[:,:,i][dose_points] = 0
            # simple (naive) interpolation. See Fig. 2 au Chen 2009
            hypSurf = self._interp_dose_along_ax2(hypSurf,lookup,0)
            hypSurf = self._interp_dose_along_ax2(hypSurf,lookup,1)
            # Here, we could try to mask layer by layer all position of pixels below threshold
            # to speed up calculation (only w/ skfmm)
         else:
            raise IndexError('Only 2 and 3 spatial dimension supported at this moment')

      dst = edt(hypSurf,sampling=self.delta)
      # dst = skfmm.distance(hypSurf)

      self.dist_map = dst

   def _interp_dose_along_ax3(self,hypsrf,lookup,ax):
      """ interpolate the dose axis along a given spatial axis (3D)

          lookup: a matrix of the shape of spatial dimensions where each elements
                  corresponds to the dose bin at that point (produced by np.digitize)
          hypsrfs: the hypersurface: [x,y,z,dose_bin]
          ax: the index along which the interpolation is done (x=0,y=1,z=2)
      """

      dims = self.ref_img.shape # the spatial dimensions triplet (x, y, z)
      hs = np.copy(hypsrf)

      ax_slc = slice(None) # take the whole ax (same as [:])
      ax_order = [ax] + [x for x in [0,1,2] if x != ax] # [ax, oax1, oax2]
      ax_sort = np.argsort(ax_order)

      # a function that take (ax, oax1, oax2) and return (x,y,z)
      xyz = lambda x,y,z : ([x,y,z][ax_sort[0]], [x,y,z][ax_sort[1]], [x,y,z][ax_sort[2]])

      for oax1 in range(dims[ax_order[1]]):
         for oax2 in range(dims[ax_order[2]]):
            v = lookup[xyz(ax_slc,oax1,oax2)]
            jumps = np.diff(v) # number of pixels in gap
            idx_jump = np.argwhere(np.abs(jumps) > 1).flatten() # next dose jumps by more than 1 bin

            for i in idx_jump:
               # starting point to pad on hyper surface: (i,v[i])
               pad = 0
               if jumps[i] < 0: # dose decrease
                  pad = int(np.floor(jumps[i]/2)) # divide the gap in 2
                  hs[xyz(i,oax1,oax2) + (slice((v[i] + pad),v[i]),)] = 0 # pad down
                  hs[xyz(i+1,oax1,oax2) + (slice(v[i+1],(v[i+1]-pad)),)] = 0 # pad up
               else: # dose increase
                  pad = int(np.ceil(jumps[i]/2))
                  hs[xyz(i,oax1,oax2) + (slice(v[i],(v[i]+pad)),)] = 0 # pad up
                  hs[xyz(i+1,oax1,oax2) + (slice((v[i+1] - pad),v[i+1]),)] = 0 # pad down
      return hs

   def _interp_dose_along_ax2(self,hypsrf,lookup,ax):
      """ interpolate the dose axis along a given spatial axis (2D)

          lookup: a matrix of the shape of spatial dimensions where each elements
                  corresponds to the dose bin at that point (produced by np.digitize)
          hypsrfs: the hypersurface: [x,y,z,dose_bin]
          ax: the index along which the interpolation is done (x=0,y=1,z=2)
      """

      dims = self.ref_img.shape # the spatial dimensions triplet (x, y, z)
      hs = np.copy(hypsrf)

      ax_slc = slice(None) # take the whole ax (same as [:])
      ax_order = [ax] + [x for x in [0,1] if x != ax] # [ax, oax1]
      ax_sort = np.argsort(ax_order)

      # a function that take (ax, oax1, oax2) and return (x,y,z)
      xy = lambda x,y : ([x,y][ax_sort[0]], [x,y][ax_sort[1]])

      for oax1 in range(dims[ax_order[1]]):
         #for oax2 in range(dims[ax_order[2]]):
         v = lookup[xy(ax_slc,oax1)]
         jumps = np.diff(v) # number of pixels in gap
         idx_jump = np.argwhere(np.abs(jumps) > 1).flatten() # next dose jumps by more than 1 bin

         for i in idx_jump:
            # starting point to pad on hyper surface: (i,v[i])
            pad = 0
            if jumps[i] < 0: # dose decrease
               pad = int(np.floor(jumps[i]/2)) # divide the gap in 2
               hs[xy(i,oax1) + (slice((v[i] + pad),v[i]),)] = 0 # pad down
               hs[xy(i+1,oax1) + (slice(v[i+1],(v[i+1]-pad)),)] = 0 # pad up
            else: # dose increase
               pad = int(np.ceil(jumps[i]/2))
               hs[xy(i,oax1) + (slice(v[i],(v[i]+pad)),)] = 0 # pad up
               hs[xy(i+1,oax1) + (slice((v[i+1] - pad),v[i+1]),)] = 0 # pad down
      return hs

   def compute_gamma(self,img):
      """ compute gamma between img and ref
          img must have the same number and size of pixels
      """

      assert self.ref_img.shape == img.shape, 'reference and test must have the same dimensions'

      lookup = np.digitize(img,self.dbins) - 1 # values to lookup in dist_map
      gamma_map = np.ones(img.shape)*999 # initialize
      # print(gamma_map.shape)

      #gamma values corrresponds to the pixel values on dist_map at the location of img
      for i in range(self.ndbins):
         test_points = lookup == i
         if self.ndim == 3:
            gamma_map[test_points] = self.dist_map[:,:,:,i][test_points]
         elif self.ndim == 2:
            gamma_map[test_points] = self.dist_map[:,:,i][test_points]
         else:
            raise IndexError('Only 2 and 3 spatial dimension supported at this moment')

      if self.min_threshold:
         msk = self.ref_img < self.min_threshold
         gamma_map = np.ma.array(gamma_map,mask=msk)

      return gamma_map

   def __call__(self,img):
      """ compute the gamma between ref and img
      """

      return self.compute_gamma(img)

```



使用方法：

```python
from gamma3D import gamma
import numpy as np

Delta = 3 # [% of max]
delta = 3 # [mm]

ref = np.ones([20,20,20])
test = np.ones([20,20,20])

g = gamma(Delta,delta,[1,1,1])
g.set_reference(ref)
gamma_map = g(test)
```



## 18. CT值分布

