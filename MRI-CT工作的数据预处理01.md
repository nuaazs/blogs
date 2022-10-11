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



## 4. CT 降噪

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





## 5. MRI的N4偏置场校正

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





## 6. 生成numpy文件完成数据集的制作

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



