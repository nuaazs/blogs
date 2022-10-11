## 前言

这篇文章记录MR-CT数据集制作过程，和之前的不同，考虑审稿意见，这次换用ANTs做非刚性配准。



## 1. 数据预处理

参数定义及文件夹生成：

```python
import ants
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from glob import glob
import shutil
import re
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
os.environ[ "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" ] = "40"
os.environ[ "ANTS_RANDOM_SEED" ] = "3"


ROOT = '/home/iint/lmz/ct_t1/raw_data'
SAVE_PATH = r"/home/iint/lmz/ct_t1/reg_data"
OUTPUT_PATH = r"/home/iint/lmz/ct_t1/SYNRA_data/"
PNG_PATH = r"/home/iint/lmz/ct_t1/SYNRA_png/"
n4_path = os.path.join(OUTPUT_PATH,"n4")
moved_path = os.path.join(OUTPUT_PATH,"moved_ct")
n4_denoise_path = os.path.join(OUTPUT_PATH,"n4_denoise")
os.makedirs(n4_path,exist_ok=True)
os.makedirs(moved_path,exist_ok=True)
os.makedirs(n4_denoise_path,exist_ok=True)
```

### 1.1 读取Dicom序列，保存为raw_nii

读取原始的dicom文件（从医院获取到的），`t1_data_info.csv`为人工筛选的T1 MR相关信息，保存为nii文件。

```python
def save_nii(files_path,save_path):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(files_path)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(files_path, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    # print(f"读取到的图片大小为：{image3D.GetSize()}")
    sitk.WriteImage(image3D, save_path)

df= pd.read_csv(
    filepath_or_buffer = r"/home/iint/lmz/ct_t1/t1_data_info.csv",
    sep= ',',
    error_bad_lines= False ,
    na_values= 'NULL'
)

t1_mri_df = df[df["Modality"] == "MR"]
for index, row in t1_mri_df.iterrows():
    fold_id = str(row["FOLD_ID"]).zfill(3) # 2 变 002
    ct_path = os.path.join(ROOT,fold_id,"CT")
    mr_path = os.path.join(ROOT,fold_id,"MRI",row["SeriesDescription                    "]) # SeriesDescription 后面有空格
    # print(ct_path)
    # print(mr_path)
    save_nii(ct_path,f"{SAVE_PATH}/{fold_id}_ct.nii")
    save_nii(mr_path,f"{SAVE_PATH}/{fold_id}_t1.nii")
```



### 1.2 初步配准（之前的刚性迭代配准）（可以不需要了）



### 1.3 ANTs非刚性配准、降噪、N3/N4矫正

```python
n4_path = os.path.join(OUTPUT_PATH,"n4")
moved_path = os.path.join(OUTPUT_PATH,"moved_ct")
n4_denoise_path = os.path.join(OUTPUT_PATH,"n4_denoise")
os.makedirs(n4_path,exist_ok=True)
os.makedirs(moved_path,exist_ok=True)
os.makedirs(n4_denoise_path,exist_ok=True)

plots_path = os.path.join(OUTPUT_PATH,"plots")
os.makedirs(plots_path,exist_ok=True)

# numpy的归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range)


target_files = sorted([_file for _file in sorted(os.listdir(SAVE_PATH)) if "ct" in _file])
input_files =  sorted([_file for _file in sorted(os.listdir(SAVE_PATH)) if "t1" in _file])
assert len(target_files) == len(input_files)


for patient_id,(input_file,target_file) in enumerate(zip(input_files,target_files)):
    moving = ants.image_read(os.path.join(SAVE_PATH,target_file))
    fixed = ants.image_read(os.path.join(SAVE_PATH,input_file))
    filename = input_file.split(".")[-2].split("_")[-2]
    print(f"\n=>\tNow loading:{filename}")
    moving += 1000
    
    moving.plot(title='moving',axis=1,filename=)
    moving.plot(title='moving',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_moving.png"))
    fixed.plot(title='fixed',axis=1)
    fixed.plot(title='fixed',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_fixed.png"))
#     reg = ants.registration(fixed=fixed, moving=moving,
#                        type_of_transform='SyNOnly', grad_step=0.5, flow_sigma=0, total_sigma=0, reg_iterations=(100, 40, 20), syn_metric='CC', syn_sampling=4, verbose=True)
    reg = ants.registration(fixed=fixed, moving=moving,type_of_transform='SyN', reg_iterations = [100,100,20])

    moved = reg["warpedmovout"]
    moved.plot(title='moved',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_moved.png"))
    
    
    fixed_n4 = ants.n4_bias_field_correction(fixed)
    fixed_n4.plot(title='fixed_n4',axis=1)
    fixed_n4.plot(title='fixed_n4',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_fixed_n4.png"))
    fixed_n4_denoise = ants.denoise_image(fixed_n4)
    fixed_n4_denoise.plot(title='fixed_n4_denoise',axis=1)
    fixed_n4_denoise.plot(title='fixed_n4_denoise',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_fixed_n4_denoise.png"))
    
    
    ants.plot(moved,overlay=fixed_n4_denoise,overlay_cmap='hot',overlay_alpha=0.5,axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_overlay_1.png"))#
    ants.plot(moved,overlay=fixed_n4_denoise,overlay_cmap='hot',overlay_alpha=0.5,axis=0,cbar=True,filename=os.path.join(plots_path,filename+"_overlay_0.png"))#

    # SAVE nii
    ants.image_write(moved,os.path.join(moved_path,filename+".nii"))
    ants.image_write(fixed_n4,os.path.join(n4_path,filename+".nii"))
    ants.image_write(fixed_n4_denoise,os.path.join(n4_denoise_path,filename+".nii"))
    break
```



### 1.4 写为png

```python
import ants
# numpy的归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range)


png_path = r"./SyNRA_reg_numpy/"
os.makedirs(png_path,exist_ok=True)
target_files = sorted(os.listdir(moved_path))
#input_files = sorted([_file for _file in os.listdir(ROOT) if "t1" in _file])
#assert len(target_files) == len(input_files)
print(moved_path)
print(len(target_files))
i = 0
for patient_id,file_name in enumerate(target_files):
    print(file_name)
    file_name = file_name.split(".")[-2]
    print(file_name)
    target_filepath = os.path.join(moved_path,target_files[patient_id])
    input_filepath = os.path.join(ROOT,f"{file_name}_t1.nii")

    target_img = ants.image_read(target_filepath)
    input_img = ants.image_read(input_filepath)
    
    target_img = ants.n3_bias_field_correction(target_img, downsample_factor=3)
    input_img = ants.n3_bias_field_correction(input_img, downsample_factor=3)
    
    
    #new_spacing = np.array(moving.spacing)*4
    #moving.spacing
    #moving = ants.resample_image(moving,(64,64,100),True,4)
    input_img = ants.resample_image(input_img,(1,1,5),False,4)
    target_img = ants.resample_image(target_img,(1,1,5),False,4)

    # ants.plot(target_img)
    # ants.plot(input_img)
    
    
    
    target_data = target_img.numpy()
    input_data = input_img.numpy()

    # print(target_filepath)
    # print(input_filepath)
    # print(target_data.shape)
    # print(input_data.shape)
    # print(input_data.shape)
    if input_data.shape== target_data.shape:
        pass
    else:
        print(f"pass: {file_name}")
        continue
    assert input_data.shape== target_data.shape
    
    x_nums,y_nums,z_nums = input_data.shape
    _a,_b,slices_num = input_data.shape
    
    input_data = normalization(input_data)
    target_data = normalization(target_data)
    
    for slice in range(slices_num):
        target_numpy = np.array(target_data[:,:,slice])
        input_numpy = np.array(input_data[:,:,slice])
        # 删除空白图片
        if (np.max(target_numpy) == np.min(target_numpy)) or (np.max(input_numpy) == np.min(input_numpy)) :
            continue

        patient_id_pre = '%03d' % (patient_id+1)
        prefix='%03d' % i
        
        pic_path = os.path.join(png_path,file_name)+f"_{slice}.png"
        pic_array = np.hstack([target_numpy*255,input_numpy*255])
        img = Image.fromarray(pic_array)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((512, 256), Image.ANTIALIAS)

        img.save(pic_path)
        i = i+1
print(f"Done! Total:{i} pairs")
```

对于3d维度不同的数据，还需要单独处理：

```python
def padding_image(input_data)：
    a,b=input_data.shape
    input_data_copy = input_data.copy()
    if a < b:
        edge_width = (b-a)//2
        pic = np.zeros((b,b))
        pic[edge_width+1:b-edge_width,:] = input_data_copy

    elif a > b:
        edge_width = (a-b)//2
        pic = np.zeros((a,a))
        pic[:,edge_width+1:a-edge_width] = input_data_copy
    return pic

for file_name in ["012","xxx", ...]:
    input_filepath = os.path.join(INPUT_ROOT,f"{file_name}_t1.nii")
    input_img = ants.image_read(input_filepath)
    input_img = ants.resample_image(input_img,(1,5,1),False,4)
    input_data = input_img.numpy()

    target_filepath = os.path.join(TARGE_ROOT,f"{file_name}_t1.nii")
    target_img = ants.image_read(target_filepath)
    target_img = ants.resample_image(target_img,(1,5,1),False,4)
    target_data = target_img.numpy()

    _a,_b,slices_num = input_data.shape
    
    input_data = normalization(input_data)
    target_data = normalization(target_data)
    assert input_data.shape== target_data.shape
    
    for _slice in range(slices_num):
        # 对于矢状面，slices 在中间
        input_numpy = padding_image(np.array(input_data[:,_slice,:]))
        target_numpy = padding_image(np.array(target_data[:,_slice,:]))
        
        # 删除空白图片
        if (np.max(target_numpy) == np.min(target_numpy)) or (np.max(input_numpy) == np.min(input_numpy)) :
            continue

        patient_id_pre = '%03d' % (patient_id+1)
        prefix='%03d' % i
        
        pic_path = os.path.join(png_path,file_name)+f"_{slice}.png"
        pic_array = np.hstack([target_numpy*255,input_numpy*255])
        img = Image.fromarray(pic_array)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((512, 256), Image.ANTIALIAS)

        img.save(pic_path)
        i = i+1
print(f"Done! Total:{i} pairs")
```



### 1.5 mask分割去除不需要的部分

```python
from ants_reg import reg,save_png
from utils import otsu_crop,load_nii
otsu_crop(data=args.png_path, temp_id=args.temp_id,size=args.size, delay=args.delay, is_save=args.is_save)
```



ants_reg.py:

```python
# import
import os
import ants
import numpy as np
from PIL import Image
import nibabel as nib
# from multiprocessing.dummy import Pool as ThreadPool
from utils import normalization

def reg(regdata_path,welldata_path,type_of_transform):
    # makedirs
    target_files = sorted([_file for _file in sorted(
        os.listdir(regdata_path)) if "ct" in _file])
    input_files = sorted([_file for _file in sorted(
        os.listdir(regdata_path)) if "t1" in _file])
    assert len(target_files) == len(input_files)
    n4_path = os.path.join(welldata_path, "n4")
    moved_path = os.path.join(welldata_path, "moved_ct")
    n4_denoise_path = os.path.join(welldata_path, "n4_denoise")
    plots_path = os.path.join(welldata_path, "plots")
    denoise_path = os.path.join(welldata_path, "denoise")
    os.makedirs(denoise_path, exist_ok=True)
    os.makedirs(n4_path, exist_ok=True)
    os.makedirs(moved_path, exist_ok=True)
    os.makedirs(n4_denoise_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    for (input_file, target_file) in zip(input_files, target_files):
            try:
                moving = ants.image_read(os.path.join(
                    regdata_path, target_file))
                fixed = ants.image_read(os.path.join(
                    regdata_path, input_file))
                filename = input_file.split(".")[-2].split("_")[-2]
                print(f"=> Now loading:{filename}")
                moving += 1000

                moving.plot(title='moving', axis=1, cbar=True, filename=os.path.join(
                    plots_path, filename+"_moving.png"))
                fixed.plot(title='fixed', axis=1, cbar=True,
                        filename=os.path.join(plots_path, filename+"_fixed.png"))
            #     reg = ants.registration(fixed=fixed, moving=moving,
            #                        type_of_transform=type_of_transform, grad_step=0.5, flow_sigma=0, total_sigma=0, reg_iterations=(100, 40, 20), syn_metric='CC', syn_sampling=4, verbose=True)
                print(f"\t-> Registration.")
                reg = ants.registration(
                    fixed=fixed, moving=moving, type_of_transform=type_of_transform, reg_iterations=[100, 100, 20])

                moved = reg["warpedmovout"]
                moved.plot(title='moved', axis=1, cbar=True,
                        filename=os.path.join(plots_path, filename+"_moved.png"))

                print(f"\t-> N4 bias.")
                fixed_n4 = ants.n4_bias_field_correction(fixed)
                fixed_n4.plot(title='fixed_n4', axis=1, cbar=True, filename=os.path.join(
                    plots_path, filename+"_fixed_n4.png"))
                
                print(f"\t-> Denosing correction.")

                fixed_n4_denoise = ants.denoise_image(fixed_n4)
                fixed_n4_denoise.plot(title='fixed_n4_denoise', axis=1, cbar=True, filename=os.path.join(
                    plots_path, filename+"_fixed_n4_denoise.png"))
                
                fixed_denoise = ants.denoise_image(fixed)
                fixed_denoise.plot(title='fixed_denoise', axis=1, cbar=True, filename=os.path.join(
                    plots_path, filename+"_fixed_denoise.png"))

                ants.plot(moved, overlay=fixed_n4_denoise, overlay_cmap='hot', overlay_alpha=0.5,
                        axis=1, cbar=True, filename=os.path.join(plots_path, filename+"_overlay_1.png"))
                ants.plot(moved, overlay=fixed_n4_denoise, overlay_cmap='hot', overlay_alpha=0.5,
                        axis=0, cbar=True, filename=os.path.join(plots_path, filename+"_overlay_0.png"))

                # SAVE nii
                print(f"\t-> Saving niis.")
                ants.image_write(moved, os.path.join(moved_path, filename+".nii"))
                ants.image_write(fixed_n4, os.path.join(n4_path, filename+".nii"))
                ants.image_write(fixed_n4_denoise, os.path.join(
                    n4_denoise_path, filename+".nii"))
                ants.image_write(fixed_denoise, os.path.join(
                    denoise_path, filename+".nii"))
            except:
                print(f"=> !!!!ERROR:{filename}")

    return n4_path,moved_path,n4_denoise_path,plots_path,denoise_path


def get_paths(regdata_path,welldata_path,type_of_transform):
    # makedirs
    target_files = sorted([_file for _file in sorted(
        os.listdir(regdata_path)) if "ct" in _file])
    input_files = sorted([_file for _file in sorted(
        os.listdir(regdata_path)) if "t1" in _file])
    assert len(target_files) == len(input_files)
    n4_path = os.path.join(welldata_path, "n4")
    moved_path = os.path.join(welldata_path, "moved_ct")
    n4_denoise_path = os.path.join(welldata_path, "n4_denoise")
    plots_path = os.path.join(welldata_path, "plots")
    denoise_path = os.path.join(welldata_path, "denoise")
    os.makedirs(denoise_path, exist_ok=True)
    os.makedirs(n4_path, exist_ok=True)
    os.makedirs(moved_path, exist_ok=True)
    os.makedirs(n4_denoise_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    return n4_path,moved_path,n4_denoise_path,plots_path,denoise_path


def save_png(png_path,moved_path,denoise_path):
    target_files = sorted(os.listdir(moved_path))
    print(f"Targets: {target_files}")
    input_files = sorted(os.listdir(denoise_path))
    assert len(target_files) == len(input_files)
    i = 0
    for patient_id, file_name in enumerate(target_files):
        file_name = file_name.split(".")[-2]
        target_filepath = os.path.join(moved_path, target_files[patient_id])
        input_filepath = os.path.join(denoise_path, input_files[patient_id])

        target_img = ants.image_read(target_filepath)
        input_img = ants.image_read(input_filepath)

        target_img = ants.n3_bias_field_correction(target_img, downsample_factor=3)
        input_img = ants.n3_bias_field_correction(input_img, downsample_factor=3)

        input_img = ants.resample_image(input_img,(1,1,5),False,4)
        target_img = ants.resample_image(target_img,(1,1,5),False,4)

        # ants.plot(target_img)
        # ants.plot(input_img)



        target_data = target_img.numpy()
        input_data = input_img.numpy()
        if input_data.shape == target_data.shape:
            pass
        else:
            print(f"\t=> pass: {file_name}")
            continue
        assert input_data.shape == target_data.shape
        _a, _b, slices_num = input_data.shape

        input_data = normalization(input_data)
        target_data = normalization(target_data)

        for slice in range(slices_num):

            target_numpy = np.array(target_data[:, :, slice])
            input_numpy = np.array(input_data[:, :, slice])

            if (np.max(target_numpy) == np.min(target_numpy)) or (np.max(input_numpy) == np.min(input_numpy)):
                continue
            patient_id_pre = '%03d' % (patient_id+1)
            prefix = '%03d' % i
            pic_path = os.path.join(png_path, file_name)+f"_{slice}.png"
            pic_array = np.hstack([target_numpy*255, input_numpy*255])
            img = Image.fromarray(pic_array)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((512, 256), Image.ANTIALIAS)

            img.save(pic_path)
            i = i+1
    print(f"\t=> Total:{i} pairs png.")

# multiprocessing
# def ants_reg(input_file,target_file):
#     moving = ants.image_read(os.path.join(args.regdata_path,target_file))
#     fixed = ants.image_read(os.path.join(args.regdata_path,input_file))
#     filename = input_file.split(".")[-2].split("_")[-2]
#     print(f"=> Now loading:{filename}")
#     moving += 1000

#     moving.plot(title='moving',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_moving.png"))
#     fixed.plot(title='fixed',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_fixed.png"))
# #     reg = ants.registration(fixed=fixed, moving=moving,
# #                        type_of_transform='SyNOnly', grad_step=0.5, flow_sigma=0, total_sigma=0, reg_iterations=(100, 40, 20), syn_metric='CC', syn_sampling=4, verbose=True)
#     print(f"\t-> Registration.")
#     reg = ants.registration(fixed=fixed, moving=moving,type_of_transform='SyN', reg_iterations = [100,100,20])

#     moved = reg["warpedmovout"]
#     moved.plot(title='moved',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_moved.png"))

#     print(f"\t-> N4 bias.")
#     fixed_n4 = ants.n4_bias_field_correction(fixed)
#     fixed_n4.plot(title='fixed_n4',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_fixed_n4.png"))
#     print(f"\t-> Denosing correction.")
#     fixed_n4_denoise = ants.denoise_image(fixed_n4)
#     fixed_n4_denoise.plot(title='fixed_n4_denoise',axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_fixed_n4_denoise.png"))
#     ants.plot(moved,overlay=fixed_n4_denoise,overlay_cmap='hot',overlay_alpha=0.5,axis=1,cbar=True,filename=os.path.join(plots_path,filename+"_overlay_1.png"))#
#     ants.plot(moved,overlay=fixed_n4_denoise,overlay_cmap='hot',overlay_alpha=0.5,axis=0,cbar=True,filename=os.path.join(plots_path,filename+"_overlay_0.png"))#

#     # SAVE nii
#     print(f"\t-> Saving niis.")
#     ants.image_write(moved,os.path.join(moved_path,filename+".nii"))
#     ants.image_write(fixed_n4,os.path.join(n4_path,filename+".nii"))
#     ants.image_write(fixed_n4_denoise,os.path.join(n4_denoise_path,filename+".nii"))
# def process(item):
#     input_file,target_file = item
#     ants_reg(input_file,target_file)
#     # log.info("item: %s" % item)
#     time.sleep(5)

# pool = ThreadPool()
# pool.map(process, zip(input_files,target_files))
# pool.close()
# pool.join()
```



utils.py

```python
import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import pearsonr

# Save nii to reg_path
def load_nii(rawdata_path,regdata_path,select_patients_csv):
    df = pd.read_csv(
        filepath_or_buffer=select_patients_csv,
        sep=',',
        error_bad_lines=False,
        na_values='NULL'
    )
    t1_mri_df = df[df["Modality"] == "MR"]
    for index, row in t1_mri_df.iterrows():
        fold_id = str(row["FOLD_ID"]).zfill(3)
        ct_path = os.path.join(rawdata_path, fold_id, "CT")
        mr_path = os.path.join(rawdata_path, fold_id, "MRI",
                                row["SeriesDescription                    "])
        save_nii(ct_path, f"{regdata_path}/{fold_id}_ct.nii")
        save_nii(mr_path, f"{regdata_path}/{fold_id}_t1.nii")


def all_files_under(path, extension='png', append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames

def histogram(img, bins=256):
    h, w = img.shape
    hist = np.zeros(bins)
    for i in range(h):
        for j in range(w):
            a = img.item(i, j)
            hist[a] += 1

    return hist

def cumulative_histogram(hist, bins=256):
    cum_hist = hist.copy()
    for i in range(1, bins):
        cum_hist[i] = cum_hist[i-1] + cum_hist[i]

    return cum_hist

def n4itk(img):
    ori_img = img.copy()
    mr_img = sitk.GetImageFromArray(img)
    mask_img = sitk.OtsuThreshold(mr_img, 0, 1, 200)

    # Convert to sitkFloat32
    mr_img = sitk.Cast(mr_img, sitk.sitkFloat32)
    # N4 bias field correction
    num_fitting_levels = 4
    num_iters = 200
    try:
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([num_iters] * num_fitting_levels)
        cor_img = corrector.Execute(mr_img, mask_img)
        cor_img = sitk.GetArrayFromImage(cor_img)

        cor_img[cor_img<0], cor_img[cor_img>255] = 0, 255
        cor_img = cor_img.astype(np.uint8)
        return ori_img, cor_img  # return origin image and corrected image
    except (RuntimeError, TypeError, NameError):
        print('[*] Catch the RuntimeError!')
        return ori_img, ori_img

def histogram_matching(img, ref, bins=256):
    assert img.shape == ref.shape

    result = img.copy()
    h, w = img.shape
    pixels = h * w

    # histogram
    hist_img = histogram(img)
    hist_ref = histogram(ref)
    # cumulative histogram
    cum_img = cumulative_histogram(hist_img)
    cum_ref = cumulative_histogram(hist_ref)
    # normalization
    prob_img = cum_img / pixels
    prob_ref = cum_ref / pixels

    new_values = np.zeros(bins)
    for a in range(bins):
        j = bins - 1
        while True:
            new_values[a] = j
            j = j - 1

            if j < 0 or prob_img[a] >= prob_ref[j]:
                break

    for i in range(h):
        for j in range(w):
            a = img.item(i, j)
            b = new_values[a]
            result.itemset((i, j), b)

    return result

def get_mask(image, task='m2c'):
    # Bilateral Filtering
    img_blur = cv2.bilateralFilter(image, 5, 75, 75)
    img_blur = image.copy()
    th, img_thr = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_mor = img_thr.copy()

    # For loop closing
    for ksize in range(21, 3, -2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        img_mor = cv2.morphologyEx(img_mor, cv2.MORPH_CLOSE, kernel)

    # Copy the thresholded image.
    im_floodfill = img_mor.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img_mor.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0,0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    img_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    pre_mask = img_mor | img_floodfill_inv

    # Find the biggest contour
    mask = np.zeros((h, w), np.uint8)
    max_pix, max_cnt = 0, None
    contours, hierachy = cv2.findContours(pre_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        num_pix = cv2.contourArea(cnt)
        if num_pix > max_pix:
            max_pix = num_pix
            max_cnt = cnt

    cv2.drawContours(mask, [max_cnt], 0, 255, -1)

    if task.lower() == 'm2c':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=2)

    return mask

def load_data(img_names, is_test=False, size=256):
    mrImgs, ctImgs, maskImgs = [], [], []
    for _, img_name in enumerate(img_names):
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        mrImg, ctImg, maskImg = img[:, :size], img[:, size:2*size], img[:, -size:]

        if not is_test:
            mrImg, ctImg, maskImg = data_augment(mrImg, ctImg, maskImg)

        # maskImg converte to binary image
        maskImg[maskImg < 127.5] = 0.
        maskImg[maskImg >= 127.5] = 1.

        mrImgs.append(transform(mrImg))
        ctImgs.append(transform(ctImg))
        maskImgs.append(maskImg.astype(np.uint8))

    return np.expand_dims(np.asarray(mrImgs), axis=3), np.expand_dims(np.asarray(ctImgs), axis=3), \
           np.expand_dims(np.asarray(maskImgs), axis=3)

def data_augment(mrImg, ctImg, maskImg, size=256, scope=20):
    # Random translation
    jitter = np.random.randint(low=0, high=scope)

    mrImg = cv2.resize(mrImg, dsize=(size+scope, size+scope), interpolation=cv2.INTER_LINEAR)
    ctImg = cv2.resize(ctImg, dsize=(size+scope, size+scope), interpolation=cv2.INTER_LINEAR)
    maskImg = cv2.resize(maskImg, dsize=(size+scope, size+scope), interpolation=cv2.INTER_LINEAR)

    mrImg = mrImg[jitter:jitter+size, jitter:jitter+size]
    ctImg = ctImg[jitter:jitter+size, jitter:jitter+size]
    maskImg = maskImg[jitter:jitter+size, jitter:jitter+size]

    # Random flip
    if np.random.uniform() > 0.5:
        mrImg, ctImg, maskimg = mrImg[:, ::-1], ctImg[:, ::-1], maskImg[:, ::-1]

    return mrImg, ctImg, maskImg

def transform(img):
    return (img - 127.5).astype(np.float32)

def inv_transform(img, max_value=255., min_value=0., is_squeeze=True, dtype=np.uint8):
    if is_squeeze:
        img = np.squeeze(img)           # (N, H, W, 1) to (N, H, W)

    img = np.round(img + 127.5)     # (-127.5~127.5) to (0~255)
    img[img>max_value] = max_value
    img[img<min_value] = min_value

    return img.astype(dtype)

def cal_mae(gts, preds):
    num_data, h, w, _ = gts.shape
    # mae = np.sum(np.abs(preds - gts)) / (num_data * h * w)
    mae = np.mean(np.abs(preds - gts))

    return mae

def cal_me(gts, preds):
    num_data, h, w, _ = gts.shape
    # me = np.sum(preds - gts) / (num_data * h * w)
    me = np.mean(preds - gts)

    return me

def cal_mse(gts, preds):
    num_data, h, w, _ = gts.shape
    # mse = np.sum(np.abs(preds - gts)**2) / (num_data * h * w)
    mse = np.mean((np.abs(preds - gts))**2)
    return mse

def cal_pcc(gts, preds):
    pcc, _ = pearsonr(gts.ravel(), preds.ravel())
    return pcc

def save_nii(files_path,output_path):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(files_path)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(files_path, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    sitk.WriteImage(image3D, output_path)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range)

def otsu_crop(data, temp_id, size=256, delay=0, is_save=True,do_n4=False):

    save_folder = os.path.join(os.path.dirname(data), 'preprocessing')
    
    if is_save and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print(f"Saved folder:{save_folder}")

    save_folder2 = os.path.join(os.path.dirname(data), 'post')
    if is_save and not os.path.exists(save_folder2):
        os.makedirs(save_folder2)

    # read all files paths
    filenames = all_files_under(data, extension='png')

    # read template image
    temp_filename = filenames[temp_id]
    ref_img = cv2.imread(temp_filename, cv2.IMREAD_GRAYSCALE)
    ref_img = ref_img[:, -size:].copy()
    if do_n4:
        _, ref_img = n4itk(ref_img)  # N4 bias correction for the reference image

    for idx, filename in enumerate(filenames):
        print('\t=> idx: {}, filename: {}'.format(idx, filename))

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        ct_img = img[:, :size]
        mr_img = img[:, -size:]

        # N4 bias correction
        if do_n4:
            ori_img, cor_img = n4itk(mr_img)
        else:
            ori_img, cor_img = ct_img, mr_img
        # Dynamic histogram matching between two images
        
        #his_mr = histogram_matching(cor_img, ref_img)
        
        # Mask estimation based on Otsu auto-thresholding
        mask = get_mask(mr_img, task='m2c')
        # Masked out
        masked_ct = ct_img & mask
        # masked_mr = his_mr & mask
        masked_mr = mr_img & mask
        #canvas = get_canvas(ori_img, cor_img, his_mr, masked_mr,
        canvas = get_canvas(ori_img, cor_img, mr_img, masked_mr,
                        mask, ct_img, masked_ct, size=size, delay=delay)
        canvas2 = np.hstack((masked_mr, masked_ct, mask))

        if is_save:
            cv2.imwrite(os.path.join(
                save_folder, os.path.basename(filename)), canvas)
            cv2.imwrite(os.path.join(
                save_folder2, os.path.basename(filename)), canvas2)

def get_canvas(ori_mr, cor_mr, his_mr, masked_mr, mask, ori_ct, masked_ct, size=256, delay=0, himgs=2, wimgs=5, margin=5):
    canvas = 255 * np.ones((himgs * size + (himgs-1) * margin,
                            wimgs * size + (wimgs-1) * margin), dtype=np.uint8)
    first_rows = [ori_mr, cor_mr, his_mr, masked_mr, mask]
    second_rows = [
        ori_ct, 255*np.ones(ori_ct.shape), 255*np.ones(ori_ct.shape), masked_ct, mask]
    for idx in range(len(first_rows)):
        canvas[:size, idx*(margin+size):idx*(margin+size) +
               size] = first_rows[idx]
        canvas[-size:, idx*(margin+size):idx*(margin+size) +
               size] = second_rows[idx]
    return canvas
```



