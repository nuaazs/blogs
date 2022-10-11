## 前言

数据处理： [第一篇](https://iint.icu/mri-ct%e5%b7%a5%e4%bd%9c%e7%9a%84%e6%95%b0%e6%8d%ae%e9%a2%84%e5%a4%84%e7%90%86/)记录 和 [第二篇](https://iint.icu/mri-ct%e5%b7%a5%e4%bd%9c%e7%9a%84%e6%95%b0%e6%8d%ae%e5%a4%84%e7%90%8602/)记录



## 一、将CT值划分成25个不同的组织，并生成mat数组文件

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

def array_to_mat(ct_array,mat_file_path,backbone,name="fake.mat"):
    os.makedirs(mat_file_path,exist_ok=True)
    if "fake" in name:
        np.save(os.path.join(mat_file_path,backbone+"_fake.npy"),ct_array)
    if "real" in name:
        np.save(os.path.join(mat_file_path,backbone+"_real.npy"),ct_array)
        
        
    hu_list = [-999999,-950,-120,-88,-53,-23,7,18,80,120,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,99999999]

    for i in range(ct_array.shape[0]):
        for j in range(ct_array.shape[1]):
            for k in range(ct_array.shape[2]):
                
                if ct_array[i,j,k]<-950:
                    ct_array[i,j,k]=1
                elif ct_array[i,j,k]<-120:
                    ct_array[i,j,k]=2
                elif ct_array[i,j,k]<-88:
                    ct_array[i,j,k]=3
                elif ct_array[i,j,k]<-53:
                    ct_array[i,j,k]=4
                elif ct_array[i,j,k]<-23:
                    ct_array[i,j,k]=5
                elif ct_array[i,j,k]<7:
                    ct_array[i,j,k]=6
                elif ct_array[i,j,k]<18:
                    ct_array[i,j,k]=7
                elif ct_array[i,j,k]<80:
                    ct_array[i,j,k]=8
                elif ct_array[i,j,k]<120:
                    ct_array[i,j,k]=9
                elif ct_array[i,j,k]<200:
                    ct_array[i,j,k]=10
                elif ct_array[i,j,k]<300:
                    ct_array[i,j,k]=11
                elif ct_array[i,j,k]<400:
                    ct_array[i,j,k]=12
                elif ct_array[i,j,k]<500:
                    ct_array[i,j,k]=13
                elif ct_array[i,j,k]<600:
                    ct_array[i,j,k]=14
                elif ct_array[i,j,k]<700:
                    ct_array[i,j,k]=15
                elif ct_array[i,j,k]<800:
                    ct_array[i,j,k]=16
                elif ct_array[i,j,k]<900:
                    ct_array[i,j,k]=17
                elif ct_array[i,j,k]<1000:
                    ct_array[i,j,k]=18
                elif ct_array[i,j,k]<1100:
                    ct_array[i,j,k]=19
                elif ct_array[i,j,k]<1200:
                    ct_array[i,j,k]=20
                elif ct_array[i,j,k]<1300:
                    ct_array[i,j,k]=21
                elif ct_array[i,j,k]<1400:
                    ct_array[i,j,k]=22
                elif ct_array[i,j,k]<1500:
                    ct_array[i,j,k]=23
                elif ct_array[i,j,k]<1600:
                    ct_array[i,j,k]=24
                else:
                    ct_array[i,j,k]=25
    io.savemat(os.path.join(mat_file_path,name), {'name': ct_array})
    print("to mat done!")
    return ct_array

import numpy  as np
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse
    
def mae(img1, img2):
    mae = np.mean( abs(img1 - img2)  )
    return mae    
def ssim(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim/denom

def get_mask(mask_root,filename,size=256):
    
    nii_img = nb.load(os.path.join(mask_root,filename))
    nii_data = nii_img.get_data()
    nii_data_array =  np.array(nii_data)
    nii_reshape= []
    for slice_num in range(nii_data.shape[2]):
        pre_array = nii_data[:,:,slice_num]
        img = Image.fromarray(pre_array)
        nii_reshape.append(np.array(img.resize((size,size), Image.ANTIALIAS)))
    nii_reshape = np.array(nii_reshape)
    nii_reshape = nii_reshape.transpose((1,2,0))
    return nii_reshape
```

```python
p_id_list = ["002","010","016"]
dataset_name = "cbamunet_gengandweb_20211008"   #"unet_gengandweb_20210929_002"  #"cbamunetgan_gengandweb_20211004"   #"cbamunet_gengandweb_20210929_002" # "pix2pix_gengandweb_20211004"
pth_name =    "generator_best_cbam_unet"     #"generator_unet_100"  #"generator_best_cbam_unet" #"generator_best_cbam_unet"  # "generator_pix2pix_100"
backbone =   "cbamunet"      #"unet"  #"cbamunetgan" #"cbam_unet" #"pix2pix"  

for p_id in p_id_list:

    print(f"Now:{p_id}")

    test_root = "/home/zhaosheng/paper2/test_output/"+dataset_name+"/"+pth_name+"/"
    
    data_path = "/home/zhaosheng/paper2/data/data/test/targets"
    real_nii_path_name= [file for file in (os.listdir("/home/zhaosheng/paper2/test_output/real_data/"+p_id)) if "ct.nii" in file][0]
    real_nii_path = "/home/zhaosheng/paper2/test_output/real_data/"+p_id+"/"+real_nii_path_name
    #print(real_nii_path)

    
    fake_output,real_output= [],[]
    
    temp_list = sorted([file for file in (os.listdir(test_root)) if "geng01_"+p_id in file])
    #print(temp_list)
    for item in temp_list :

        fake = np.load(os.path.join(test_root,item))[0,0,:,:]
        real = np.load(os.path.join(data_path,item))

        img = Image.fromarray(fake)
        img = img.resize((256, 256), Image.ANTIALIAS)
        fake = np.array(img)

        img_real = Image.fromarray(real)
        img_real = img_real.resize((256, 256), Image.ANTIALIAS)
        real = np.array(img_real)

        fake_output.append(fake)
        real_output.append(real)

    fake_output_array = np.array(fake_output)
    #print(fake_output_array.shape)
    
    fake_output_array = fake_output_array.transpose((1,2,0))
    
    real_output_array = np.array(real_output)
    #print(real_output_array.shape)
    real_output_array = real_output_array.transpose((1,2,0))

    #print(fake_output_array.shape)
    assert fake_output_array.shape==(256,256,fake_output_array.shape[2])
    assert real_output_array.shape==fake_output_array.shape


    nii_img = nb.load(real_nii_path)
    nii_data = nii_img.get_data()
    nii_data =  np.array(nii_data)

    assert nii_data.shape[2] == fake_output_array.shape[2]

    for slice in range(nii_data.shape[2]):
        now_max = np.max(new_data[:,:,slice])
        now_min = np.min(new_data[:,:,slice])
        _range = now_max - now_min
        fake_output_array[:,:,slice]  = fake_output_array[:,:,slice]*_range
        fake_output_array[:,:,slice]  = fake_output_array[:,:,slice]+now_min

        real_output_array[:,:,slice]  = real_output_array[:,:,slice]*_range
        real_output_array[:,:,slice]  = real_output_array[:,:,slice]+now_min
        
    fake_output_array[fake_output_array<-1000] = -1000
    fake_output_array[fake_output_array>2000] = 2000

    real_output_array[real_output_array<-1000] = -1000
    real_output_array[real_output_array>2000] = 2000
    

    fake_output_mat = array_to_mat(fake_output_array,"/home/zhaosheng/paper2/test_output/mat/"+p_id+"/",backbone,backbone+"_fake.mat")
    
    real_output_mat = array_to_mat(real_output_array,"/home/zhaosheng/paper2//test_output/mat/"+p_id+"/",backbone,backbone+"_real.mat")
```



## 二、可视化不同网络结果的CT值的分布

```python
backbone_list = ["cbamunet","unet","cbamunetgan","pix2pix"] # "unetgan"

mae_dict = {}
ssim_dict = {}


all_pixel = []
for backbone in backbone_list:
    p_id = 2
    # 002 010 016
    mae_list,ssim_list,name_list = [],[],[]    
    patient = "%03d"%p_id
    real = np.load("/home/zhaosheng/paper2/test_output/mat/"+patient+"/"+backbone+"_real.npy")
    fake = np.load("/home/zhaosheng/paper2/test_output/mat/"+patient+"/"+backbone+"_fake.npy")
    mae_all = mae(real,fake)
    ssim_all = ssim(real,fake)
    print(f"{backbone}网络的平均MAE为：{mae_all}")
    print(f"{backbone}网络的平均SSIM为：{ssim_all}")
    assert real.shape ==(256,256,real.shape[2])
    assert real.shape ==(256,256,real.shape[2])
    mask_root = os.path.join("/home/zhaosheng/paper2/test_output/real_data/",patient,"reg_nii")

    mask_list = {}
    mask_brain = get_mask(mask_root,"mask_Brain.nii.gz")
    mask_list['Brain'] = mask_brain
    mask_brain_stem = get_mask(mask_root,"mask_BrainStem1.nii.gz")
    mask_list['Brain Stem'] = mask_brain_stem

#     mask_eye_l = get_mask(mask_root,"mask_Eye_L.nii.gz")
#     mask_list['Left Eye'] = mask_eye_l

#     mask_eye_r = get_mask(mask_root,"mask_Eye_R.nii.gz")
#     mask_list['Right Eye'] = mask_eye_r

    mask_lens_l = get_mask(mask_root,"mask_Lens_L.nii.gz")
    mask_list['Left Lens'] = mask_lens_l

    mask_lens_r = get_mask(mask_root,"mask_Lens_R.nii.gz")
    mask_list['Right Lens'] = mask_lens_r

#     mask_optic_nerve_l = get_mask(mask_root,"mask_OpticNerve_L.nii.gz")
#     mask_list['Left Optic Nerve'] = mask_optic_nerve_l

    mask_ctv = get_mask(mask_root,"mask_CTV2017.nii.gz")
    mask_list['CTV'] = mask_ctv

#     mask_gtv = get_mask(mask_root,"mask_GTV2021.nii.gz")
#     mask_list['GTV'] = mask_gtv

    mask_inner_ear_l = get_mask(mask_root,"mask_InnerEar_L.nii.gz")
    mask_list['Inner Ear'] = mask_inner_ear_l

#     mask_inner_ear_r = get_mask(mask_root,"mask_InnerEar_R.nii.gz")
#     mask_list['Right Inner Ear'] = mask_inner_ear_r

#     mask_pgtv = get_mask(mask_root,"mask_PGTV2021.nii.gz")
#     mask_list['pGTV'] = mask_pgtv

#     mask_TMJ = get_mask(mask_root,"mask_TMJ_L.nii.gz")
#     mask_list['TMJ'] = mask_TMJ

#     print(mask_list.keys())
#     print(nii_reshape.shape)


    for organ in mask_list.keys():
        mask = mask_list[organ]

    #     for i in range(fake.shape[2]):
    #         if i % 5 == 0:
    #             plt.figure(dpi=150)
    #             plt.imshow(np.hstack([real[:,:,i],fake[:,:,i],mask[:,:,i]]))
    #             plt.colorbar()
    #             plt.show()


        real_organ = real[mask==1]
        fake_organ = fake[mask==1]
        for i in range(len(real_organ)):
            if backbone =="cbamunet":
                all_pixel.append([real_organ[i],organ,"CT"])
            if backbone =="cbamunet":
                net_name = "Attn-Unet"
            if backbone =="cbamunetgan":
                net_name = "Attn-Unet GAN"
            if backbone =="pix2pix":
                net_name = "Pix2Pix"
            if backbone =="unet":
                net_name = "Unet"
            all_pixel.append([fake_organ[i],organ,"sCT "+net_name])
            name_list.append(net_name)

        print("ssim of {}-{}: {} ".format(organ,backbone,ssim(real_organ,fake_organ)))
    #     print("mse of {}: {}".format(organ,mse(real_organ,fake_organ)))
        print("mae of {}-{}: {}".format(organ,backbone,mae(real_organ,fake_organ)))
        mae_list.append(mae(real_organ,fake_organ))
        ssim_list.append(ssim(real_organ,fake_organ))
        

        evl_df = pd.DataFrame(all_pixel,columns=['CT Value(HU)','Organ','Type'])

        
    mae_dict[net_name] = mae_list
    ssim_dict[net_name] = ssim_list
        
plt.figure(dpi=200)
sns.boxplot(x = 'Organ'
            ,y = 'CT Value(HU)'   #x,y交换位置箱线图 横向     
            #,data = evl_df[evl_df['Organ']=="left_eye"]
            ,data = evl_df
            ,hue = 'Type'          # 按照性别分类
            #,palette = 'Purples'  # 设置颜色调色盘
            #,flierprops=flierprops
            ,orient="v"
            ,whis=5
            ,showfliers=False
           ) 
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/Qh3Pq.jpg)



## 三、概率密度分布，小提琴图

```python
backbone_list = ["cbamunet","unet","cbamunetgan","pix2pix"] # "unetgan"


all_pixel = []

for backbone in backbone_list:
    p_id = 2
    # 002 010 016
    
    patient = "%03d"%p_id
    real = np.load("/home/zhaosheng/paper2/test_output/mat/"+patient+"/"+backbone+"_real.npy")
    fake = np.load("/home/zhaosheng/paper2/test_output/mat/"+patient+"/"+backbone+"_fake.npy")
    assert real.shape ==(256,256,real.shape[2])
    assert real.shape ==(256,256,real.shape[2])
    mask_root = os.path.join("/home/zhaosheng/paper2/test_output/real_data/",patient,"reg_nii")

    mask_list = {}


    mask_ctv = get_mask(mask_root,"mask_CTV2017.nii.gz")
    mask_list['CTV'] = mask_ctv




    for organ in mask_list.keys():
        mask = mask_list[organ]

        real_organ = real[mask==1]
        fake_organ = fake[mask==1]
        for i in range(len(real_organ)):
            if backbone =="cbamunet":
                all_pixel.append([real_organ[i],organ,"CT"])
            if backbone =="cbamunet":
                net_name = "Attn-Unet"
            if backbone =="cbamunetgan":
                net_name = "Attn-Unet GAN"
            if backbone =="pix2pix":
                net_name = "Pix2Pix"
            if backbone =="unet":
                net_name = "Unet"
           
        for i in range(len(real_organ)):
            if backbone =="cbamunet":
                all_pixel.append([real_organ[i],organ,"CT"])
            all_pixel.append([fake_organ[i],organ,net_name])

        print("ssim of {}-{}: {} ".format(organ,backbone,ssim(real_organ,fake_organ)))
    #     print("mse of {}: {}".format(organ,mse(real_organ,fake_organ)))
        print("mae of {}-{}: {}".format(organ,backbone,mae(real_organ,fake_organ)))

        evl_df = pd.DataFrame(all_pixel,columns=['CT Value(HU)','Organ','Type'])



plt.figure(dpi=150)
sns.violinplot(x = 'Organ'
            ,y = 'CT Value(HU)'   #x,y交换位置箱线图 横向     
            #,data = evl_df[evl_df['Organ']=="left_eye"]
            ,data = evl_df
            ,hue = 'Type'          # 按照性别分类
            #,palette = 'Purples'  # 设置颜色调色盘
            #,flierprops=flierprops
            ,orient="v"
            ,whis=5
            ,showfliers=False
           ) 
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/snD6R.jpg)



## 四、MAE

```python

import matplotlib.pyplot as plt
import numpy as np




plt.figure(dpi=200)
#name_list
# #mae_list
X = [1,2,3,4]

total_width, n = 0.8, 4
width = total_width / n

#for organ in mask_list.keys():
for backbone in ['Attn-Unet','Attn-Unet GAN', 'Unet',  'Pix2Pix']:
    plt.bar(x=X, #每个柱子的名称
            height=mae_dict[backbone],# year_2019, #每个柱子的高度
            
            #bottom=0, #柱子起始位置对应纵坐标， 默认从0开始
            align='edge',# 柱子名称位置，默认'center',可选'edge'
            #color='pink',
            #edgecolor='b',
            linewidth=1,
            #tick_label=['Brain', 'Brain Stem', 'CTV', 'Inner Ear'], #自定义每个柱子的名称
            #yerr=[10,20,30,40], #添加误差棒
            #ecolor='red',#误差棒颜色，默认黑色
            capsize=5, #误差棒上下的横线长度
            log=False,#y轴坐标取对数
            width=width,
            label=backbone
           )
    for i in range(len(X)):
        X[i] = X[i] + width

plt.xticks(np.array(X)-0.4, ['Brain', 'Brain Stem', 'CTV', 'Inner Ear'],color='blue',rotation=0)#此处locs参数与X值数组相同

plt.ylabel("MAE")
plt.xlabel("Organs")
plt.legend()
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/3tRnt.jpg)



## 五、SSIM

```python

import matplotlib.pyplot as plt
import numpy as np




plt.figure(dpi=200)
#name_list
# #mae_list
X = [1,2,3,4]

total_width, n = 0.8, 4
width = total_width / n

#for organ in mask_list.keys():
for backbone in ['Attn-Unet','Attn-Unet GAN', 'Unet',  'Pix2Pix']:
    plt.bar(x=X, #每个柱子的名称
            height=ssim_dict[backbone],# year_2019, #每个柱子的高度
            
            #bottom=0, #柱子起始位置对应纵坐标， 默认从0开始
            align='edge',# 柱子名称位置，默认'center',可选'edge'
            #color='pink',
            #edgecolor='b',
            linewidth=1,
            #tick_label=['Brain', 'Brain Stem', 'CTV', 'Inner Ear'], #自定义每个柱子的名称
            #yerr=[10,20,30,40], #添加误差棒
            #ecolor='red',#误差棒颜色，默认黑色
            capsize=5, #误差棒上下的横线长度
            log=False,#y轴坐标取对数
            width=width,
            label=backbone
           )
    for i in range(len(X)):
        X[i] = X[i] + width

plt.xticks(np.array(X)-0.4, ['Brain', 'Brain Stem', 'CTV', 'Inner Ear'],color='blue',rotation=0)#此处locs参数与X值数组相同

plt.ylabel("SSIM")
plt.xlabel("Organs")
plt.legend()
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/9nFzG.jpg)



## 六、 侧面图像

```python
import matplotlib.pyplot as plt
import os
import numpy as np

ROOT = r"/home/zhaosheng/paper2/test_output/cbamunetgan_gengandweb_20211008/generator_best_cbam_unet"

file_list = sorted([file for file in os.listdir(ROOT) if "geng01_002" in file])
output = []
for file in file_list:
    array = np.load(os.path.join(ROOT,file))[0,0,:,:]
    output.append(array)
output = np.array(output)



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



def normalization(data):
    _range = np.max(data) - np.min(data)
    
#     print("Patient:{}, y=ax+b".format(id))
#     print("a={}".format(_range))
#     print("b={}".format(np.min(data)))
    return (data - np.min(data)) / (_range)


def get_mask(mask_root,filename,size=256):
    
    nii_img = nb.load(os.path.join(mask_root,filename))
    nii_data = nii_img.get_data()
    nii_data_array =  np.array(nii_data)
    nii_reshape= []
    for slice_num in range(nii_data.shape[2]):
        pre_array = nii_data[:,:,slice_num]
        img = Image.fromarray(pre_array)
        nii_reshape.append(np.array(img.resize((size,size), Image.ANTIALIAS)))
    nii_reshape = np.array(nii_reshape)
    # nii_reshape = nii_reshape.transpose((1,2,0))
    return nii_reshape


REAL_FILE = r"/home/zhaosheng/paper2/test_output/real_data/002/002_ct.nii"


real_ct_array = normalization(get_mask(r"/home/zhaosheng/paper2/test_output/real_data/002/","002_ct.nii"))




```

```python

ax_aspect = 128/24
n = 80
plt.figure(dpi=200)



a1 = plt.subplot(3,3,1)
plt.imshow(output[:,:,n],cmap='gray')
a1.set_aspect(ax_aspect)
plt.axis('off')


a2 = plt.subplot(3,3,2)
plt.imshow(real_ct_array[:,:,n],cmap='gray')
a2.set_aspect(ax_aspect)
plt.axis('off')
# plt.colorbar()



a3 = plt.subplot(3,3,3)
plt.imshow(output[:,:,n]-real_ct_array[:,:,n],cmap='gray')
a3.set_aspect(ax_aspect)
plt.axis('off')
# plt.colorbar()





n = 110

a4 = plt.subplot(3,3,4)
plt.imshow(output[:,:,n],cmap='gray')
a4.set_aspect(ax_aspect)
plt.axis('off')



a5 = plt.subplot(3,3,5)
plt.imshow(real_ct_array[:,:,n],cmap='gray')
a5.set_aspect(ax_aspect)
plt.axis('off')



a6 = plt.subplot(3,3,6)
plt.imshow(output[:,:,n]-real_ct_array[:,:,n],cmap='gray')
a6.set_aspect(ax_aspect)
plt.axis('off')



n = 200

a7 = plt.subplot(3,3,7)
plt.imshow(output[:,:,n],cmap='gray')
a7.set_aspect(ax_aspect)
plt.axis('off')


a8 = plt.subplot(3,3,8)
plt.imshow(real_ct_array[:,:,n],cmap='gray')
a8.set_aspect(ax_aspect)
plt.axis('off')
# plt.colorbar()



a9 = plt.subplot(3,3,9)
plt.imshow((output[:,:,n]-real_ct_array[:,:,n]),cmap='gray')
a9.set_aspect(ax_aspect)
plt.axis('off')
# plt.colorbar()



plt.tight_layout()
plt.show()

```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/U0Wou.jpg)

|                  | Patient1    | Patient2    | Patient3    |
| ---------------- | ----------- | ----------- | ----------- |
| Unet256-GAN      | 66.91847405 | 75.7184181  | 51.22233685 |
| Resnet9block-GAN | 62.97917012 | 74.29159858 | 48.00514584 |
| Attn-GAN         | 59.32602055 | 68.93765585 | 39.55022302 |

## Prerequisites

Tested with:

- Python 3.7
- Pytorch 1.7.1
- CUDA 11.0
- Pytorch Lightning 1.1.8
- waymo_open_dataset 1.3.0

## Preparation

The repository contains training and inference code for CoMo-MUNIT training on waymo open dataset. In the paper, we refer to this experiment as Day2Timelapse. All the models have been trained on a 32GB Tesla V100 GPU. We also provide a mixed precision training which should fit smaller GPUs as well (a usual training takes ~9GB).

### Environment setup

We advise the creation of a new conda environment including all necessary packages. The repository includes a requirements file. Please create and activate the new environment with

```
conda env create -f requirements.yml
conda activate comogan
```

### Dataset preparation

First, download the Waymo Open Dataset from [the official website](https://waymo.com/open/). The dataset is organized in `.tfrecord` files, which we preprocess and split depending on metadata annotations on time of day. Once you downloaded the dataset, you should run the `dump_waymo.py` script. It will read and unpack the `.tfrecord` files, also resizing the images for training. Please run

## Pretrained weights

We release a pretrained set of weights to allow reproducibility of our results. The weights are downloadable from [here](https://www.rocq.inria.fr/rits_files/computer-vision/comogan/logs_pretrained.tar.gz). Once downloaded, unpack the file in the root of the project and test them with the inference notebook.

## Training

The training routine of CoMoGAN is mainly based on the CycleGAN codebase, available with details in the official repository.

To launch a default training, run

```
python train.py --path_data path/to/waymo/training/dir --gpus 0
```

You can choose on which GPUs to train with the `--gpus` flag. Multi-GPU is not deeply tested but it should be managed internally by Pytorch Lightning. Typically, a full training requires 13GB+ of GPU memory unless mixed precision is set. If you have a smaller GPU, please run

```
python train.py --path_data path/to/waymo/training/dir --gpus 0 --mixed_precision
```

Please note that performances on mixed precision trainings are evaluated only qualitatively.
