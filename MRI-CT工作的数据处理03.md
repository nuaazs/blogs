## 前言

之前 [第一篇](https://iint.icu/mri-ct%e5%b7%a5%e4%bd%9c%e7%9a%84%e6%95%b0%e6%8d%ae%e9%a2%84%e5%a4%84%e7%90%86/)记录 和 [第二篇](https://iint.icu/mri-ct%e5%b7%a5%e4%bd%9c%e7%9a%84%e6%95%b0%e6%8d%ae%e5%a4%84%e7%90%8602/)记录 中处理完的数据，训练完成之后发现效果并不好，可能是由于配准不够好导致的。这篇文章在总结一下数据清洗的思路。



## 一、将之前的数据集都保存为png图片

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/NrqqI.jpg)

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

INPUT_ROOT = "/home/zhaosheng/paper2/data/data/inputs"
inputs_files = sorted( [os.path.join(INPUT_ROOT,file) for file in os.listdir(INPUT_ROOT) if "npy" in file])

TARGET_ROOT = "/home/zhaosheng/paper2/data/data/targets"
targets_files = sorted( [os.path.join(TARGET_ROOT,file) for file in os.listdir(TARGET_ROOT) if "npy" in file])
files = zip(inputs_files,targets_files)

for file in files:
    print(file)
    _input = np.load(file[0])
    _target = np.load(file[1])
    filename = file[1].split("/")[-1].split(".")[-2]
    array = np.hstack([_input,_target])
    image = Image.fromarray(array*255)
    if image.mode == "F":
        image = image.convert('RGB')
    image.save("./png/"+filename+".png")  
#     plt.figure(dpi=200)
#     plt.imshow(array)
#     plt.colorbar()
#     plt.show()
```



## 二、手动筛选、放大较小的图片



## 三、生成mat文件

```python

def nii_to_mat(nii_file_path,mat_file_path):
    img = nib.load(nii_file_path)

    ct_array = np.asanyarray(img.dataobj)
    ct_array = np.array(ct_array)
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
                    
#                 for num in range(len(hu_list)):
#                     if (hu_list[num]<=ct_array[i,j,k]<hu_list[num+1]):
#                         ct_array[i,j,k]=num+1
    io.savemat(mat_file_path, {'name': ct_array})
    #print("{}->{} Done!".format((nii_file_path.split("/")[-1],mat_file_path.split("/")[-1])))
    print("to mat done!")
# 016 -> p 109    体素数量(512, 512, 41)   体素大小(0.40434101, 0.40434101, 3.)
# 017 -> t 001
```

