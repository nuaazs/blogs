```python
import os
file_path = "./"
def get_one_hot(input_array):
    ct_array = input_array.copy()
    hu_list = [-999999,-950,-120,-88,-53,-23,7,18,80,120,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,99999999]
    a = ct_array.shape[0]
    b = ct_array.shape[1]
    for i in range(a):
        for j in range(b):
            if ct_array[i,j]<-950:
                ct_array[i,j]=1
            elif ct_array[i,j]<-120:
                ct_array[i,j]=2
            elif ct_array[i,j]<-88:
                ct_array[i,j]=3
            elif ct_array[i,j]<-53:
                ct_array[i,j]=4
            elif ct_array[i,j]<-23:
                ct_array[i,j]=5
            elif ct_array[i,j]<7:
                ct_array[i,j]=6
            elif ct_array[i,j]<18:
                ct_array[i,j]=7
            elif ct_array[i,j]<80:
                ct_array[i,j]=8
            elif ct_array[i,j]<120:
                ct_array[i,j]=9
            elif ct_array[i,j]<200:
                ct_array[i,j]=10
            elif ct_array[i,j]<300:
                ct_array[i,j]=11
            elif ct_array[i,j]<400:
                ct_array[i,j]=12
            elif ct_array[i,j]<500:
                ct_array[i,j]=13
            elif ct_array[i,j]<600:
                ct_array[i,j]=14
            elif ct_array[i,j]<700:
                ct_array[i,j]=15
            elif ct_array[i,j]<800:
                ct_array[i,j]=16
            elif ct_array[i,j]<900:
                ct_array[i,j]=17
            elif ct_array[i,j]<1000:
                ct_array[i,j]=18
            elif ct_array[i,j]<1100:
                ct_array[i,j]=19
            elif ct_array[i,j]<1200:
                ct_array[i,j]=20
            elif ct_array[i,j]<1300:
                ct_array[i,j]=21
            elif ct_array[i,j]<1400:
                ct_array[i,j]=22
            elif ct_array[i,j]<1500:
                ct_array[i,j]=23
            elif ct_array[i,j]<1600:
                ct_array[i,j]=24
            else:
                ct_array[i,j]=25

    ct_array = ct_array.flatten()
    num_classes = 25
    num_labels = ct_array.shape[0]
    z = np.zeros([num_labels, num_classes])
    for i in range(num_labels):  # 4
        j = int(ct_array[i])
        z[i][j-1] = 1
    output= z.reshape(a,b,num_classes)
    return output

    

    
for png in sorted([os.path.join(file_path,filename) for filename in os.listdir(file_path) if ".png" in filename]):
    ... (省略获取numpy的代码)
```





CT图片归一化的时候，取最大值为1700，最小值为-1000：

```python
target_numpy[target_numpy>1700] = 1700
target_numpy[target_numpy<-1000] = -1000
```

