## 背景

病人体模剂量计算算完剂量分布后需要绘制DVH图像，其他器官的已经手动勾画完成了，但是皮肤难以勾画。想利用opencv直接寻找图像轮廓完成皮肤部位的像素提取。



## 实例代码

- cv2.inRange在图像中查找形状；
- cv2.findContours寻找轮廓；

### 步骤

观察发现所有的形状颜色基本都是黑色，**定义颜色边界，黑色~深灰色**

1. 加载图像
2. 提取出图像中介于白色和深灰色的像素
3. 提取轮廓
4. 遍历轮廓并展示

```python
# USAGE
# python find_shapes.py --image shapes.png

# 导入必要的包
import argparse

import cv2
import imutils
import numpy as np

# 构建命令行参数及解析
# --image：图像的路径
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())

# 加载图像
image = cv2.imread(args["image"])
cv2.imshow("origin Image", image)
cv2.waitKey(0)

# 目标：检测图像中的黑色形状 使用cv2.inRange实际上检测这些黑色形状非常容易
# 检测图像中的所有黑色形状
# OpenCV以BGR格式存储图像，定义图像颜色的上下边界
lower = np.array([0, 0, 0])  # 纯黑色
upper = np.array([15, 15, 15])  # 包含非常深的灰色阴影
shapeMask = cv2.inRange(image, lower, upper)

# 原始图像中的所有黑色形状现在在黑色背景上均为白色。
# 检测shapeMask中的轮廓
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)  # 兼容不同版本的openCV结果
print("I found {} black shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)

# 遍历检测到的轮廓
for c in cnts:
    # 画出轮廓并展示 绿色
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

```



## 实际操作

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/2hC3C.jpg)

```python
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
ROOT = "./"
SAVE_PATH = "./"


# 均值迁移去噪声+二值化
def threshold_demo(image):
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #cv.imshow("mask", binary)
    return binary


for filename in sorted([file for file in os.listdir(ROOT) if "png" in file]):
    print(f"Filename:{filename}")
    filepath = os.path.join(ROOT,filename)

    # 读取图像
    src1 = cv.imread(filepath)
    src2 = cv.imread(filepath)

    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src1)
    # 调用方法实现二值化
    binary = threshold_demo(src2)

    # 轮廓发现
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)# cv.CHAIN_APPROX_SIMPLE

    temp_array = np.zeros((256,256))
    
    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        if area < 3600:
            continue
        print(area)
        cv.drawContours(src1, contours, c, (0, 0, 255), 2, 8)
        for onedot in contours[c]:
            x = onedot[0,0]
            y = onedot[0,1]
            temp_array[y,x]=1
    # 显示
    cv.imshow("contours-demo", src1)
    cv.waitKey(1000)
    cv.destroyAllWindows()
    np.save(os.path.join(SAVE_PATH,filename.split(".")[-2]),temp_array)
#     plt.imshow(temp_array,cmap="gray")
#     plt.colorbar()
#     plt.show()
```

