# Numpy 转为OpenCV Image并寻找边界

场景：

CT和肿瘤都用numpy array格式保存，希望通过opencv可视化肿瘤的轮廓并绘制在ct图像上。

```python
def plot_ct_mask(ct,mask):
    """
    Args:
        ct(np array) : 2d,128*128
        mask(np array) : 2d,128*128
    """
    # normalize ct -> [0,1]
    ct = normalize(ct)
    ct_img = np.array(ct*255),dtype=np.uint8) # 必须要用8位
    mask_img = np.array(mask*255,dtype=np.uint8)
    ct_img = cv2.cvtColor(ct_img,cv2.COLOR_GRAY2BGR)
    
    # findContours只能在单通道图片上进行，所以mask不用转BGR
    
    ret,thresh = cv2.threshold(mask_img,0,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findCountours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ct_img,contours,-1,(0,255,0),1)
    cv2.imwrite("output.png",ct_img)
    
```

