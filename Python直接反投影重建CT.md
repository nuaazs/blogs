好像经常用到，保存一下：

```python
import matplotlib.pyplot as plt
import numpy as np
import struct
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio
from cv2 import cv2
FILEPATH = "output.v"
LENGTH = 1024
WIDTH = 1024
HEIGHT = 180

# 读取数据文件
f = open(r"C:\Users\nuaazs\Desktop\total.dat",'rb')
data_raw = struct.unpack('f'*LENGTH*WIDTH*HEIGHT,f.read(4*LENGTH*WIDTH*HEIGHT))
data = np.asarray(data_raw).reshape(HEIGHT,LENGTH,WIDTH)

# 检查数据
'''
test = data[0,:,:]
plt.imshow(test)
plt.colorbar()
plt.show()
'''

#重建中间一层
mid_sin_image = []
for pic_num in range(180):
    mid_sin_image.append(data[pic_num,511,:])

plt.imshow(iradon, cmap='gray')
plt.show()
```

