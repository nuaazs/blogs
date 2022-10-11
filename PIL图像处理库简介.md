# PIL图像处理库简介

## 1. Introduction

PIL(Python Image Libary)

friendly fork for PIL





## 3. Image class

Image类是PIL中的核心类，有多张方式进行初始化，常用方法如下：



### 1) open(filename,mode)

打开一张图像

```python
from PIL import Image
Image.open("xxx.jpg","r")
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=296x299 at 0x7F62BDB5B0F0

im = Image.open("dog.jpg","r")
print(im.size,im.format,im.mode)
# (296, 299) JPEG RGB
```

`Image.open` 返回一个Image对象，该对象有`size` ,`format` ,`mode` 等属性。

size：表示图像的宽度和高度（像素表示）

format：表示图像的格式，常见的包括`JPEG` ，`PNG` 等格式

mode：表示图像模式，定义了像素类型还有图像深度，常见的有`RGB` ,`HSV` 。

一般来说`L` (luminance)表示灰度图像，`RGB` 表示真彩图像，`CMYK` 表示预先压缩的图像。

一旦得到了打开的Image对象之后，就可以使用众多方法，比如使用`im.show()`可以展示上面得到的图像。



### 2） save(filename,format)

保存指定格式的图像

```python
im.save("dog.png","png")
```



### 3) thumbnail(size,resample)

创建缩略图

```python
im.thumbanail((50,50),resample=Image.BICUBIC)
im.show()
```

上面的代码可以创建一个指定大小(size)的缩略图,需要注意的是，thumbnail方法是原地操作，返回值是None。第一个参数是指定的缩略图的大小，第二个是采样的，有`Image.BICUBIC`，`PIL.Image.LANCZOS`，`PIL.Image.BILINEAR`，`PIL.Image.NEAREST`这四种采样方法。默认是`Image.BICUBIC`。



### 4) crop(box)

裁剪矩形区域

```python
im = Image.open("dog.jpg","r")
box = (100,100,200,200)
region = im.crop(box)
region.show()
im.crop()
```

上面的代码在im图像上裁剪了一个box矩形区域，然后显示出来。box是一个有四个数字的元组(upper_left_x,upper_left_y,lower_right_x,lower_right_y),分别表示裁剪矩形区域的左上角x,y坐标,右下角的x,y坐标,规定图像的最左上角的坐标为原点(0,0),宽度的方向为x轴，高度的方向为y轴，每一个像素代表一个坐标单位。crop()返回的仍然是一个Image对象。



### 5) transpose(method)

图像翻转或者旋转

```python
im_rotate_180 = im.transpose(Image,ROTATE_180)
im_rotate_180.show()
```

上面的代码将im逆时针旋转180°，然后显示出来,`method`是transpose的参数，表示选择什么样的翻转或者旋转方式，可以选择的值有:
  \- `Image.FLIP_LEFT_RIGHT` ,表示将图像左右翻转
  \- `Image.FLIP_TOP_BOTTOM` ,表示将图像上下翻转
  \- `Image.ROTATE_90` ,表示将图像逆时针旋转90°
  \- `Image.ROTATE_180` ,表示将图像逆时针旋转180°
  \- `Image.ROTATE_270` ,表示将图像逆时针旋转270°
  \-`Image.TRANSPOSE` ,表示将图像进行转置(相当于顺时针旋转90°)
  \- `Image.TRANSVERSE` ,表示将图像进行转置,**再水平翻转**



### 6) paste(region,box,mask)

将一个图像粘贴到另一个图像

```python
im.paste(region,(100,100),None)
im.show()
```

上面的代码将region图像粘贴到左上角为(100,100)的位置。region是要粘贴的Image对象,box是要粘贴的位置，可以是一个两个元素的元组，表示粘贴区域的左上角坐标,也可以是一个四个元素的元组，表示左上角和右下角的坐标。如果是四个元素元组的话,box的size必须要和region的size保持一致，否则将会被convert成和region一样的size。



### 7）split()

颜色通道分离

```python
r,g,b = im.split()
r.show()
g.show()
b.show()
```

split()方法可以原来图像的各个通道分离,比如对于RGB图像，可以将其R,G,B三个颜色通道分离。

### 8) merge(mode,channels)

颜色通道合并.

```python
im_merge = Image,merge("RGB",[b,r,g])
im_merge.show()
```

merge方法和split方法是相对的，其将多个单一通道的序列合并起来，组成一个多通道的图像，mode是合并之后图像的模式，比如”RGB”,channels是多个单一通道组成的序列。



### 9) resize(size,resample,box)

```python
im_resize = im.resize((200,200))
im_resize
#<PIL.Image.Image image mode=RGB size=200x200 at 0x7F62B9E23470>
im_resize.show()
im_resize_box = im.resize((100,100),box = (0,0,50,50))
im_resize_box.show()

```



resize方法可以将原始的图像转换大小,size是转换之后的大小,`resample` 是重新采样使用的方法，仍然有`Image.BICUBIC`，`PIL.Image.LANCZOS`，`PIL.Image.BILINEAR`，`PIL.Image.NEAREST`这四种采样方法，默认是`PIL.Image.NEAREST`,box是指定的要resize的图像区域，是一个用四个元组指定的区域(含义和上面所述box一致)。



### 10) convert(mode,matrix,dither,palette,colors)

mode 转换

```python
im_L = im.convert("L")
im_L.show()
im_rgb = im_l.convert("RGB")
im_rgb.show()
im_L.mode
# 'L'
im_rgb.mode
# 'RGB'
```

convert方法可以改变图像的mode,一般是在’RGB’(真彩图)、’L’(灰度图)、’CMYK’(压缩图)之间转换。上面的代码就是首先将图像转化为灰度图，再从灰度图转化为真彩图。值得注意的是,从灰度图转换为真彩图，虽然理论上确实转换成功了，但是实际上是很难恢复成原来的真彩模式的(不唯一)。



### 11) filter(filter)

应用过滤器

```python
im = Image.open("dog.jpg","r")
from PIL import ImageFilter
im_blur = im.filter(ImageFilter.BLUR)
im_blur.show()
im_find_edges = im.filter(ImageFilter.FIND_EDGES)
im_find_edges.show()
im_find_edges.save("find_edges.jpg")
im_blur.save("blur.jpg")
```

filter方法可以将一些过滤器操作应用于原始图像，比如模糊操作，查找边、角点操作等。filter是过滤器函数，在`PIL.ImageFilter`函数中定义了大量内置的filter函数，比如`BLUR`(模糊操作),`GaussianBlur`(高斯模糊),`MedianFilter`(中值过滤器)，`FIND_EDGES`(查找边)等。



### 12) point(lut,mode)

对图像像素操作

```python
im_point = im.point(lambda x:x*1.5)
im_point.show()
im_point.save("im_point.jpg")
```

point方法可以对图像进行单个像素的操作，上面的代码对point方法传入了一个匿名函数,表示将图像的每个像素点大小都乘以1.5,mode是返回的图像的模式。





下面是一个结合了`point`函数,`split`函数,`paste`函数以及`merge`函数的小例子。

```python
source = im.split()
R,G,B = 0,1,2
mask = source[R].point(lambda x: x<100 and 255) 
# x<100,return 255,otherwise return 0
out_G = source[G].point(lambda x:x*0.7)
>>> # 将out_G粘贴回来，但是只保留'R'通道像素值<100的部分
source[G].paste(out_G,None,mask)
# 合并成新的图像
im_new = Image.merge(im.mode,source)
im_new.show()
im.show()
```



### 13) ImageEnhance()

图像增强

```python
from PIL import ImageEnhance
brightness = ImageEnhanBce.Brightness(im)
im_brightness = brightness.enhance(1.5)
im_brightness.show()
im_contrast = ImageEnhance.Contrast(im)
im_contrast.enhance(1.5)
# <PIL.Image.Image image mode=RGB size=296x299 at 0x7F62AE271AC8>
im_contrast.enhance(1.5).show()
```

`ImageEnhance` 是PIL下的一个子类，主要用于图像增强，比如增加亮度(Brightness),增加对比度(Contrast)等。上面的代码将原来图像的亮度增加50%,将对比度也增加了50%。



### 14) ImageSequence()

处理图像序列

下面的代码可以遍历gif图像中的所有帧，并分别保存为图像

```python
from PIL import ImageSequence
from PIL import Image
gif = Image.open("xx.gif")
for i,frame in enumerate(ImageSequence,Iterator(gif),1):
    if frame.mode == 'JPEG';
    	frame.save("%d.jpg" % i)
    else:
        frame.save("%d.png" % i)
```

除了上面使用迭代器的方式以外，还可以一帧一帧读取gif,比如下面的代码:

```python
index = 0
while 1:
	try:
		gif.seek(index)
		gif.save("%d.%s" %(index,'jpg' if gif.mode == 'JPEG' else 'png'))
		index += 1
	except EOFError:
		print("Reach the end of gif sequence!")
	break
```



上面的代码在读取到gif的最后一帧之后，会throw 一个 EOFError,所以我们只要捕获这个异常就可以了。