## 无透明度

```python
import PIL.Image as Image
layer1 = Image.open("image.jpg").convert('RGBA') # 底图背景
layer2 = Image.open("mask.png").convert('RGBA')  # mask

# layer1和layer2要相同大小，否则需要resize
final = Image.new("RGBA",layer1.size)
final = Image.alpha_composite(final,layer1)
final = Image.alpha_composite(final,layer2)

final = final.convert('RGB')
final.save('image_mask.jpg')
```

亲测效果：
alpha_composite效果最好，边缘平滑，paste也可，blend周围会有毛躁和失真



## 添加透明度

尝试了blend和网上其他方法，透明图片叠加透明度后颜色会变，并且边缘毛躁，后来发现调整alpha通道亮度就不会失真。

```python
# 透明度添加
# reference：https://www.qedev.com/python/37695.html
import PIL.Image as Image,ImageEnhance
layer1 = Image.open("image.jpg").convert('RGBA')   # 底图背景
layer2 = Image.open("mask.png").convert('RGBA')    # mask

r,g,b,a = layer2.split()
# opacity为透明度，范围(0,1)
opacity = 0.4
alpha = ImageEnhance.Brightness(a).enhance(opacity)
layer2.putalpha(alpha)

# 使用alpha_composite叠加，两者需要相同size
final = Image.new("RGBA",layer1.size)
final = Image.alpha_composite(final,img)
final = Image.alpha_composite(final,layer2)

# 使用paste叠加，无需相同大小，可调整box位置
layer = Image.new('RGBA',layer1.size,(0,0,0,0))
layer.paste(layer2,(100,100))
Image.composite(layer,layer1,layer).convert('RGB')
```

不好用的方法

```python
# 方法1：
# factor添加透明度，颜色失真,边缘处理毛躁
img = Image.blend(layer1, layer2, factor)

# 方法2：
# 设置水印图片透明度，颜色失真
rand = random.randint(100,255)
layer2= np.array(layer2)[:-1] + (rand, )
layer2= Image.fromarray(np.uint8(layer2))
# 再使用alpha_composite叠加图片
final = Image.new("RGBA", layer1.size)
final = Image.alpha_composite(final, layer1)
final = Image.alpha_composite(final, layer2)
```

