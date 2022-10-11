## 1.设置colorbar字体大小

```python
plt.rcParams['font.size'] = 13
```

## 2. colorbar设置显示的刻度个数和指定的刻度值
通过matplotlib.ticker.MaxNLocator(nbins=n)来设置colorbar上的刻度值个数
```python
import matplotlib.ticker as ticker
fig = plt.figure()
ax = fig.gca()
im = ax.imshow(np.random.random([10, 10]))
cb1 = plt.colorbar(im, fraction=0.03, pad=0.05)
tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数
cb1.locator = tick_locator
cb1.update_ticks()
plt.show()
```
但这样colorbar上的刻度值不是从最大值和最小值开始和结束，如果想将colorbar上的刻度值从最大值和最小值开始和结束，可以手动设置colorbar上显示的刻度值：
```python
import matplotlib.ticker as ticker
fig = plt.figure()
ax = fig.gca()
data = np.random.random([10, 10])
im = ax.imshow(data)
cb1 = plt.colorbar(im, fraction=0.03, pad=0.05)
tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
cb1.locator = tick_locator
cb1.set_ticks([np.min(data), 0.25,0.5, 0.75, np.max(data)])
cb1.update_ticks()
plt.show()
```
## 3. 在Matplotlib中更改Colorbar的缩放/单位
```python
func = lambda x,pos: "{:g}".format(x*1000)
fmt = matplotlib.ticker.FuncFormatter(func)

plt.colorbar(..., format=fmt)
```



## 4. 实际使用场景

画出Ct矢状面剖面图

```python
COLOR_BAR_FONT_SIZE = 6
MODE = "cor"
n = 100
pat_num = "044"

import matplotlib.ticker as ticker

fake_img = nib.load("./"+pat_num+"_fake.nii")
fake_ct_array = np.asanyarray(fake_img.dataobj)
fake_ct_array = np.array(fake_ct_array)

real_img = nib.load("./"+pat_num+"_real.nii")
real_ct_array = np.asanyarray(real_img.dataobj)
real_ct_array = np.array(real_ct_array)

assert fake_ct_array.shape == real_ct_array.shape
a,b,c = fake_ct_array.shape

if MODE == "sag" or MODE == "cor":
    ax_aspect = b/a
else:
    ax_aspect = 1




func = lambda x,pos: "{:g}HU".format(x)
fmt = ticker.FuncFormatter(func)

# plt.colorbar(..., format=fmt)



plt.figure(dpi=400)

a1 = plt.subplot(1,3,1)

if MODE == "sag":
    fake_array = fake_ct_array[:,:,n]
    real_array = real_ct_array[:,:,n]
elif MODE == "cor":
    fake_array = fake_ct_array[:,n,:]
    real_array = real_ct_array[:,n,:]
else:
    fake_array = fake_ct_array[n,:,:]
    real_array = real_ct_array[n,:,:]
    
im =plt.imshow(fake_array,cmap='gray')
a1.set_aspect(ax_aspect)
plt.axis('off')
plt.rcParams['font.size'] = COLOR_BAR_FONT_SIZE
#plt.colorbar()
cb1 = plt.colorbar(im, fraction=0.03, pad=0.05, format=fmt)
tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数
cb1.locator = tick_locator
cb1.update_ticks()


a2 = plt.subplot(1,3,2)
im2 =plt.imshow(real_array,cmap='gray')
a2.set_aspect(ax_aspect)
plt.axis('off')
plt.rcParams['font.size'] = COLOR_BAR_FONT_SIZE
cb1 = plt.colorbar(im2, fraction=0.03, pad=0.05, format=fmt)
tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数
cb1.locator = tick_locator
cb1.update_ticks()


a3 = plt.subplot(1,3,3)
im3 =plt.imshow(fake_array-real_array,cmap='gray')
a3.set_aspect(ax_aspect)
plt.axis('off')
plt.rcParams['font.size'] = COLOR_BAR_FONT_SIZE
cb1 = plt.colorbar(im3, fraction=0.03, pad=0.05, format=fmt)
tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数
cb1.locator = tick_locator
cb1.update_ticks()



plt.tight_layout()
plt.savefig("./"+pat_num+"_"+MODE+".jpg")
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/PyqGG.jpg)

