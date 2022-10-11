matplotlib

## 1. 图形结构

### 1.1 figure层

指整张图，可设置整张图的分辨率（dpi），长宽（figsize）、标题（title）等特征；

可包含多个axes，可简单理解为多个子图（下图为两个axes）； 

figure置于canvas系统层之上，用户不可见。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/TN2dK.jpg)

### 1.2 axes层

每个子图，可以绘制各种图形，例如柱状图（bar），饼图（pie函数），箱图（boxplot）等；

设置每个图的外观网格线（grid）的开关、坐标轴（axis）开关等；

设置每个坐标轴（axis）的名字（label）、子图标题（title）、图例（legend）等；

设置坐标轴范围（scale）、坐标轴刻度（tricks）等；

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/WZWcF.jpg)





## 2. **matplotlib两种画绘图方法**

### 2.1 使用matplotlib.pyplot

这种绘图主要使用pyplot模块，pyplot.py代码量有3000多行（windows下存储于xxx\site-packages\matplotlib\pyplot.py），该脚本里面有大量def定义的函数，绘图时就是调用pyplot.py中的函数。

```python
# matplotlib.pyplot 接口
import numpy as np


import matplotlib.pyplot as plt  # 导入pyplot，matplotlib.pyplot简写为plt


def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(dpi=100)

plt.subplot(211)
plt.plot(t1, f(t1), color="tab:blue", marker="o")
plt.plot(t2, f(t2), color="black")
plt.title("demo")


plt.subplot(212)
plt.plot(t2, np.cos(2 * np.pi * t2), color="tab:orange", linestyle="--")
plt.suptitle("matplotlib.pyplot api")
plt.show()

```



### 2.2 面向对象方法

画比较复杂的图形时，面向对象方法会更方便。这种绘图方式主要使用matplotlib的两个子类：matplotlib.figure.Figure和matplotlib.axes.Axes，画每张图时，画布为matplotlib.figure.Figure的一个实例，每个子图为matplotlib.axes.Axes的一个实例，分别可以继承父类的所有方法，也就是说你绘图时，你想设置的元素（网格线啊，坐标刻度啊等）都可以在二者的属性中找出来使用。

#### matplotlib.figure.Figure

#### matplotlib.axes.Axes

```python
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

fig, axs = plt.subplots(2, dpi=100)
#fig为matplotlib.figure.Figure对象的实例figure
#axs为matplotlib.axes.Axes对象实例（每个子图）组成的numpy.ndarray
axs[0].plot(t1, f(t1), color='tab:blue', marker='o')
axs[0].plot(t2, f(t2), color='black')

#两种设置标题的方法
#axs[0].set_title('haha')#使用matplotlib.axes.Axes的set_title方法设置小标题
axs[0].set(title='demo1')

axs[1].plot(t2, np.cos(2*np.pi*t2), color='tab:orange', linestyle='--')
fig.suptitle('matplotlib object-oriented')#使用matplotlib.figure.Figure中的suptitle方法设置Figure标题
plt.show()
```



## 3. 坐标轴|刻度值|刻度|标题设置

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] =['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(dpi=150)

#整张图figure的标题自定义设置
plt.suptitle('整张图figure的标题：suptitle',#标题名称
             x=0.5,#x轴方向位置
             y=0.98,#y轴方向位置
             size=15, #大小
             ha='center', #水平位置，相对于x,y，可选参数：{'center', 'left', right'}, default: 'center'
             va='top',#垂直位置，相对不x,y，可选参数：{'top', 'center', 'bottom', 'baseline'}, default: 'top'
             weight='bold',#字体粗细，以下参数可选
            # '''weight{a numeric value in range 0-1000, 'ultralight', 'light', 
             #'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 
             #'demi', 'bold', 'heavy', 'extra bold', 'black'}'''
             
             #其它可继承自matplotlib.text的属性
             #标题也是一种text，故可使用text的属性，所以这里只是展现了冰山一角
             rotation=1,##标题旋转，传入旋转度数，也可以传入vertical', 'horizontal'
            )


plt.subplot(1,1,1)#绘制一个子图


#设置文本属性字典
font_self = {'family':'Microsoft YaHei',#设置字体
             'fontsize': 10,#标题大小
             'fontweight' : 'bold',#标题粗细，默认plt.rcParams['axes.titleweight']
             'color' : (.228,.21,.28),
             #'verticalalignment': 'baseline',
            # 'horizontalalignment': 'right'
            }


#每个子图标题自定义设置
plt.title('每个子图axes的标题：title', 
          fontdict=font_self,
          loc='left',#{'center', 'left', 'right'} 
          #下面两个参数可以在前面字典中设置，也可以在这设置；存在时，loc指title在整个figure的位置，例如上面的left指与figure的最左边对齐，而不是与axes最左边对齐
          #ha='center',#会影响loc的使用，可选参数：{'center', 'left', right'}, default: 'center'
          #va='center'#会影响loc的使用，可选参数：{'top', 'center', 'bottom', 'baseline'}, default: 'top'
          pad=7,#子图标题与上坐标轴的距离，默认为6.0
          
          #其它可继承自matplotlib.text的属性
          rotation=360,#标题旋转，传入旋转度数，也可以传入vertical', 'horizontal'
         
         )




#坐标轴的开启与关闭操作
plt.gca().spines['top'].set_visible(False)#关闭上坐标轴
plt.gca().spines['bottom'].set_visible(True)#开启x轴坐标轴
plt.gca().spines['left'].set_visible(True)#开启y轴坐标轴
plt.gca().spines['right'].set_visible(False)#关闭右轴
##plt.gca()具有大量属性，也可以对刻度值、刻度、刻度值范围等操作，可自行实验，这里只提到了冰山一角

plt.gca().spines['bottom'].set_color('black')#x轴（spines脊柱）颜色设置
plt.gca().spines['bottom'].set_linewidth(10)#x轴的粗细，下图大黑玩意儿就是这里的杰作
plt.gca().spines['bottom'].set_linestyle('--')#x轴的线性
#同样这里只提到了轴spines属性的冰山一角，也可自行实验

#绘制网格线
plt.grid()


#坐标轴刻度（tick）与刻度值（tick label）操作
plt.tick_params(axis='x',#对那个方向（x方向：上下轴；y方向：左右轴）的坐标轴上的tick操作，可选参数{'x', 'y', 'both'}
                which='both',#对主刻度还是次要刻度操作，可选参数为{'major', 'minor', 'both'}
                colors='r',#刻度颜色
                
                #以下四个参数控制上下左右四个轴的刻度的关闭和开启
                top='on',#上轴开启了刻度值和轴之间的线
                bottom='on',#x轴关闭了刻度值和轴之间的线
                left='on',
                right='on',
                
                direction='out',#tick的方向，可选参数{'in', 'out', 'inout'}                
                length=10,#tick长度
                width=2,#tick的宽度
                pad=10,#tick与刻度值之间的距离
                labelsize=10,#刻度值大小
                labelcolor='#008856',#刻度值的颜色
                zorder=0,
                
                #以下四个参数控制上下左右四个轴的刻度值的关闭和开启
                labeltop='on',#上轴的刻度值也打开了此时
                labelbottom='on',                
                labelleft='on',
                labelright='off',
                
                labelrotation=45,#刻度值与坐标轴旋转一定角度
                
                grid_color='pink',#网格线的颜色，网格线与轴刻度值对应，前提是plt.grid()开启了
                grid_alpha=1,#网格线透明度
                grid_linewidth=10,#网格线宽度
                grid_linestyle='-',#网格线线型，{'-', '--', '-.', ':', '',matplotlib.lines.Line2D中的都可以用              
                
                
               )


#plt.xticks([])#x轴刻度值trick的关闭
plt.xticks(np.arange(0, 2, step=0.2),list('abcdefghigk'),rotation=45)
#自定义刻度标签值，刻度显示为您想要的一切（日期，星期等等）



#设置刻度范围
plt.xlim(0,2)#x坐标轴刻度值范围
plt.ylim(0,2)#y坐标轴刻度值范围
#plt.gca().set((xlim=[0, 2], ylim=[0, 2])#x轴y轴坐标轴范围操作


#设置刻度值之间步长(间隔)
from matplotlib.pyplot import MultipleLocator
plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
#plt.minorticks_off()#是否每个刻度都要显示出来




plt.xlabel('X轴标题',
           labelpad=22,#x轴标题xlabel与坐标轴之间距离
           fontdict=font_self,#设置xlabel的字体、大小、颜色、粗细
           
           #类似于上面，可继承自matplotlib.text的属性
           rotation=90
          
          )
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/etkkt.jpg)

## 4. 掌握marker和linestyle使用

### 4.1 标记marker

matplotlib一般marker位于`matplotlib.lines import Line2D`中，共计30+种，可以输出来康康有哪些：

```python
['.', ',', '1', '2', '3', '4', '+', 'x', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
```

这些marker都长什么样纸了，来康康：

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/jDm4I.jpg)

37种marker，一般绘图可以完全满足了，如果您说不满足，想要更酷炫的，比如说下面这种，

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/bX5oi.jpg)

matplotlib官网上就有，[请戳](https://matplotlib.org/tutorials/text/mathtext.html)。一时冲动爬了下辣个网站，有400+种，长如下这样的：

```python

可以显示的形状    marker名称
ϖ  \varpi
ϱ  \varrho
ς  \varsigma
ϑ  \vartheta
ξ  \xi
ζ  \zeta
Δ  \Delta
Γ  \Gamma
Λ  \Lambda
Ω  \Omega
Φ  \Phi
Π  \Pi
Ψ  \Psi
Σ  \Sigma
Θ  \Theta
Υ  \Upsilon
Ξ  \Xi
℧  \mho
∇  \nabla
ℵ  \aleph
ℶ  \beth
ℸ  \daleth
ℷ  \gimel
/  /
[  [
⇓  \Downarrow
⇑  \Uparrow
‖  \Vert
↓  \downarrow
⟨  \langle
⌈  \lceil
⌊  \lfloor
⌞  \llcorner
⌟  \lrcorner
⟩  \rangle
⌉  \rceil
⌋  \rfloor
⌜  \ulcorner
↑  \uparrow
⌝  \urcorner
\vert
{  \{
\|
}  \}
]  ]
|
⋂  \bigcap
⋃  \bigcup
⨀  \bigodot
⨁  \bigoplus
⨂  \bigotimes
⨄  \biguplus
⋁  \bigvee
⋀  \bigwedge
∐  \coprod
∫  \int
∮  \oint
∏  \prod
∑  \sum
```

#### matplotlib中marker 怎么使用 

非常简单，入门级marker使用时，marker=marker名称；高手级和自定义级marker使用时，marker=\$marker名称\$；

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文
plt.figure(dpi=200)
#常规marker使用
plt.plot([1,2,3],[1,2,3],marker=4, markersize=15, color='lightblue',label='常规marker')
plt.plot([1.8,2.8,3.8],[1,2,3],marker='2', markersize=15, color='#ec2d7a',label='常规marker')

#非常规marker使用
#注意使用两个$符号包围名称
plt.plot([1,2,3],[4,5,6],marker='$\circledR$', markersize=15, color='r', alpha=0.5,label='非常规marker')
plt.plot([1.5,2.5,3.5],[1.25,2.1,6.5],marker='$\heartsuit$', markersize=15, color='#f19790', alpha=0.5,label='非常规marker')
plt.plot([1,2,3],[2.5,6.2,8],marker='$\clubsuit$', markersize=15, color='g', alpha=0.5,label='非常规marker')

#自定义marker
plt.plot([1.2,2.2,3.2],[1,2,3],marker='$666$', markersize=15, color='#2d0c13',label='自定义marker')
plt.legend(loc='upper left')
for i in ['top','right']:
    plt.gca().spines[i].set_visible(False)
```

### 4.2 线型linestyle

线性 （linestyle）可分为字符串型的元组型的：

#### 字符型

```python
linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'；solid’， (0, ()) ， '-'三种都代表实线。
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'
```

#### 元组型

直接修改元组中的数字可以呈现不同的线型，所以有无数种该线型

```python

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 2))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/ldZKc.jpg)

#### 线型使用

 ```python

import matplotlib.pyplot as plt
plt.figure(dpi=120)
#字符型linestyle使用方法
plt.plot([1,2,3],[1,2,13],linestyle='dotted', color='#1661ab', linewidth=5, label='字符型线性：dotted')

#元组型lintstyle使用方法 
plt.plot([0.8,0.9,1.5],[0.8,0.9,21.5],linestyle=(0,(3, 1, 1, 1, 1, 1)), color='#ec2d7a', linewidth=5, label='元组型线性：(0,(3, 1, 1, 1, 1, 1)')

for i in ['top','right']:
    plt.gca().spines[i].set_visible(False)

#自定义inestyle  
plt.plot([1.5,2.5,3.5],[1,2,13],linestyle=(0,(1,2,3,4,2,2)), color='black', linewidth=5, label='自定义线性：(0,(1,2,3,4,2,2)))')
plt.plot([2.5,3.5,4.5],[1,2,13],linestyle=(2,(1,2,3,4,2,2)), color='g', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
plt.legend()
 ```

#### 元组线型详解

第一个0的意义，比较黑色和绿色线性即可知道

1,2 第一小段线宽1磅，第一和第二段之间距离2磅

3,4 第二小段线宽3磅，第二和第三段之间距离4磅

2,2 第三小段线宽2磅，第三和第四段之间距离2磅



## 5. 绘图风格

### 5.1 matplotlib有哪些绘图风格

使用plt.style.available输出所有风格名称，共计26种。

 ```python
import matplotlib.pyplot as plt       
print(plt.style.available)
 ```

```python
['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']
```

每种风格的源码都在路径`xx\Lib\site-packages\matplotlib\mpl-data\stylelib`之下，您可以模仿着自定义一个自己的风格，这很简单。

点开一个ggplot的绘图风格看看 ，里面都是线型，颜色等的设置。

### 5.2 绘图风格使用

```python
plt.style.use('ggplot')#使用ggplot风格。
```



## 6. 绘制散点图scatter

### 6.1 鸢尾花（iris）数据集

数据集导入、查看特征

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn import datasets
iris = datasets.load_iris()
dir(iris)
# ['DESCR', 'data', 'feature_names', 'target', 'target_names']
```

DESCR:为数据集的描述信息，输出来看看：

data:鸢尾花四个特征的数据

```python
print(type(iris.data))  # <class 'numpy.ndarray'>#数据格式为numpy.ndarray
print(iris.data.shape) #  (150, 4)#数据集大小为150行4列
iris.data[:10,:]
"""
array([[5.1, 3.5, 1.4, 0.2],#数据集前十行

       [4.9, 3. , 1.4, 0.2],

       [4.7, 3.2, 1.3, 0.2],

       [4.6, 3.1, 1.5, 0.2],

       [5. , 3.6, 1.4, 0.2],

       [5.4, 3.9, 1.7, 0.4],

       [4.6, 3.4, 1.4, 0.3],

       [5. , 3.4, 1.5, 0.2],

       [4.4, 2.9, 1.4, 0.2],

       [4.9, 3.1, 1.5, 0.1]])


"""
```

feature_names:以上4列数据的名称，从左到右依次为花萼长度、花萼宽度、花瓣长度、花瓣宽度，单位都是cm。

```python
print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```

target:使用数字0. ,1. ,2.标识每行数据代表什么类的鸢尾花。

```python
print(iris.target)#150个元素的list

"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
"""
```

target_names:鸢尾花的名称，Setosa（山鸢尾花）、Versicolour（杂色鸢尾花）、Virginica（维吉尼亚鸢尾花）。

```python
print(iris.target_names)
# ['setosa' 'versicolor' 'virginica']
```

将鸢尾花数据集转为DataFrame数据集

```python
x, y = iris.data,iris.target
pd_iris = pd.DataFrame(np.hstack((x,y.reshape(150,1))),columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'])
#np.hstack()类似linux中的paste
#np.vstack()类似linux中的cat
pd_iris.head()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/1V6eS.jpg)

### 6.2 matplotlib.pyplot.scatter法绘制散点图

```python
# 取数据集前两列绘制简单散点图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import datasets
iris = datasets.load_iris()
x, y = iris.data, iris.target
pd_iris = pd.DataFrame(np.hstack((x, y.reshape(150,1))),columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'])
plt.figure(dpi=100)
plt.scatter(pd_iris['sepal length(cm)'],pd_iris['sepal width(cm)'])
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/6YoNK.jpg)

三种不同鸢尾花的数据使用不同的图形（marker）和颜色表示

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
#数据准备
from sklearn import datasets
iris=datasets.load_iris()
x, y = iris.data, iris.target
pd_iris = pd.DataFrame(np.hstack((x, y.reshape(150,1))),columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'])
 
 
plt.figure(dpi=150)#设置图的分辨率
plt.style.use('dark_background') #使用Solarize_Light2风格绘图
iris_type = pd_iris['class'].unique() # 根据class列将点分为三类
iris_name = iris.target_names # 获取每一类的名称
colors =['#c72e29','#098154','#fb832d']#三种不同颜色
markers =['$\clubsuit$','.','+']#三种不同图形

for i in range(len(iris_type)):
    plt.scatter(pd_iris.loc[pd_iris['class']== iris_type[i],'sepal length(cm)'],#传入数据x
                pd_iris.loc[pd_iris['class']== iris_type[i],'sepal width(cm)'],#传入数据y
                s =50,#散点图形（marker）的大小
                c = colors[i],#marker颜色
                marker = markers[i],#marker形状
                #marker=matplotlib.markers.MarkerStyle(marker = markers[i],fillstyle='full'),#设置marker的填充
                alpha=0.8,#marker透明度，范围为0-1
                facecolors='r',#marker的填充颜色，当上面c参数设置了颜色，优先c
                edgecolors='none',#marker的边缘线色
                linewidths=1,#marker边缘线宽度，edgecolors不设置时，该参数不起作用
                label = iris_name[i])#后面图例的名称取自label
 
plt.legend(loc ='upper right')

```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BMX4o.jpg)

### 6.3 matplotlib.axes.Axes.scatter法绘制散点图

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()
x, y = iris.data, iris.target
pd_iris = pd.DataFrame(np.hstack((x,y.reshape(150,1))),columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'])
fig ,ax = plt.subplots(dpi=150)
iris_type = pd_iris['class'].unique()
iris_name = iris.target_names
colors = ['#c72e29','#098154','#fb832d']
markers = ['$\clubsuit$','.','+']

for i in range(len(iris_type)):
    plt.scatter(pd_iris.loc[pd_iris['class']== iris_type[i],'sepal length(cm)'],#传入数据x
                pd_iris.loc[pd_iris['class']== iris_type[i],'sepal width(cm)'],#传入数据y
                s =50,#散点图形（marker）的大小
                c = colors[i],#marker颜色
                marker = markers[i],#marker形状
                #marker=matplotlib.markers.MarkerStyle(marker = markers[i],fillstyle='full'),#设置marker的填充
                alpha=0.8,#marker透明度，范围为0-1
                facecolors='r',#marker的填充颜色，当上面c参数设置了颜色，优先c
                edgecolors='none',#marker的边缘线色
                linewidths=1,#marker边缘线宽度，edgecolors不设置时，改参数不起作用
                label = iris_name[i])#后面图例的名称取自label
plt.legend(loc='upper right')
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/U6wMO.jpg)



## 7. 绘制折线图plot

### 7.1 折线图中常用参数

```python
linestyle #线型
linewidth #线宽
marker #marker形状
markersize #marker大小
label #图例
alpha #线和marker的透明度
```

### 7.2 折线图详细参数

```python
import math
import matplotlib.pyplot as plt
plt.figure(dpi=200)
plt.plot(range(2,10),# x轴方向变量
         [math.sin(i) for i in range(2,10)], # y轴方向变量
         linestyle = '--',#线型
         linewidth = 2, #线宽
         color = 'red',#线和marker的颜色，当markeredgecolor markerfacecolor有设定时，仅控制line颜色
         
         # marker的特性设置
         marker='^',
         markersize='15',
         markeredgecolor='green', #marker外框颜色
         markeredgewidth=2, #marker外框宽度
         markerfacecolor='red', #marker填充色
         fillstyle='top', # marker填充形式，可选{'full','left','right','bottom','top','none'}
         markerfacecoloralt='blue', #未被填充部分颜色
         markevery=2, #每隔一个画一个marker
         label='sin(x)', #图例
         alpha = 0.3,#线和marker的透明度
         )
## 下面两条线使用默认参数绘制
plt.plot(range(2,10),[math.cos(i) for i in range(2,10)], label='cos(x)')
plt.plot(range(2,10),[2*math.cos(i)+1 for i in range(2,10)], label='2*cos(x)+1')
plt.legent()#绘制图例

plt.xlabel('x')
plt.ylabel('f(x)')
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/DCGPU.jpg)



## 8. 垂直|水平|堆积条形图

### 1. **垂直柱形图**

```python
import matplotlib.pyplot as plt
import numpy as np
plt.figure(dpi=100)
labels=['Jack','Ros','Jimmy']
year_2019 = np.arange(1,4)
plt.bar(np.arange(len(labels)), #每个柱子的名称
        year_2019, #每个柱子的高度
        bottom=0, #柱子起始位置对应纵坐标， 默认从0开始
        align='center',# 柱子名称位置，默认'center',可选'edge'
        color='pink',
        edgecolor='b',
        linewidth=1,
        tick_label=labels, #自定义每个柱子的名称
        yerr=[0.1,0.2,0.3], #添加误差棒
        ecolor='red',#误差棒颜色，默认黑色
        capsize=5, #误差棒上下的横线长度
        log=False,#y轴坐标取对数
		)
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/YX9xq.jpg)

### 2. 多个垂直柱形图并列显示

```python
import matplotlib.pyplot as plt
import numpy as np
plt.figure(dpi=100)
labels = ['Jack','Rose','Jimmy']
year_2019=np.arange(1,4)
year_2020=np.arange(1,4)+1
bar_width=0.4

plt.bar(np.arange(len(labels))-bar_width/2, #为了两个柱子一样宽
		year_2019,
        color='#B5495B',
        width=bar_width,
        label='year_2019' #图例       
       )
plt.bar(np.arange(len(labels))+bar_width/2,
        year_2020,
        color='#2ca02c',
        width=bar_width,
        label='year_2020'
       )
plt.xticks(np.arange(0,3,step=1),labels,rotation=45)#定义柱子名称
plt.legend(loc=2)#图例在左边
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/DoK4P.jpg)

### 3. 柱形图高度显示在柱子上方

```python
import matplotlib.pyplot as plt
import numpy as np
plt.figure(dpi=100)
labels = ['Jack','Rose','Jimmy']
year_2019=np.arange(1,4)
year_2020=np.arange(1,4)+1
bar_width=0.4

bar1 = plt.bar(np.arange(len(labels))-bar_width/2,#为了两个柱子一样宽
        year_2019,
        color='#B5495B',
        width=bar_width, 
        label='year_2019'#图例
        
       )
bar2 = plt.bar(np.arange(len(labels))+bar_width/2,
        year_2020,
        color='#2ca02c',
        width=bar_width,
        label='year_2020'#图例
        
       )
plt.xticks(np.arange(0, 3, step=1),labels,rotation=45)#定义柱子名称
plt.legend(loc=2)#图例放置左边


def autolabel(rects):
    """柱子上添加柱子的高度"""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 0.8),#柱子上方距离
                    textcoords="offset points",
                    ha='center', va='bottom'
                    )
        


autolabel(bar1)
autolabel(bar2)
plt.tight_layout()
"""
tight_layout会自动调整子图参数，使之填充整个图像区域。这是个实验特性，可能在一些情况下不工作。它仅仅检查坐标轴标签、刻度标签以及标题的部分。
"""
plt.show()
```



### 4. 堆积柱形图

```python
import matplotlib.pyplot as plt
import numpy as np
plt.figure(dpi=100)
labels = ['Jack','Rose','Jimmy']
year_2019=np.arange(1,4)
year_2020=np.arange(1,4)+1
bar_width=0.4

plt.bar(np.arange(len(labels)),
        year_2019,
        color='#B5495B',
        width=bar_width, 
        label='year_2019'
        )
plt.bar(np.arange(len(labels)),
        year_2020,
        color='#2ca02c',
        width=bar_width,
        bottom=year_2019,#上面柱子起始高度设置为第一个柱子的结束位置，默认从0开始
        label='year_2020'#图例
        )
plt.xticks(np.arange(0, 3, step=1),labels,rotation=45)#定义柱子名称
plt.legend(loc=2)#图例在左边
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/4U5es.jpg)

### 4. 水平柱状图参数详解

> 注意比较和垂直柱形图中参数的细微差别

```python
import matplotlib.pyplot as plt
import numpy as np
plt.figure(dpi=100)
labels=['Jack','Rose','Jimmy']
year_2019=np.arange(1,4)
plt.barh(np.arange(len(labels)), #每个柱子的名称
         width=year_2019, #柱子高度
         height=0.8,#柱子宽度，默认0.8
         left=1,#柱子底部位置对应x轴的横坐标，类似bar()中的bottom
         aligh='center',#柱子名称位置，默认为'center',可选'edge'
         color='pink',#柱子填充色
         edgecolor='b',#柱子外框线颜色
         linewidth=1,#柱子外框线宽度
         tick_label=labels,#自定义每个柱子的名称
         xerr=[0.1,0.2,0.3],#添加误差棒
         ecolor='red',#误差棒颜色，默认为黑色
         capsize=5,#误差棒上下的横县长度
         log=False,#y轴坐标取对数

		)
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/WnPxb.jpg)



## 9. 直方图（histogram）

### 9.1 绘图数据集准备

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn import datasets
iris=datasets.load_iris()
x, y = iris.data, iris.target
pd_iris = pd.DataFrame(np.hstack((x,y.reshape(150,1))),columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'] )

```

选取`pd_iris['sepal length(cm)']`数据绘制直方图，查看数据基本情况：

```python
pd_iris['sepal length(cm)'].head() #前五行

0    5.1
1    4.9
2    4.7
3    4.6
4    5.0
Name: sepal length(cm), dtype: float64
        
pd_iris['sepal length(cm)'].describe() #简单统计下数据
count    150.000000
mean       5.843333
std        0.828066
min        4.300000
25%        5.100000
50%        5.800000
75%        6.400000
max        7.900000
Name: sepal length(cm), dtype: float64

```



### 9.2 matplotlib.pyplot.hist直方图参数详解

修改对应参数，即可体验对应参数的功能；

大部分参数使用默认值即可。

```python
import palettable
import random
plt.figure(dpi=150)
data=pd_iris['sepal length(cm)']
n, bins, patches=plt.hist(x=data,
                          ##箱子数(bins)设置，以下三种不能同时并存
                          #bins=20,#default: 10
                          #bins=[4,6,8],#分两个箱子，边界分别为[4,6),[6,8]
                          #bins='auto',# 可选'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
                          #选择最合适的bin宽，绘制一个最能反映数据频率分布的直方图 
                         
                          #range=(5,7),#最左边和最右边箱子边界，不指定时，为(x.min(), x.max())
                          #density=True, #默认为False，y轴显示频数；为True y轴显示频率，频率统计结果=该区间频数/(x中总样本数*该区间宽度)
                          #weights=np.random.rand(len(x)),#对x中每一个样本设置权重，这里随机设置了权重
                          cumulative=False,#默认False，是否累加频数或者频率，及后面一个柱子是前面所有柱子的累加
                          bottom=0,#设置箱子y轴方向基线，默认为0，箱子高度=bottom to bottom + hist(x, bins)
                          histtype='bar',#直方图的类型默认为bar{'bar', 'barstacked', 'step', 'stepfilled'}
                          align='mid',#箱子边界值的对齐方式，默认为mid{'left', 'mid', 'right'}
                          orientation='vertical',#箱子水平还是垂直显示，默认垂直显示('vertical')，可选'horizontal'
                          rwidth=1.0,#每个箱子宽度，默认为1，此时显示50%
                          log=False,#y轴数据是否取对数，默认不取对数为False
                          color=palettable.colorbrewer.qualitative.Dark2_7.mpl_colors[1],
                          label='sepal length(cm)',#图例
                          #normed=0,#功能和density一样，二者不能同时使用
                          facecolor='black',#箱子颜色 
                          edgecolor="black",#箱子边框颜色
                          stacked=False,#多组数据是否堆叠
                          alpha=0.5#箱子透明度
                         )
plt.xticks(bins)#x轴刻度设置为箱子边界

for patch in patches:#每个箱子随机设置颜色
    patch.set_facecolor(random.choice(palettable.colorbrewer.qualitative.Dark2_7.mpl_colors))

#直方图三个返回值
print(n)#频数
print(bins)#箱子边界
print(patches)#箱子数

#直方图绘制分布曲线
plt.plot(bins[:10],n,'--',color='#2ca02c')
plt.hist(x=[i+0.1 for i in data],label='new sepal length(cm)',alpha=0.3)
plt.legend()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/yrwcm.jpg)

## 10. 热图heatmap

### 1. matplotlib绘制热图

matplotlib可通过以下两种方法绘制heamap；

- matplotlib.axes.Axes.imshow
- matplotlib.pyplot.imshow

原始效果图，挺丑陋的；



改进后效果图（虽然要写很多辅助函数实现，但是可以很好的实现自定义热图，**需要高度个性化**的小伙伴可以去摸索）； 

### 2. seaborn绘制热图

seaborn在matplotlib的基础上封装了个seaborn.heatmap，**非常傻瓜式操作**，我等调包侠的福音，效果可以赶得上R语言了，不逼逼，下面上干货：

#### 2.1 数据集准备

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import palettable#python颜色库
from sklearn import datasets 


plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文

iris=datasets.load_iris()
x, y = iris.data, iris.target
pd_iris = pd.DataFrame(np.hstack((x, y.reshape(150, 1))),columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'] )

plt.figure(dpi=200, figsize=(10,6))
data1 = np.array(pd_iris['sepal length(cm)']).reshape(25,6)#Series转np.array


df = pd.DataFrame(data1, 
                  index=[chr(i) for i in range(65, 90)],#DataFrame的行标签设置为大写字母
                  columns=["a","b","c","d","e","f"])#设置DataFrame的列标签
```

用来绘制热图的数据集是什么样子的？其实就是取iris中的一列（150个值），转化为一个25x6的DataFrame数据集，如下：

```python
print(df.shape)
df.head()
```

#### 2.2 seaborn绘制heatmap

语法：seaborn.heatmap

##### 2.2.1 seaborn默认参数绘制heatmap

```python
plt.figure(dpi=120)
sns.heatmap(data=df,#矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签        
     )
plt.title('所有参数默认')
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/4VaQJ.jpg)

##### 2.2.2 colorbar（图例）范围修改：vmin、vmax

```python
#右侧colorbar范围修改
#注意colorbar范围变换，左图颜色也变
plt.clf()
plt.figure(dpi=200)
sns.heatmap(data=df,
            vmin=5,#图例中最小显示值
            vmax=8,
           )
plt.title('change vim,vmax')
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/xz3Kp.jpg)

##### 2.2.3 修改热图颜色盘（colormap）：cmp

感觉默认颜色太丑陋，可以换个颜色盘，cmp参数控制hetmap颜色；
可以使用matplotlib颜色盘、seaborn颜色盘、palettable库中颜色盘

> 使用matplotlib中colormap

了解matplotlib中所有colormap请戳：[matplotlib中colormap使用详解](https://mp.weixin.qq.com/s?__biz=MzUwOTg0MjczNw==&mid=2247484329&idx=1&sn=20ec36f7f5077221671b32d47c3412c8&chksm=f90d47f7ce7acee11449c5584a11a020cf05c27f7a60b9357bbdc35181cc7e8420c594d09fc5&scene=21#wechat_redirect)

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
           cmap=plt.get_cmap('Set3'),
           )
plt.title('use matplotlib cmap=plt.get_cmap('Set3')')
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/kc2z4.jpg)

```python
#感觉太油腻，太花哨，那就来个纯一点的（色度依次增加，请看右边图例颜色变化）
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=plt.get_cmap('Greens'),#matplotlib中的颜色盘'Greens'
           )
plt.title("cmap=plt.get_cmap('Greens')")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/RlBKh.jpg)

```python
#色度依次递减
plt.figure(dpi=120)
sns.heatmap(data=df,                 
            #cmap选取的颜色条，有的是由浅到深（'Greens'），有的是相反的（'Greens_r'）
            cmap=plt.get_cmap('Greens_r'),#matplotlib中的颜色盘'Greens_r'
           )
plt.title("使用matplotlib中的颜色盘：cmap=plt.get_cmap('Greens_r')")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/M8lBY.jpg)


> 使用Seaborn颜色盘

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=sns.dark_palette("#2ecc71", as_cmap=True),#seaborn 深色色盘：sns.dark_palette使用
           )
plt.title("使用seaborn dark颜色盘：cmap=sns.dark_palette('#2ecc71', as_cmap=True)")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/RxJ0T.jpg)

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=sns.light_palette("#2ecc71", as_cmap=True),#淡色色盘：sns.light_palette()使用
           )
plt.title("使用seaborn light颜色盘：sns.light_palette('#2ecc71', as_cmap=True)")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/jhg6C.jpg)

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=sns.diverging_palette(10, 220, sep=80, n=7),#区分度显著色盘：sns.diverging_palette()使用
           )
plt.title("使用seaborn diverging颜色盘：sns.diverging_palette(10, 220, sep=80, n=7)")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/Z7AaJ.jpg)

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=sns.cubehelix_palette(as_cmap=True),#渐变色盘：sns.cubehelix_palette()使用
           )
plt.title("使用seaborn cubehelix颜色盘：sns.diverging_palette(220, 20, sep=20, as_cmap=True)")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BrFzM.jpg)

> 使用palettable库中颜色盘

关于python palettable库使用请戳：[python Palettable库使用详解](https://mp.weixin.qq.com/s?__biz=MzUwOTg0MjczNw==&mid=2247484380&idx=1&sn=f591dd15bf5feb65fafe8622652b868d&chksm=f90d4782ce7ace9407245a0db4813c1d6547f88a06d5af5c5853c9211d1f5579939f5f76d6ed&scene=21#wechat_redirect)

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,#使用palettable库中颜色条
           )
plt.title("使用palettable库颜色盘：palettable.cartocolors.diverging.ArmyRose_7.mpl_colors")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/OaDTX.jpg)

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,#使用palettable库中颜色条
           )
plt.title("使用palettable库颜色盘：palettable.cmocean.diverging.Curl_10.mpl_colors")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/fAeCz.jpg)

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.tableau.TrafficLight_9.mpl_colors,#使用palettable库中颜色条
           )
plt.title("使用palettable库颜色盘：palettable.tableau.TrafficLight_9.mpl_colors")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/JtRVs.jpg)

##### 2.2.4 修改图例中心数据值大小：center

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
            center=7,#color bar的中心数据值大小，可以控制整个热图的颜盘深浅
           )
plt.title("color bar的中心数据值大小：center")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/zhqjl.jpg)

##### 2.2.5 热图中文本开关

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
            annot=True,#默认为False，当为True时，在每个格子写入data中数据
           )
plt.title("每个格子写入data中数据：annot=True")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/tlhsr.jpg)

##### 2.2.6 格子中数据的格式化输出:fmt

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
            annot=True,#默认为False，当为True时，在每个格子写入data中数据
            fmt=".2f",#设置每个格子中数据的格式，参考之前的文章，此处保留两位小数
           )
plt.title("格子中数据的格式化输出：fmt")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/D1geS.jpg)



##### 2.2.7 格子中数据（字体大小、磅值、颜色）等设置：annot_kws

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
            annot=True,#默认为False，当为True时，在每个格子写入data中数据
            annot_kws={'size':8,'weight':'normal', 'color':'blue'},#设置格子中数据的大小、粗细、颜色
           )
plt.title("格子中数据（字体大小、磅值、颜色）等设置：annot_kws")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/3GNue.jpg)

##### 2.2.8 格子外框宽度、颜色设置：linewidths、linecolor

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
            linewidths=1,#每个格子边框宽度，默认为0
            linecolor='red',#每个格子边框颜色,默认为白色
            
           )
plt.title("格子外框宽度、颜色设置：linewidths、linecolor")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BifPb.jpg)

##### 2.2.9 图例开关：cbar

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
            cbar=False,#右侧图例(color bar)开关，默认为True显示
           )
plt.title("图例开关：cbar")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/IEGtm.jpg)

##### 2.2.10 图例位置、名称、标签等设置：cbar_kws

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
            cbar=True,
            cbar_kws={'label': 'ColorbarName', #color bar的名称
                           'orientation': 'horizontal',#color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                           "ticks":np.arange(4.5,8,0.5),#color bar中刻度值范围和间隔
                           "format":"%.3f",#格式化输出color bar中刻度值
                           "pad":0.15,#color bar与热图之间距离，距离变大热图会被压缩
                                                   },
            
           )
plt.title("图例位置、名称、标签等设置：cbar_kws")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/A2OLH.jpg)

##### 2.2.11 热图中只显示部分符合条件的数据：mask

```python
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
            mask=df<6.0,#热图中显示部分数据：显示数值小于6的数据 
         )
plt.title("热图中只显示部分符合条件的数据：mask")
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/QyxEH.jpg)

##### 2.2.12 自定义x轴、y轴标签：xticklabels、yticklabels

```python
# linewidths、linecolor参数
plt.figure(dpi=120)
sns.heatmap(data=df,
            cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
            xticklabels=['a','b','c','d','e','f'] , #x轴方向刻度标签开关、赋值，可选“auto”, bool, list-like（传入列表）, or int,
            yticklabels=True, #y轴方向刻度标签开关、同x轴
         )
plt.title("自定义x轴、y轴标签：xticklabels、yticklabels")
#['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'] 
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/YZVZQ.jpg)



## 11. 箱图boxplot

线图用来展现数据的分布，能直观的展示数据的关键指标（如下四分位数、上四分位数、中位数、最大值、最小值、离散点/异常值点）；箱线图可直观展示不同组数据的差异；下面详细介绍python中`matplotlib`及`seaborn`库绘制箱图。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/XePK4.jpg)

### 1. 数据集准备及箱图简介

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import palettable
from sklearn import datasets 


plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文

iris=datasets.load_iris()
x, y = iris.data, iris.target
pd_iris = pd.DataFrame(np.hstack((x, y.reshape(150, 1))),columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'] )
```

```python
pd_iris["sepal width(cm)"].describe()

count    150.000000
mean       3.054000
std        0.433594
min        2.000000
25%        2.800000（下四分位数，25% 的数据小于等于此值。）
50%        3.000000（中位数，50% 的数据小于等于此值。）
75%        3.300000（上四分位数，75% 的数据小于等于此值。）
max        4.400000
Name: sepal width(cm), dtype: float64
```



### 2. seaborn.boxplot箱图外观设置

#### 2.1 默认参数

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],#传入一组数据
            orient='v'#箱子垂直显示，默认为'h'水平显示
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/fKz6N.jpg)

```python
plt.figure(dpi=100)
sns.boxplot(x=pd_iris["sepal width(cm)"],#传入一组数据
            orient='h'#箱子垂直显示，默认为'h'水平显示
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/uuiiM.jpg)

#### 2.2 箱图异常值属性设置

异常值关闭显示

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            showfliers=False,#异常值关闭显示
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/UUQLU.jpg)



异常值marker大小设置

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            orient='v',
            fliersize=15,#设置离散值marker大小，默认为5
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/RbO81.jpg)

异常值marker形状、填充色、轮廓设置

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            orient='v',
            flierprops = {'marker':'o',#异常值形状
                          'markerfacecolor':'red',#形状填充色
                          'color':'black',#形状外廓颜色
                         },
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/IfYuK.jpg)

#### 2.3 箱图上下横线属性设置

上线横线关闭

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            showcaps=False,#上下横线关闭
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/PiF9d.jpg)

上下横线颜色、线型、线宽等设置

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            capprops={'linestyle':'--','color':'red'},#设置上下横线属性
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/ExPkr.jpg)

#### 2.4 箱图上下须线属性设置

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            whiskerprops={'linestyle':'--','color':'red'},#设置上下须属性
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/rGVpF.jpg)

#### 2.5 箱图箱子设置

箱子设置缺口

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            orient='v',
            notch=True,#箱子设置缺口
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/qUzhi.jpg)

箱子不填充颜色

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            orient='v',
            color='white',#箱子不填充
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/FYL6I.jpg)

箱子外框、内部填充色

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
           boxprops = {'color':'red',#箱子外框
           'facecolor':'pink'#箱子填充色
           },#设置箱子属性
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/QKPOQ.jpg)

#### 2.6 箱图中位数线属性设置

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
           medianprops = {'linestyle':'--','color':'red'},#设置中位数线线型及颜色
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/toh8Z.jpg)

#### 2.7 箱图均值属性设置

均值使用点显示、设置点形状、填充色

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            showmeans=True,#箱图显示均值，
            meanprops = {'marker':'D','markerfacecolor':'red'},#设置均值属性
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/XWCb2.jpg)

均值使用线显示 、线型、颜色设置

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            showmeans=True,#箱图显示均值，
            meanline=True,#显示均值线
            meanprops = {'linestyle':'--','color':'red'},#设置均值线属性
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/tVNso.jpg)

#### 2.8 箱图中所有线属性设置

```python
plt.figure(dpi=100)
sns.boxplot(y=pd_iris["sepal width(cm)"],
            orient='v',
            linewidth=8#设置箱子等线的宽度
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/rs8ZX.jpg)

### 3. seaborn.boxplot**分组**箱图

#### 3.1 分组绘图（方法一）

```python
plt.figure(dpi=100)
class_name=[iris.target_names[0] if i==0.0 else iris.target_names[1] if i==1.0 else iris.target_names[2] for i in pd_iris['class']]
sns.boxplot(x=class_name,#按照pd_iris["sepal width(cm)"]分组，即按照每种鸢尾花（'setosa', 'versicolor', 'virginica'）分组绘图
            y=pd_iris["sepal width(cm)"],#绘图数据
            orient='v'
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/x70xH.jpg)

#### 3.2 分组绘图（方法二）

```python
plt.figure(dpi=100)
sns.boxplot(x='class',
            y='sepal width(cm)',
            data=pd_iris,#data的作用就是x，y每次不需要输入pd_iris
            orient='v'
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/fjeII.jpg)

#### 3.3 箱子颜色设置

设置箱子颜色

```python
import palettable
plt.figure(dpi=100)
sns.boxplot(x='class',
            y='sepal width(cm)',
            data=pd_iris,
            orient='v',
            palette=palettable.tableau.TrafficLight_9.mpl_colors,#设置每个箱子颜色
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/xYBUk.jpg)

设置箱子颜色饱和度

```python
import palettable
plt.figure(dpi=100)
sns.boxplot(x='class',
            y='sepal width(cm)',
            data=pd_iris,
            orient='v',
            palette=palettable.tableau.TrafficLight_9.mpl_colors,
            saturation=0.3,#设置颜色饱和度
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/2Rs4M.jpg)

#### 3.4 箱子间距设置

```python
import palettable
plt.figure(dpi=100)
sns.boxplot(x='class',
            y='sepal width(cm)',
            data=pd_iris,
            orient='v',
            palette=palettable.tableau.TrafficLight_9.mpl_colors,
            saturation=0.3,#设置颜色饱和度
            width=1.0,#设置箱子之间距离，为1时，每个箱子之间距离为0
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/nST4a.jpg)

#### 3.5 每个小组再按子组绘图

```python
plt.figure(dpi=100)
class_name=[iris.target_names[0] if i==0.0 else iris.target_names[1] if i==1.0 else iris.target_names[2] for i in pd_iris['class']]
sns.boxplot(x=class_name,
            y=pd_iris['sepal width(cm)'],
            hue=pd_iris['petal width(cm)'],#每类按照子类分组：上图三类再按照'sepal width(cm)'分组绘图
            orient='v'
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/YR3Lt.jpg)

#### 3.6 按顺序绘制箱图

```python
plt.figure(dpi=100)
class_name=[iris.target_names[0] if i==0.0 else iris.target_names[1] if i==1.0 else iris.target_names[2] for i in pd_iris['class']]
sns.boxplot(x=class_name,
            y=pd_iris["sepal width(cm)"],
            hue=pd_iris['petal width(cm)'],
            order=["virginica", "versicolor", "setosa"],#设置箱子的显示顺序
            hue_order=sorted(list(pd_iris['petal width(cm)'].unique())),#设置每个子类中箱子的显示顺序，此处设置从小到大排序
            orient='v'
           )
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/IkHcv.jpg)