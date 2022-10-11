# Matplotlib如何画出高大上风格？

我们用Python里的Matplotlib库画出的分析图可能往往如下图所示：它看起来很普通

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/OgACz.jpg)

```python
import matplotlib.pyplot as plt
%matplotlib inline
from numpy import arange

fandango_2015['Fandango_Stars'].plot.kde(label='2015',legend=True)
fandango_2016['fandango'].plot.kde(label='2016',legend=True)

plt.title("Comparing distribution shapes for Fandango's ratings\n(2015 vs 2016)")
plt.xlabel('rate')
plt.xlim(0,5)
plt.xticks(arange(0,5,.5))
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/XJYEd.jpg)

这里我们用的是经典的fivethirtyeight 风格,所需的代码很简单,只需加上一行:

```python
plt.style.use('fivethirtyeight')
```

fivethirtyeight 是matplotlib众多风格的一种，我们可以通过以下代码查看各种风格:

```python
import matplotlib.style as style
style.available
```

```python
['seaborn-deep',
'seaborn-muted',
'bmh',
'seaborn-white',
'dark_background',
'seaborn-notebook',
'seaborn-darkgrid',
'grayscale',
'seaborn-paper',
'seaborn-talk',
'seaborn-bright',
'classic',
'seaborn-colorblind',
'seaborn-ticks',
'ggplot',
'seaborn',
'_classic_test',
'fivethirtyeight',
'seaborn-dark-palette',
'seaborn-dark',
'seaborn-whitegrid',
'seaborn-pastel',
'seaborn-poster']
```

fivethirtyeight风格就在其中。实际上fivethirtyeight是一个数据科学的内容网站，你会被它令人敬畏的视觉效果所打动。