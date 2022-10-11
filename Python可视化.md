# Python可视化

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/Jthxw.jpg)

## 导入数据集

```python
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel("E:/First.xlsx", "Sheet1")
```



## 可视化为直方图

```python
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(df['Age'],bins=7)
# Labels and Title
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('#Employee')
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/LdWgZ.jpg)

## 可视化为箱线图

```python
ax = fig.add_subplot(1,1,1)
ax.boxplot(df['Age'])
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/JQBxk.jpg)



## 可视化为小提琴图

```python
import seaborn as sns
sns.violinplot(df['Age'],df['Gender'])
sns.despine()
```

![image-20210107100013664](C:\Users\nuaazs\AppData\Roaming\Typora\typora-user-images\image-20210107100013664.png)



## 可视化为条形图

```python
var = df.groupby('Gender').Sales.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Sum of Sales')
ax1.set_title('Gender wise Sum of Sales')
var.plot(kind='bar')
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/ZWAZ3.jpg)



## 可视化为折线图

```python
var = df.groupby('BMI').Sales.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('BMI')
ax1.set_ylabel('Sum of Sales')
ax1.set_title('BMI wise Sum of Sales')
var.plot(kind='line')
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/c3ob8.jpg)



## 可视化为堆叠柱状图

```python
var = df.groupby(['BMI','Gender']).Sales.sum()
var.unstack().plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/0RHkZ.jpg)



## 可视化为散点图

```python
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['Age'],df['Sales'])
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/xdL1w.jpg)



## 可视化为泡泡图

```python
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['Age'],df['Sales'],s=df['Income'])
# Added thrid variable income as size of the bubble
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/HiIox.jpg)



## 可视化为饼状图

```python
var = df.groupby(['Gender']).sum().stack()
temp = var.unstack()
type(temp)

x_list = temp['Sales']
label_list = temp.index
pyplot.axis("equal")
# The pie chart is oval by default.
# To make it a circle use pyplot.axis("equal")
# To show the percentage of each pie slice, pass an output
# format to the autopctparameter

plt.pie(x_list, labels=label_list, autopct="%1.1f%%")
plt.title("Pastafarianism expenses")
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/xzsmB.jpg)



## 可视化为热度图

```python
import numpy as np
# Generate a random number, you can  refer your data values alse
data = np.random.rand(4,2)
rows = list('1234')
# rows categories
columns = list('MF')
# column categories

fig, ax = plt.subplots()
# Advance color controls
ax.pcolor(data, cmap=plt.cm.Reds, edgecolors='k')
ax.set_xticks(np.arange(0,2)+0.5)
ax.set_yticks(np.arange(0,4)+0.5)

# Here we position the tick labels for x and y axis
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# Values against each labels
ax.set_xticklabels(columns, minor=False, fontsize=20)
ax.set_yticklabels(rows, minor=False, fontsize=20)
plt.show()
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/fIQD3.jpg)



