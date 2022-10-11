```python
from math import *

def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab
 
def corrcoef(x,y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x,y)
    #求平方和
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den
```

皮尔逊相关系数衡量的是一种线性相关程度的大小，取值位于-1和1之间。大于0是正相关，小于0是负相关。相关系数绝对值越大，说明线性相关的程度越强。

相关系数是最早由统计学家卡尔·皮尔逊设计的统计指标，是研究变量之间线性相关程度的量，一般用字母 r 表示。由于研究对象的不同，相关系数有多种定义方式，较为常用的是皮尔逊相关系数。相关系数r的绝对值一般在0.8以上，认为A和B有强的相关性。0.3到0.8之间，可以认为有弱的相关性。0.3以下，认为没有相关性。记录[来源](https://zhidao.baidu.com/question/526371815.html)。