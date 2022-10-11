# 理解numpy的axis

## 1. numpy库创建数组

```python
# 假设下图是一个单波段图像的灰度图，用numpy数据表示就是
a=np.array([[1,100,55],[79,22,79],[16,47,21]])
print(a)
print(a.shape)
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/6Jj8A.jpg)


可以看到输出的shape是`(3,3)`，第一个`3`对应的是最外层的`[]`,下面有3个子块分别是`[1,100,55]`,`[79,22,79]`,`[16,47,21]`,所以这一层的尺度为`3`。再看第二层`[]`，如`[1,100,55]`下面有3个独立的子块，分别为`1`,`100`,`55`，所以这一层的尺度为`3`。
依次类推我们看看三维的情况，假如下面是一个3波段的图像数据。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/kZrmZ.jpg)

```python
a=np.array([[[1,100],[79,22],[16,195],[189,56]],
[[32,4],[21,88],[57,250],[18,93]],
[[246,18],[75,37],[45,247],[6,47]]])
print(a)
print(a.shape)
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/3IuB4.jpg)

输出的结果为3通道，4行，2列。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/EHFIS.jpg)

结合上图，最外层为 `[]`,内部有三个 `[]`,所以shape的第一位为`3`。

第二层为`[]`,每个`[]`内部有4个 `[]`,所以shape的第二位为`4`。

第三层为 `[]`,每个 `[]`中有2个元素,所以shape的第三位为`2`。

> 总结：使用numpy创建数组时，每增加一个[]表示增加一个维度，一个[]可以看成一个完整的块，这点对下一步分维度计算时的理解很重要。每层[]下有几块，shape对应的值就为几。



## 2. 设置axis时的计算原理

仍然使用上面三通道图像的例子。分别计算`axis=0,1,2`的sum值

再来回顾一下上面说的使用numpy创建数组时，每增加一个`[]`表示增加一个维度，一个`[]`可以看成一个完整的块。
使用axis参数时，就是计算对应`[]`下的子块。计算完后删除对应层的`[]`，维度会少一层，想要不删除`[]`，可以使用`keep_dims=True`参数，我们分别测试一下。

### 2.1 axis=0

```python
a=np.array([[[1,100],[79,22],[16,195],[189,56]],
[[32,4],[21,88],[57,250],[18,93]],
[[246,18],[75,37],[45,247],[6,47]]])

sum_axis0=np.sum(a,axis=0)
print("axis=0:")
print(sum_axis0,"\n")
print(sum_axis0.shape)

sum_axis0_keep=np.sum(a,axis=0,keepdims=True)
print("axis=0,keep_dims=true:")
print(sum_axis0_keep)
print(sum_axis0_keep.shape)
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/zQjOU.jpg)

可以看到，使用`keep_dims=True`之后，会保留对应层的维度。

下面使用图像来还原一下计算过程。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/vcgIA.jpg)

当`axis=1`时，对应`绿色 []`这一层，我们的例子中有三个 `绿色[]`，分别单独计算。

第一个 `[]`下有4个 `[]`子块，把这4个子块对应位置相加：`[1,100]+[79,22]+[16,195]+[189,56]=[285,373]`，其他两个` []`依次类推，得到统计结果。当`keep_dims=False`时，删除 []这一层，对应的就是`(3,1,2)`中的第二位`1`。


### 2.2 axis=2

```python
a=np.array([[[1,100],[79,22],[16,195],[189,56]],
[[32,4],[21,88],[57,250],[18,93]],
[[246,18],[75,37],[45,247],[6,47]]])

sum_axis2=np.sum(a,axis=2)
print("axis=2:")
print(sum_axis2,"\n")
print(sum_axis2.shape)

sum_axis2_keep=np.sum(a,axis=2,keepdims=True)
print("axis=2,keep_dims=true:")
print(sum_axis2_keep)
print(sum_axis2_keep.shape)
```

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/umFye.jpg)



## 3. 总结

1、numpy的数组中，每一层[]对应一个维度。
2、使用axis统计时，表示在第axis层下进行统计，统计的单位为当前级别的子块，按子块进行运算。
3、默认keep_dims=False，numpy计算后会删除对应axis层，因为该维度元素个数计算后始终为1，要想保持计算前后维度一致，需要设置keep_dims的值为True。