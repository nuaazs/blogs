## 单通道输出

在训练时，输出通道为1，网络的输出数值是任意的。标签是单通道的二值图，对输出使用sigmoid，使其数值归一化到[0,1]，然后和标签做交叉熵损失。

训练结束后，将输出的output经过sigmoid函数，然后取阈值（一般为0.5），大于阈值则为1否则取0，从而得到最终的预测结果。

```python
# 第一种
output = net(input) # net最后一层没有使用sigmoid
Loss = torch.nn.BCEWithLogitsLoss() # 会先做sigmoid然后求交叉熵
loss = Loss(output, target)

# 第二种
output = net(input) #net最后一层没有使用sigmoid
output = F.sigmoid(output)
Loss = torch.nn.BCEWithLoss()
loss = Loss(output, target)

# 预测
output = net(input)
output = F.sigmoid(output)
predict = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
```



## 多通道输出

在训练时，输出通道为2，网络的输出数值是任意的。让网络的输出经过softmax，归一化到[0,1],在各通道中，同一位置加起来的数值会等于1。标签是单通道的二值图，首先使用one-hot编码，使其变为二通道，当前通道值为1，另一通道上就为0。然后将输出和标签做交叉熵损失。

训练结束后，取每个像素位置上对应最大值的通道序号为最终的预测值，从而得到最终的预测结果。

```python
# 训练
output = net(input) # net的最后一层没有使用sigmoid
Loss = torch.nn.CrossEntropyLoss()
loss = Loss(output, target)

# 预测
output = net(input) # net的最后一层没有使用sigmoid
predict = output.argmax(dim=1)
```

