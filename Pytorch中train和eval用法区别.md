## Pytorch:model.train 与 model.eval用法区别

使用Pytorch进行训练和测试，要把实例化的model指定train/eval



### 原问题：

> I have a model that works perfectly when there are multiple input.However, if there is only one datapoint as input, i will get the above error. Does anyone have an idea on what's going on here?

### 回答：

> Adding `model.eval()` is the key.
>
> Most likely you have a `nn.BatchNorm` layer somewhere in your model, which expects more then 1 value to calculate the running mean and std of the current batch.
>
> In case you want to validate your data, call `model.eval()` before feeding the data, as this will change the behavior of the BatchNorm layer to use the running estimates instead of calculating them for the current batch.

### model.eval()

使用eval时，框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，一旦test的batch_size过小时，就很容易收到BN层的影响导致生成的图片错误。

不启用BatchNormalization和Dropout。

### model.train()

启用BatchNormaliztion和Dropout。



训练完train样本后，生产的模型model要用来测试样本。在`model(test)` 之前，不要加上`model.eval()` ，否则的话，有输入数据，即使不训练也会改变权值。



