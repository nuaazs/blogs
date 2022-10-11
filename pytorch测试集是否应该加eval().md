## 背景
在用pytorch做图像转换的时候，发现在测试过程中如果使用了`--eval`那么测试性能大幅度下降。但是去掉eval之后性能正常。

## 解释
知乎：eval()会改变BN和Dropout的参数。所以网络会和当前数据集不匹配。简单讲就是：训练时，BN和Dropout会学出一个针对训练集数据分布的良好参数，但一般测试集与训练集分布不同。所以用eval()来使得测试更加合理（不然你需要手动改，eval()可以简化操作）。但有些情况在逻辑上就有要求同分布，加eval()反而会降低。看你实际需求，我做GAN的时候不懂，测试的时候保留了几个训练集的生成结果，结果也是一团糊，就是因为加了eval()。

https://discuss.pytorch.org/t/bad-prediction-of-batches-when-model-eval/32970