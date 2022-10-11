## 报错
> UserWarning Argument interpolation should be of type InterpolationMode instead of int

## 解决
这其实是torchvision和pillow不兼容导致的，我的设备里torchvision=0.9.1 and pillow=7.0.0，即使我把pillow升级到8.3.1，依然有warning。

那只能降低torchvision了，但是torchvision的版本号一般都是和pytorch绑定好的，我们需要不依赖torch来更改torchvison的版本，这可以通过以下指令实现 参考解决安装torchvision自动更新torch到最新版本
```shell
pip install --no-deps torchvision==0.8.2
```
再将pillow降到6.2.2即可，其他版本的pillow就没有尝试过了