# 保存Numpy的索引

将numpy的索引保存至变量，方便在for循环里面调用。

利用`np.s_`保存：

```python
import numpy as np
_index = np.s_[1:5:2,::3]

# 使用方法
result = raw[_index]
```

