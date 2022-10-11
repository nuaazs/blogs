报错：
```shell
➜  ~ visdom
Downloading scripts. It might take a while.
ERROR:root:Error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:777) while downloading https://unpkg.com/jquery@3.1.1/dist/jquery.min.js
ERROR:root:Error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:777) while downloading https://unpkg.com/bootstrap@3.3.7/dist/js/bootstrap.min.js
ERROR:root:Error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:777) while downloading https://unpkg.com/react@16.2.0/umd/react.production.min.js
```
Solution：

打开 server.py文件
```shell
/Users/xxx/Library/Python/3.6/lib/python/site-packages/visdom/server.py
```
在 `server.py`文件头，添加如下代码：
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```