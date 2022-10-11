```shell
pip freeze>requirements.txt
pip download -r requirements.txt -d path/to/folder
```
新环境中：
```shell
pip install -r requirements.txt --find-links=path/to/folder
```