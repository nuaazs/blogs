```python
import sys
from shutil import copyfile
import oss2
from pyperclip import copy # 用来操作粘贴板
from random import choice

map = ["a", "b", "c", "d", "e", "f", "g", "h",
"i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
"u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5",
"6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H",
"I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
"U", "V", "W", "X", "Y", "Z"]

def url_s(m):
    url = ""
    for i in range(m):
        url = url + str(choice(map))
    return url

# 获取远程文件名
# def get_remote_file_name(local_name):
# name = uuid.uuid4().__str__().replace("-", "").upper()
# local_name = str(local_name).rsplit(".")
# return "pics/%s.%s" % (name, local_name[-1])

# 随机生成短链
def get_shourt_file_name(local_name, m):
    name = url_s(m)
    local_name = str(local_name).rsplit(".")
    return "pics/%s.%s" % (name, local_name[-1])


BUCKET_NAME = "bucket名称"
PRE = "https://bucket名称.oss-cn-hangzhou.aliyuncs.com/"
length = 5
# PIC_STYLE = "!1"
ENDPOINT = "oss-cn-地理位置.aliyuncs.com"
ACCESS_KEY_ID = "****"
ACCESS_KEY_SECRET = "*****"
addr = r"C:\Users\zhaosheng\OneDrive\markdown"

src_file = r"要上传的源文件地址"
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

remote_file_name = get_shourt_file_name(src_file, int(length))

bucket.put_object_from_file(remote_file_name, src_file) # 上传文件
result_str = "![](%s%s)" % (PRE, remote_file_name)
copy(result_str) # 将结果复制到粘贴板方便直接使用
print('okay')
print(result_str)
f = open(addr + 'log.txt', 'a')
f.write(src_file + ':' + result_str + '\n')
f.close()

# 记录log
if (1):
    copyfile(src_file, addr + remote_file_name)
print("移动至" + addr + remote_file_name)


```

