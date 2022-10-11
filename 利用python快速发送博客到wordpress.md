 

## 安装

```shell
pip3 install python-wordpress-xmlrpc

pip install python-wordpress-xmlrpc

pip install python-frontmatter
```







## 使用

```python
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
import frontmatter
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import GetPosts, NewPost
from wordpress_xmlrpc.methods.users import GetUserInfo
from wordpress_xmlrpc.methods import posts
from wordpress_xmlrpc.methods import taxonomies
from wordpress_xmlrpc import WordPressTerm
from wordpress_xmlrpc.compat import xmlrpc_client
from wordpress_xmlrpc.methods import media, posts
import datetime
import sys


def sends():
    wp = Client('http://网址/xmlrpc.php', '用户名', '密码')
    filename = r'C:\Users\zhaosheng\OneDrive\markdown\PyTorch常用代码段整理.md'
    with open(filename, 'rb') as html:
        post_content_html = html.read()
    post = WordPressPost()
    post.title = "Quicker"
    post.date_modified = datetime.datetime.now()
    post.content = post_content_html
    post.post_status = 'publish'
    post.terms_names = {
        'post_tag': ['quicker'],  # 文章所属标签，没有则自动创建
        'category': ['quicker']  # 文章所属分类，没有则自动创建
    }

    post.custom_fields = []  # 自定义字段列表
    post.custom_fields.append({  # 添加一个自定义字段
        'key': 'price',
        'value': 3
    })
    post.custom_fields.append({  # 添加第二个自定义字段
        'key': 'ok',
        'value': '天涯海角'
    })

    wp.call(NewPost(post))
    time.sleep(3)
    print('posts updates')

if __name__=='__main__':
    sends()
```

