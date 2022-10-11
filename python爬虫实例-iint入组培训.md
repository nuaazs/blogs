



# Python爬虫基础介绍



## 0.正则表达式

https://zhuanlan.zhihu.com/p/127807805

https://www.runoob.com/python3/python3-reg-expressions.html

https://www.runoob.com/regexp/regexp-syntax.html

https://iint.icu/python_cookbook/



## 1. 认识网页结构

网页一般由三部分组成，分别是 HTML（超文本标记语言）、CSS（层叠样式表）和 JavaScript（活动脚本语言）。

### 1.1 HTML

HTML 是整个网页的结构，相当于整个网站的框架。带`＜`、`＞`符号的都是属于 HTML 的标签，并且标签都是成对出现的。

### 1.2 CSS

CSS 表示样式，`＜style type=＂text/css＂＞ `表示下面引用一个 CSS，在 CSS 中定义了外观。

CSS选择器：https://www.w3school.com.cn/cssref/css_selectors.asp

### 1.3 JavaScript

交互的内容和各种特效都在 JavaScript中，JavaScript描述了网站中的各种功能。





## 2. HTTP工作原理

HTTP协议定义Web客户端如何从Web服务器请求Web页面，以及服务器如何把Web页面传送给客户端。HTTP协议采用了请求/响应模型。客户端向服务器发送一个**请求报文**，请求报文包含`请求的方法`、`URL`、`协议版本`、`请求头部`和`请求数据`。

服务器以一个**状态行**作为响应，响应的内容包括`协议的版本`、`成功或者错误代码`、`服务器信息`、`响应头部`和`响应数据`。

###  2.1 HTTP 请求/响应的步骤

客户端连接到Web服务器->发送Http请求->服务器接受请求并返回HTTP响应->释放连接TCP连接->客户端浏览器解析HTML内容

#### 1. 客户端连接到Web服务器

一个HTTP客户端，通常是浏览器，与Web服务器的HTTP端口（默认为80）建立一个TCP套接字连接。

#### 2. 发送HTTP请求

通过TCP套接字，客户端向Web服务器发送一个文本的请求报文，一个请求报文由请求行、请求头部、空行和请求数据4部分组成。

#### 3. 服务器接受请求并返回HTTP响应

Web服务器解析请求，定位请求资源。服务器将资源复本写到TCP套接字，由客户端读取。一个响应由状态行、响应头部、空行和响应数据4部分组成。

#### 4. 释放连接TCP连接

若connection 模式为close，则服务器主动关闭TCP连接，客户端被动关闭连接，释放TCP连接;若connection 模式为keepalive，则该连接会保持一段时间，在该时间内可以继续接收请求;

#### 5. 客户端浏览器解析HTML内容

客户端浏览器首先解析状态行，查看表明请求是否成功的状态代码。然后解析每一个响应头，响应头告知以下为若干字节的HTML文档和文档的字符集。客户端浏览器读取响应数据HTML，根据HTML的语法对其进行格式化，并在浏览器窗口中显示。





## 3. Request 和 Response

### 3.1 Request



### 3.2 Response





## 4. XML和XPATH

### 4.1 XML

用正则处理HTML文档很麻烦，我们可以先将 `HTML`文件 转换成`XML`文档，然后用 `XPath` 查找 HTML 节点或元素。

- XML 指可扩展标记语言（EXtensible Markup Language）
- XML 是一种标记语言，很类似 HTML
- XML 的设计宗旨是传输数据，而非显示数据
- XML 的标签需要我们自行定义。
- XML 被设计为具有自我描述性。
- XML 是 W3C 的推荐标准。

```xml
<?xml version="1.0" encoding="utf-8"?>
<bookstore> 
  <book category="cooking"> 
    <title lang="en">Everyday Italian</title>  
    <author>Giada De Laurentiis</author>  
    <year>2005</year>  
    <price>30.00</price> 
  </book>  

  <book category="children"> 
    <title lang="en">Harry Potter</title>  
    <author>J K. Rowling</author>  
    <year>2005</year>  
    <price>29.99</price> 
  </book>  

  <book category="web"> 
    <title lang="en">XQuery Kick Start</title>  
    <author>James McGovern</author>  
    <author>Per Bothner</author>  
    <author>Kurt Cagle</author>  
    <author>James Linn</author>  
    <author>Vaidyanathan Nagarajan</author>  
    <year>2003</year>  
    <price>49.99</price> 
  </book> 

  <book category="web" cover="paperback"> 
    <title lang="en">Learning XML</title>  
    <author>Erik T. Ray</author>  
    <year>2003</year>  
    <price>39.95</price> 
  </book> 

</bookstore>
```



### 4.2 XPATH

XPath (XML Path Language) 是一门在 XML 文档中查找信息的语言，可用来在 XML 文档中对元素和属性进行遍历。

> chrome插件: XPATH Helper,  Firefox 插件: XPATH Checker

**XPATH语法**

最常用的路径表达式：

![img](https://images2017.cnblogs.com/blog/1299879/201802/1299879-20180218145525827-1393101627.png)

![img](https://images2017.cnblogs.com/blog/1299879/201802/1299879-20180218145816155-2047958085.png)

**谓语**

谓语用来查找某个特定的节点或者包含某个指定的值的节点，被嵌在方括号中。

在下面的表格中，我们列出了带有谓语的一些路径表达式，以及表达式的结果：

![img](https://images2017.cnblogs.com/blog/1299879/201802/1299879-20180218145913546-1108037451.png)

**选取位置节点**

![img](https://images2017.cnblogs.com/blog/1299879/201802/1299879-20180218145952515-7193813.png)

**选取若干路劲**

![img](https://images2017.cnblogs.com/blog/1299879/201802/1299879-20180218150112343-2081826050.png)





## 4. lxml库

安装：`pip install lxml`

lxml 是 一个HTML/XML的解析器，主要的功能是如何解析和提取 HTML/XML 数据。

lxml和正则一样，也是用 C 实现的，是一款高性能的 Python HTML/XML 解析器，可以利用XPath语法，来快速的定位特定元素以及节点信息。

 ```python
 #!/usr/bin/env python
 # -*- coding:utf-8 -*-
 
 from lxml import etree
 
 text = '''
     <div>
         <li>11</li>
         <li>22</li>
         <li>33</li>
         <li>44</li>
     </div>
 '''
 
 #利用etree.HTML，将字符串解析为HTML文档
 html = etree.HTML(text)
 
 # 按字符串序列化HTML文档
 result = etree.tostring(html)
 
 print(result)
 ```





## 5. URLLIB

https://www.runoob.com/python3/python-urllib.html

https://www.jb51.net/article/207790.htm



## 6. 爬取南航官网

[NUAA官网](https://nuaa.edu.cn/)






```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import urllib
import urllib.request
from lxml import etree

def loadPage(url):
    """
        作用：根据url发送请求，获取服务器响应文件
        url: 需要爬取的url地址
    """
    my_request = urllib.request.Request(url)
    html = urllib.request.urlopen(my_request).read()
    # 解析HTML文档为HTML DOM模型
    content = etree.HTML(html)
    print(content)
    # 返回所有匹配成功的列表集合
    title_list = content.xpath("/html/body/div[@id='col']/div[@class='inner']/div[@class='col clearfix']/div[@class='colinfo fr']/div[2]/div[@class='col_news_list']/div[@id='wp_news_w24']/ul[@class='wp_article_list']/li[*]/div[@class='fields pr_fields']/span[@class='Article_Title']/a")
    
    href_list = content.xpath("/html/body/div[@id='col']/div[@class='inner']/div[@class='col clearfix']/div[@class='colinfo fr']/div[2]/div[@class='col_news_list']/div[@id='wp_news_w24']/ul[@class='wp_article_list']/li[*]/div[@class='fields pr_fields']/span[@class='Article_Title']/a/@href")
    
    output = ""
    for i,title in enumerate(title_list):
        href = href_list[i]
        title_text =  title.text
        title_text='<a href="http://newsweb.nuaa.edu.cn'+href+'">'+title_text+'</a>'
        output = output+str(i+1)+" : "+title_text
        output = output+"\n\n"
    return output

news_text = loadPage("http://newsweb.nuaa.edu.cn/zhyw1/list.htm")
wx = WeChat()
wx.send_data(news_text)
```





![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/gV3Sp.jpg)

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import requests
import json

class WeChat:
    def __init__(self):
        self.CORPID = 'ww618207d6305f2535'
        self.CORPSECRET = 'xxxxxxx'
        self.AGENTID = '1000002'
        self.TOUSER = "zhaosheng"

    def _get_access_token(self):
        url = 'https://qyapi.weixin.qq.com/cgi-bin/gettoken'
        values = {'corpid': self.CORPID,
                  'corpsecret': self.CORPSECRET,
                  }
        req = requests.post(url, params=values)
        data = json.loads(req.text)
        return data["access_token"]

    def get_access_token(self):
        try:
            with open('./tmp/access_token.conf', 'r') as f:
                t, access_token = f.read().split()
        except:
            with open('./tmp/access_token.conf', 'w') as f:
                access_token = self._get_access_token()
                cur_time = time.time()
                f.write('\t'.join([str(cur_time), access_token]))
                return access_token
        else:
            cur_time = time.time()
            if 0 < cur_time - float(t) < 7260:
                return access_token
            else:
                with open('./tmp/access_token.conf', 'w') as f:
                    access_token = self._get_access_token()
                    f.write('\t'.join([str(cur_time), access_token]))
                    return access_token

    def send_data(self, message):
        send_url = 'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=' + self.get_access_token()
        send_values = {
            "touser": self.TOUSER,
            "msgtype": "text",
            "agentid": self.AGENTID,
            "text": {
                "content": message
                },
            "safe": "0"
            }
        send_msges=(bytes(json.dumps(send_values), 'utf-8'))
        respone = requests.post(send_url, send_msges)
        respone = respone.json()
        return respone["errmsg"]

if __name__ == '__main__':
    wx = WeChat()
    wx.send_data("IINT")
    wx.send_data("NUAA")
```





## 其他常用工具

https://zhuanlan.zhihu.com/p/81380459
