```python
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr,formataddr
import smtplib
def _format_addr(s):
    name,addr=parseaddr(s)
    return formataddr((Header(name,'utf-8').encode(),addr))
sender_addr='your_email@163.com' #改成你自己邮箱
password='your_email'#改成你自己的密码
to_addr='your_email@163.com'#你要发送的地址
smtp_server='smtp.163.com'#这个要是用网易的邮箱就不用改
 
msg=MIMEText('车票信息','plain','utf-8')
msg['From']=_format_addr('<%s>'% sender_addr)
msg['To']=_format_addr('<%s>'%to_addr)
msg['Subject']=Header('test','utf-8').encode()
 
server=smtplib.SMTP(smtp_server,25)
server.login(sender_addr,password)
server.sendmail(sender_addr,[to_addr],msg.as_string())
server.quit()
```

