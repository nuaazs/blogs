获取所有商品id

```python
$$

import pymysql
connection = pymysql.connect(host = '{mysql_server_new}',
                             user = '{mysql_user_new}',
                             password = '{mysql_pd_new}',
                             db = '{mysql_db_new}',
                             port = {mysql_port_new}
                             )


cur = connection.cursor()
cur.execute('use {mysql_db_new};')


sql =  "SELECT ID from ysy_word_v2;";
cur.execute(sql)
rows=cur.fetchall()
#写文件
f = open(r"c:/e7/ids_list.txt",'w')
for id in rows:
    f.writelines(str(id[0]))
    f.writelines("\n")


#关闭数据库连接
f.close()
connection.close()
```

