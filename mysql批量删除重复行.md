背景：Mysql中记录的主图链接,商品ID,更新日期。因为之前没注意，都是用的insert，所以导致同一个ID有多条记录，想要批量删除老的记录。

```mysql
delete p1 from 
	product_data as p1,product_data as p2 
		where p1.ID=p2.ID and p1.date < p2.date;
```

