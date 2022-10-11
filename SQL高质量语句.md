[![数据蛙datafrog](https://pic2.zhimg.com/v2-93cff0ac34189461b8f475b54bb1d28c_xs.jpg?source=1940ef5c)](https://www.zhihu.com/people/datafrog)

[数据蛙datafrog](https://www.zhihu.com/people/datafrog)

公众号：数据蛙DataFrog

24 人赞同了该回答

最近有同学面试的时候被问到，公司里的数据量比较大，并不是练习的时候数据量比较小可以随便来写，那么在写SQL时候应该注意哪些问题，可以提高查询效率呢?数据量大的情况下，不同的SQL语句，消耗的时间相差很大。按下面方法可以提高查询的效果。温馨提示，文章需要有**索引知识**的门槛，缺少的同学右拐学习

[凡人求索：索引(一)Mysql创建索引zhuanlan.zhihu.com![图标](https://pic1.zhimg.com/zhihu-card-default_ipico.jpg)](https://zhuanlan.zhihu.com/p/76926803)

### 一:参数是子查询时，使用 EXISTS 代替 IN

如果 **IN** 的参数是**1, 2, 3** **这样的数值列表，一般还不需要特别注意。但是如果参数是子查询，那么就需要注意了。在大多时候，\[NOT\] IN 和 \[NOT\] EXISTS** 返回的结果是相同的。但是两者用于子查询时，**EXISTS** 的速度会更快一些。我们试着从 **Class\_A** 表中查出同时存在于 **Class\_B** 表中的员工。下面两条**SQL** **语句返回的结果是一样的，但是使用** **EXISTS** 的 **SQL**语句更快一些。

![](https://pic2.zhimg.com/50/v2-c79a938578ea141164855ed035f8fa18_hd.jpg?source=1940ef5c)

![](https://pic2.zhimg.com/80/v2-c79a938578ea141164855ed035f8fa18_720w.jpg?source=1940ef5c)

    -- 慢
    SELECT *
    FROM Class_A
    WHERE id IN (SELECT id
    FROM Class_B);


​    
    -- 快
    SELECT *
    FROM Class_A A
    WHERE EXISTS
    (SELECT *
    FROM Class_B B
    WHERE A.id = B.id);

使用 EXISTS 时更快的原因有以下两个。

*   如果连接列（id ）上建立了索引，那么查询 Class\_B 时不用查实际的表，只需查索引就可以了。
*   如果使用 EXISTS ，那么只要查到一行数据满足条件就会终止查询，不用像使用 IN 时一样扫描全表。在这一点上 NOT EXISTS 也一样。

当 IN 的参数是子查询时，数据库首先会执行子查询，然后将结果存储在一张临时的工作表里（内联视图），然后扫描整个视图。很多情况下这种做法都非常耗费资源。使用 EXISTS 的话，数据库不会生成临时的工作表。

要想改善 IN 的性能，除了使用 EXISTS ，还可以使用连接。前面的查询语句就可以像下面这样“扁平化”。

    -- 使用连接代替IN
    SELECT A.id, A.name
    FROM Class_A A INNER JOIN Class_B B
    ON A.id = B.id;

这种写法至少能用到一张表的“id”列上的索引。而且，因为没有了子查询，所以数据库也不会生成中间表。我们很难说与 EXISTS 相比哪个更好，但是如果没有索引，那么与连接相比，可能 EXISTS 会略胜一筹。

### 二:避免排序

我们在查询的时候，虽然我们没有想要进行排序，但是在数据库内部频繁地进行着暗中的排序。因此对于我们来说，了解都有哪些运算会进行排序很有必要，会进行排序的代表性的运算有下面这些

*   group by 子句
*   order by 子句
*   聚合函数(sum、count、avg、max、min)
*   distinct
*   集合运算符(union、intersect、except)
*   窗口函数(rank、row\_number等)

**1.使用union all 代替union**

    select * from Class_A
    union 
    select * from Class_B

这个会进行排序,如果不在乎结果中是否有重复数据，可以使用union all 代替 union .这样就不会进行排序了

    select * from Class_A
    union all
    select * from Class_B;

**2.使用exists 代替distinct**

为了排除重复数据,distinct 也会进行排序。如果需要对两张表的连接结果进行去重，可以考虑使用exists代替distinct,以避免排序

**Items**

![](https://pic1.zhimg.com/50/v2-4ee3f7ddd269c1dbfffd5914354b5fc9_hd.jpg?source=1940ef5c)

![](https://pic1.zhimg.com/80/v2-4ee3f7ddd269c1dbfffd5914354b5fc9_720w.jpg?source=1940ef5c)

**SalesHistory**

![](https://pic4.zhimg.com/50/v2-57a678010c9640c9dd33648c0bd8d46a_hd.jpg?source=1940ef5c)

![](https://pic4.zhimg.com/80/v2-57a678010c9640c9dd33648c0bd8d46a_720w.jpg?source=1940ef5c)

问题:如何从上面的商品表**Items**中找出同时存在于销售记录表**SalesHistory**中的商品。简而言之，就是找出有销售记录的商品,使用 IN 是一种做法。但是前面我们说过，当 IN 的参数是子查询时，使用连接要比使用 IN 更好。因此我们像下面这样使用**item\_no**列对两张表进行连接。

    SELECT I.item_no
    FROM Items I INNER JOIN SalesHistory SH
    ON I. item_no = SH. item_no;

因为是一对多的连接，所以**item\_no**列中会出现重复数据。为了排除重复数据，我们需要使用 DISTINCT 。

    SELECT DISTINCT I.item_no
    FROM Items I INNER JOIN SalesHistory SH
    ON I. item_no = SH. item_no;

但是，使用distinct的时候会进行排序， 其实更好的做法是使用 EXISTS 。

    SELECT item_no
    FROM Items I
    WHERE EXISTS
    (SELECT *
    FROM SalesHistory SH
    WHERE I.item_no = SH.item_no)

这条语句在执行过程中不会进行排序。而且使用 EXISTS 和使用连接一样高效。

**3.在极值函数中使用索引(MAX/MIN)**

使用这两个函数时都会进行排序。但是如果参数字段上建有索引，则 只需要扫描索引，不需要扫描整张表。以刚才的表 Items 为例来说， SQL 语句可以像下面这样写。

    SELECT MAX(item_no)
    FROM Items;

这种方法并不是去掉了排序这一过程，而是优化了排序前的查找速 度，从而减弱排序对整体性能的影响。

**4.能写在 WHERE 子句里的条件不要写在 HAVING 子句里**

*   聚合后使用HAVING 子句过滤

    SELECT sale_date, SUM(quantity)
    FROM SalesHistory
    GROUP BY sale_date
    HAVING sale_date = '2007-10-01';

*   聚合前使用WHERE 子句过滤

    SELECT sale_date, SUM(quantity)
    FROM SalesHistory
    WHERE sale_date = '2007-10-01'
    GROUP BY sale_date;

虽然结果是一样的，但是从性能上来看，第二条语句写法效率更高。原因通常有两个。第一个是在使用 **GROUP BY** 子句聚合时会进行排序，如果事先通过**WHERE** 子句筛选出一部分行，就能够减轻排序的负担。第二个是在**WHERE** 子句的条件里可以使用索引。**HAVING** 子句是针对聚合后生成的视图进行筛选的，但是很多时候聚合后的视图都没有继承原表的索引结构 。

### 三:索引是真的用到了吗

以下都是索引失效的现象 **1.索引字段上进行计算**

    select * from SomeTable 
    whre col_1 * 1.1 >100;

这种索引就会失效,执行的时候会进行全表的扫描。优化的方法就是，把运算的表达式放到查询条件的右侧

    select * from SomeTable 
        whre col_1 >100 / 1.1;

其实只要索引列上使用函数的时候，索引列就会失效

    select * from SomeTable
    where SUBTR(col_1,1,1)='a'

**2.使用 IS NULL 谓词** 通常，索引字段是不存在 NULL 的，所以指定 IS NULL 和 IS NOT NULL 的话会使得索引无法使用，进而导致查询性能低下。

    select * from SomeTable 
    where col_1 is null;

**3.使用否定形式**

下面的几种否定形式也不能用到索引

*   <>
*   !=
*   NOT IN

    select * from SomeTable 
    where col_1 <> 100;

**4.使用OR** 在 col\_1 和 col\_2 上分别建立了不同的索引，或者建立了（col\_1,col\_2 ）这样的联合索引时，如果使用 OR 连接条件，那么要么用不到索引，要么用到了但是效率比 AND 要差很多。

    SELECT *
    FROM SomeTable
    WHERE col_1 > 100
    OR col_2 = 'abc';

**5.使用联合索引时，列的顺序错误**

假设存在这样顺序的一个联合索引**col\_1, col\_2, col\_3** **。联合索引中的第一列col\_1**必须写在查询条件的开头，而且索引中列的顺序不能颠倒。如果无法保证查询条件里列的顺序与索引一致，可以考虑将联合索引 拆分为多个索引。

    ○ SELECT * FROM SomeTable WHERE col_1 = 10 AND col_2 = 100 AND col_3 = 500;
    ○ SELECT * FROM SomeTable WHERE col_1 = 10 AND col_2 = 100 ;
    × SELECT * FROM SomeTable WHERE col_1 = 10 AND col_3 = 500 ;
    × SELECT * FROM SomeTable WHERE col_2 = 100 AND col_3 = 500 ;
    × SELECT * FROM SomeTable WHERE col_2 = 100 AND col_1 = 10 ;

**6.使用 LIKE 谓词进行后方一致或中间一致的匹配**

    × SELECT * FROM SomeTable WHERE col_1 LIKE '%a';
    × SELECT * FROM SomeTable WHERE col_1 LIKE '%a%';
    ○ SELECT * FROM SomeTable WHERE col_1 LIKE 'a%';

**7.进行默认的类型转换**

    × SELECT * FROM SomeTable WHERE col_1 = 10;
    ○ SELECT * FROM SomeTable WHERE col_1 = '10';
    ○ SELECT * FROM SomeTable WHERE col_1 = CAST(10, AS CHAR(2));

默认的类型转换不仅会增加额外的性能开销，还会导致索引不可用，可以说是有百害而无一利。虽然这样写还不至于出错，但还是不要嫌麻烦，在需要类型转换时显式地进行类型转换吧（别忘了转换要写在条件表达式的右边）。

### 四:减少中间表

在 SQL 中，子查询的结果会被看成一张新表，这张新表与原始表一样，可以通过代码进行操作。这种高度的相似性使得 SQL 编程具有非常强的灵活性，但是如果不加限制地大量使用中间表，会导致查询性能下降。频繁使用中间表会带来两个问题，**一是展开数据需要耗费内存资源**，**二是原始表中的索引不容易使用到（特别是聚合时）**。因此，尽量减 少中间表的使用也是提升性能的一个重要方法。 **1.灵活使用 HAVING 子句** 对聚合结果指定筛选条件时，使用 HAVING 子句是基本原则。不习惯使用 HAVING 子句的数据库工程师可能会倾向于像下面这样先生成一张中间表，然后在 WHERE 子句中指定筛选条件。

    SELECT *
    FROM (SELECT sale_date, MAX(quantity) AS max_qty
    FROM SalesHistory
    GROUP BY sale_date) TMP ←----- 没用的中间表
    WHERE max_qty >= 10

然而，对聚合结果指定筛选条件时不需要专门生成中间表，像下面这样使用 HAVING 子句就可以。

    SELECT sale_date, MAX(quantity)
    FROM SalesHistory
    GROUP BY sale_date
    HAVING MAX(quantity) >= 10;

HAVING 子句和聚合操作是同时执行的，所以比起生成中间表后再执行的 WHERE 子句，效率会更高一些，而且代码看起来也更简洁。

    × SELECT * FROM SomeTable WHERE col_1 = 10;
    ○ SELECT * FROM SomeTable WHERE col_1 = '10';
    ○ SELECT * FROM SomeTable WHERE col_1 = CAST(10, AS CHAR(2));

建议大家写sql 的时候要留意下查询效率，工作前都是我们使用练习数据写sql,不要求性能方面，但是工作中数据大都是上千万的，sql查询性能不好的，运行起来有的需要半个多小时

**参考书籍**

[SQL进阶教程pan.baidu.com](https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/15KNYHtidkIx1PWjDuRsp0A)

[发布于 2020-06-19](https://www.zhihu.com/question/24460717/answer/1290828533)

赞同 241 条评论

分享

收藏喜欢



收起

继续浏览内容

![](https://pic4.zhimg.com/80/v2-88158afcff1e7f4b8b00a1ba81171b61_720w.png)

知乎

发现更大的世界

打开

![](https://picb.zhimg.com/80/v2-a448b133c0201b59631ccfa93cb650f3_1440w.png)

Chrome

继续