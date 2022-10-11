## 正则元字符
在说正则表达式之前我们先来看看通配符，我想通配符大家都用过。通配符主要有星号(\*)和问号(?)，用来模糊搜索文件。winodws中我们常会使用搜索来查找一些文件。如:`*.jpg`，`XXX.docx`的方式，来快速查找文件。其实正则表达式和我们通配符很相似也是通过特定的字符匹配我们所要查询的内容信息。已下代码都是区分大小写。

### 常用元字符

| **代码** | **说明**                                                     |
| -------- | ------------------------------------------------------------ |
| .        | 匹配除换行符以外的任意字符。                                 |
| \w       | 匹配字母或数字或下划线或汉字。                               |
| \s       | 匹配任意的空白符。                                           |
| \d       | 匹配数字。                                                   |
| \b       | 匹配单词的开始或结束。                                       |
| [ck]     | 匹配包含括号内元素的字符                                     |
| ^        | 匹配行的开始。                                               |
| $        | 匹配行的结束。                                               |
| \        | 对下一个字符转义。比如\$是个特殊的字符。要匹配\$的话就得用\$ |
| \|       | 分支条件，如：x\|y匹配 x 或 y。                              |

### 反义元字符

| **代码** | **说明**                                          |
| -------- | ------------------------------------------------- |
| \W       | 匹配任意不是字母，数字，下划线，汉字的字符。      |
| \S       | 匹配任意不是空白符的字符。等价于 [^ \f\n\r\t\v]。 |
| \D       | 匹配任意非数字的字符。等价于 [^0-9]。             |
| \B       | 匹配不是单词开头或结束的位置。                    |
| [^CK]    | 匹配除了CK以外的任意字符。                        |

### 特殊元字符

| **代码** | **说明**                                 |
| -------- | ---------------------------------------- |
| \f       | 匹配一个换页符。等价于 \x0c 和 \cL。     |
| \n       | 匹配一个换行符。等价于 \x0a 和 \cJ。     |
| \r       | 匹配一个回车符。等价于 \x0d 和 \cM。     |
| \t       | 匹配一个制表符。等价于 \x09 和 \cI。     |
| \v       | 匹配一个垂直制表符。等价于 \x0b 和 \cK。 |

### 限定符

| **代码** | **说明**                                                     |
| -------- | ------------------------------------------------------------ |
| *        | 匹配前面的子表达式零次或多次。                               |
| +        | 匹配前面的子表达式一次或多次。                               |
| ?        | 匹配前面的子表达式零次或一次。                               |
| {n}      | n 是一个非负整数。匹配确定的 n 次。                          |
| {n,}     | n 是一个非负整数。至少匹配n 次。                             |
| {n,m}    | m 和 n 均为非负整数，其中n <= m。最少匹配 n 次且最多匹配 m 次。 |

### 懒惰限定符

| **代码** | **说明**                                                     |
| -------- | ------------------------------------------------------------ |
| *?       | 重复任意次，但尽可能少重复。如 "acbacb"  正则  "a.*?b" 只会取到第一个"acb" 原本可以全部取到但加了限定符后，只会匹配尽可能少的字符 ，而"acbacb"最少字符的结果就是"acb" 。 |
| +?       | 重复1次或更多次，但尽可能少重复。与上面一样，只是至少要重复1次。 |
| ??       | 重复0次或1次，但尽可能少重复。如 "aaacb" 正则 "a.??b" 只会取到最后的三个字符"acb"。 |
| {n,m}?   | 重复n到m次，但尽可能少重复。如 "aaaaaaaa"  正则 "a{0,m}" 因为最少是0次所以取到结果为空。 |
| {n,}?    | 重复n次以上，但尽可能少重复。如 "aaaaaaa"  正则 "a{1,}" 最少是1次所以取到结果为 "a"。 |

### 捕获分组

| **代码**     | **说明**                                                     |
| ------------ | ------------------------------------------------------------ |
| (exp)        | 匹配exp,并捕获文本到自动命名的组里。                         |
| (?<name>exp) | 匹配exp,并捕获文本到名称为name的组里。                       |
| (?:exp)      | 匹配exp,不捕获匹配的文本，也不给此分组分配组号以下为零宽断言。 |
| (?=exp)      | 匹配exp前面的位置。如 "How are you doing" 正则"(?<txt>.+(?=ing))" 这里取ing前所有的字符，并定义了一个捕获分组名字为 "txt" 而"txt"这个组里的值为"How are you do"; |
| (?<=exp)     | 匹配exp后面的位置。如 "How are you doing" 正则"(?<txt>(?<=How).+)" 这里取"How"之后所有的字符，并定义了一个捕获分组名字为 "txt" 而"txt"这个组里的值为" are you doing"; |
| (?!exp)      | 匹配后面跟的不是exp的位置。如 "123abc" 正则 "\d{3}(?!\d)"匹配3位数字后非数字的结果 |
| (?<!exp)     | 匹配前面不是exp的位置。如 "abc123 " 正则 "(?<![0-9])123" 匹配"123"前面是非数字的结果也可写成"(?!<\d)123" |



## 小试牛刀

在C#中使用正则表达式主要是通过Regex类来实现。命名空间：

```csharp
using System.Text.RegularExpressions
```

**其中常用方法**：

| **名称**                                                     | **说明**                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [IsMatch(String, String)](https://msdn.microsoft.com/zh-cn/library/sdx2bds0(v=vs.110).aspx) | 指示 Regex 构造函数中指定的正则表达式在指定的输入字符串中是否找到了匹配项。 |
| [Match(String, String)](https://msdn.microsoft.com/zh-cn/library/0z2heewz(v=vs.110).aspx) | 在指定的输入字符串中搜索 Regex 构造函数中指定的正则表达式的第一个匹配项。 |
| [Matches(String, String)](https://msdn.microsoft.com/zh-cn/library/b9712a7w(v=vs.110).aspx) | 在指定的输入字符串中搜索正则表达式的所有匹配项。             |
| [Replace(String, String)](https://msdn.microsoft.com/zh-cn/library/vstudio/xwewhkd1(v=vs.100).aspx) | 在指定的输入字符串内，使用指定的替换字符串替换与某个正则表达式模式匹配的所有字符串。 |
| [Split(String, String)](https://msdn.microsoft.com/zh-cn/library/8yttk7sy(v=vs.110).aspx) | 在由 Regex 构造函数指定的正则表达式模式所定义的位置，拆分指定的输入字符串。 |

在使用正则表达式前我们先来看看“@”符号的使用。

学过C#的人都知道C# 中**字符串常量**可以以`@ `开头声名，这样的优点是转义序列“不”被处理，按“原样”输出，即我们不需要对转义字符加上 \ （反斜扛），就可以轻松coding。如:

```csharp
string filePath = @"c:\Docs\Source\CK.txt" // rather than "c:\\Docs\\Source\\CK.txt"
```

如要在一个用 @ 引起来的字符串中包括一个双引号，就需要使用两对双引号了。这时候你不能使用 \ 来转义双引号了，因为在这里 \ 的转义用途已经被 @ “屏蔽”掉了。如:

```csharp
string str=@"""Ahoy!"" cried the captain."  // 输出为： "Ahoy!" cried the captain. 
```

### 字符串匹配

在实际项目中我们常常需要对用户输入的信息进行验证。如：匹配用户输入的内容是否为数字，是否为有效的手机号码，邮箱是否合法....等。

实例代码：

```csharp
string RegexStr = string.Empty;
RegexStr = "^[0-9]+$"; //匹配字符串开始和结束是否为0-9的数字【定位字符】
Console.WriteLine("判断'R1123'是否为数字:{0}", Regex.IsMatch("R1123", RegexStr));
Console.WriteLine("判断'1123'是否为数字:{0}", Regex.IsMatch("1123", RegexStr));

RegexStr = @"\d+";  //匹配字符串中间是否包含数字(这里没有从开始进行匹配噢,任意位子只要有一个数字即可)
Console.WriteLine("'R1123'是否包含数字:{0}", Regex.IsMatch("R1123", RegexStr));
Console.WriteLine("'博客园'是否包含数字:{0}", Regex.IsMatch("博客园", RegexStr));

RegexStr = @"^Hello World[\w\W]*";  //已Hello World开头的任意字符(\w\W：组合可匹配任意字符)
Console.WriteLine("'HeLLO WORLD xx hh xx'是否已Hello World开头:{0}", Regex.IsMatch("HeLLO WORLD xx hh xx", RegexStr, RegexOptions.IgnoreCase));
Console.WriteLine("'LLO WORLD xx hh xx'是否已Hello World开头:{0}", Regex.IsMatch("LLO WORLD xx hh xx", RegexStr,RegexOptions.IgnoreCase));
//RegexOptions.IgnoreCase：指定不区分大小写的匹配。
```



### 字符串查找

```csharp
string RegexStr = string.Empty;
string LinkA = "<a href=\"http://www.baidu.com\" target=\"_blank\">百度</a>";
RegexStr = @"href=""[\s]+""";
Match mt= Regex.Match(LinkA, RegexStr);

Console.WriteLine("{0}.",LinkA);
Console.WriteLine("获得href中的值:{0}.",mt.Value);

RegexStr = @"<h[^23456]>[\S]+<h[1]>";    //<h[^23456]>:匹配h除了2,3,4,5,6之中的值,<h[1]>:h匹配包含括号内元素的字符
Console.WriteLine("{0}。GetH1值：{1}", "<H1>标题<H1>", Regex.Match("<H1>标题<H1>", RegexStr, RegexOptions.IgnoreCase).Value);
Console.WriteLine("{0}。GetH1值：{1}", "<h2>小标<h2>", Regex.Match("<h2>小标<h2>", RegexStr, RegexOptions.IgnoreCase).Value);
//RegexOptions.IgnoreCase:指定不区分大小写的匹配。

RegexStr = @"ab\w+|ij\w{1,}";   //匹配ab和字母 或 ij和字母
Console.WriteLine("{0}。多选结构：{1}", "abcd", Regex.Match("abcd", RegexStr).Value);
Console.WriteLine("{0}。多选结构：{1}", "efgh", Regex.Match("efgh", RegexStr).Value);
Console.WriteLine("{0}。多选结构：{1}", "ijk", Regex.Match("ijk", RegexStr).Value);

RegexStr = @"张三?丰";    //?匹配前面的子表达式零次或一次。
Console.WriteLine("{0}。可选项元素：{1}", "张三丰", Regex.Match("张三丰", RegexStr).Value);
Console.WriteLine("{0}。可选项元素：{1}", "张丰", Regex.Match("张丰", RegexStr).Value);
Console.WriteLine("{0}。可选项元素：{1}", "张飞", Regex.Match("张飞", RegexStr).Value);

/* 
 例如：
July|Jul　　可缩短为　　July?
4th|4　　   可缩短为　　4(th)?
*/

//匹配特殊字符
RegexStr = @"Asp\.net";    //匹配Asp.net字符，因为.是元字符他会匹配除换行符以外的任意字符。这里我们只需要他匹配.字符即可。所以需要转义\.这样表示匹配.字符
Console.WriteLine("{0}。匹配Asp.net字符：{1}", "Java Asp.net SQLServer", Regex.Match("Java Asp.net SQLServer", RegexStr).Value);
Console.WriteLine("{0}。匹配Asp.net字符：{1}", "C# Java", Regex.Match("C# Java", RegexStr).Value);
```

### 贪婪与懒惰

```csharp
string f = "fooot";

//贪婪匹配
RegexStr = @"f[o]+";
Match m1 = Regex.Match(f, RegexStr);
Console.WriteLine("{0}贪婪匹配（尽可能匹配多的字符）：{1}",f,m1.ToString());

//懒惰匹配
RegexStr = @"f[o]+?";
Match m2 = Regex.Match(f,RegexStr);
Console.WriteLine("{0}懒惰匹配（匹配尽可能少重复）：{1}",f,m2.ToString());
```



###  (exp)分组

在做爬虫时我们经常获得A中一些有用信息。如href,title和显示内容等。

```csharp
string TaobaoLink = "<a href=\"http://www.taobao.com\" title=\"淘宝网 - 淘！我喜欢\" target=\"_blank\">淘宝</a>";
RegexStr = @"<a[^>]+href=""(\S+)""[^>]+title=""([\s\S]+?)""[^>]+>(\S+)</a>";
Match mat = Regex.Match(TaobaoLink, RegexStr);
for (int i=0;i<mat.Groups.Count;i++)
{
    Console.WriteLine("第"+i+"组："+mat.Groups[i].Value);
}
```

在正则表达式里使用`()`包含的文本自动会命名为一个组。上面的表达式中共使用了4个`()`可以认为是分为了4组。

**输出结果共分为：4组。**

1. 为我们所匹配的字符串。
2. 是我们第一个括号[href=""(\S+)""]中(\S+)所匹配的网址信息。内容为：http://www.taobao.com。
3. 是第二个括号[title=""([\s\S]+?)""]中所匹配的内容信息。内容为：淘宝网 - 淘！我喜欢。这里我们会看到+?懒惰限定符。title=""([\s\S]+?)""  这里+?的下一个字符为"双引号，"双引号在匹配字符串后面还有三个。+?懒惰限定符会尽可能少重复，所他会匹配最前面那个"双引号。如果我们不使用+?懒惰限定符他会匹配到：淘宝网 - 淘！我喜欢" target= 会尽可能多重复匹配。
4. 是第三个括号[(\S+)]所匹配的内容信息。内容为：淘宝。

> 说明：反义元字符所对应的元字符都能组合匹配任意字符。如:[\w\W],[\s\S],[\d\D]



###  (?<name>exp) 分组取名

当我们匹配分组信息过多后，在某种场合只需取当中某几组信息。这时我们可以对分组取名。通过分组名称来快速提取对应信息。

```csharp
string Resume =  "基本信息姓名:CK|求职意向:.NET软件工程师|性别:男|学历:本专|出生日期:1988-08-08|户籍:湖北.孝感|E - Mail:9245162@qq.com|手机:15000000000";
RegexStr = @"姓名:(?<name>[\S]+)\|\S+性别:(?<sex>[\S]{1})\|学历:(?<xueli>[\S]{1,10})\|出生日期:(?<Birth>[\S]{10})\|[\s\S]+手机:(?<phone>[\d]{11})";
Match matc = Regex.Match(Resume, RegexStr);
Console.WriteLine("姓名：{0},手机号：{1}", matc.Groups["name"].ToString(), matc.Groups["phone"].ToString());
```

通过(?<name>exp)可以很轻易为分组取名。然后通过Groups["name"]取得分组值。



### 获得页面中A标签中href值

```csharp
string PageInfo = @"<hteml>
                        <div id=""div1"">
                            <a href=""http://www.baidu.con"" target=""_blank"">百度</a>
                            <a href=""http://www.taobao.con"" target=""_blank"">淘宝</a>
                            <a href=""http://www.cnblogs.com"" target=""_blank"">博客园</a>
                            <a href=""http://www.google.con"" target=""_blank"">google</a>
                        </div>
                        <div id=""div2"">
                            <a href=""/zufang/"">整租</a>
                            <a href=""/hezu/"">合租</a>
                            <a href=""/qiuzu/"">求租</a>
                            <a href=""/ershoufang/"">二手房</a>
                            <a href=""/shangpucz/"">商铺出租</a>
                        </div>
                    </hteml>";
RegexStr = @"<a[^>]+href=""(?<href>[\S]+?)""[^>]*>(?<text>[\S]+?)</a>";
MatchCollection mc = Regex.Matches(PageInfo, RegexStr);
foreach (Match item in mc)
{
    Console.WriteLine("href:{0}--->text:{1}",item.Groups["href"].ToString(),item.Groups["text"].ToString());
}
```



### Replace 替换字符串

用户在输入信息时偶尔会包含一些敏感词，这时我们需要替换这个敏感词。

```csharp
string PageInputStr = "靠.TMMD,今天真不爽....";
RegexStr = @"靠|TMMD|妈的";
Regex rep_regex = new Regex(RegexS)
```

