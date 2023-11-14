# Shell参数扩展

什么是参数扩展：

通过符号`$`获得参数中存储的值。



最简单的形式：

`$parameter`或者`${parameter}`



子串拓展(Substring Expansion)

`${parameter:offset:length}`

`${parameter:offset}`

子串扩展的意思是从offset位置开始截取长度为length的子串，如果没有提供length则是从开始到结尾。

注意：

1. 如果offset是负数，开始位置从字符串末尾开始，然后取长度为length。
2. 如果length是负数，那么length含义不再是代表字符串长度，而是另一个offset，位置从字符串末尾开始，拓展结果是offset～length之间的字符串。
3. 如果parameter是@，也就是所有位置参数时，offset必须从1开始。
4. 当offset是负值的时候，负号和冒号之间必须有空格，这是为了避免和`${parameter:-word}`混淆。



## 查找和替换

`${parameter/pattern/string}`

`${parameter//pattern/string}`

`${parameter/pattern}`

`${parameter//pattern}`

匹配后的字符串会用string替换掉

注意：

1. parameter之后如果是`/`，则只替换到匹配的第一个字符串，如果是`//`，则替换掉所有匹配到的字符串。
2. 当string为空时，则相当于匹配的字符串都删除。
3. 特殊符号`#`和`%`在这种情况下分别锚定Anchoring字符串的开始和结尾。
4. 如果bash的nocasematch选项参数时打开的s hopt-s nocasematch，则匹配过程大小写是不敏感的。



## 查找并删除

`${parameter#pattern}`

`${parameter##pattern}`

`${parameter%pattern}`

`${parameter%%pattern}`

删除匹配到的字符串

```
file=/dir1/dir2/dir3/my.file.txt
```

```shell
${file#*/} : 删除第一个 / 及其左边的字符串: dir1/dir2/dir3/my.file.txt
${file##*/} : 删除最有一个 / 及其左边的字符串：my.file.txt，相当于 basename ${file}
${file#*.} : 删除第一个 . 及其左边的字符串: file.txt
${file##*.} : 删除最后一个 . 及其左边的字符串: txt
${file%/*} : 删除最有一个 / 及其右边的字符串: /dir1/dir2/dir3，相当于 dirname ${file}
${file%%/*} : 删除第一个 / 及其右边的字符串：空值
${file%.*} : 删除最后一个 . 及其右边的字符串： /dir1/dir2/dir3/my.file
${file%%.*} : 删除第一个 . 及其右边的字符串： /dir1/dir2/dir3/my
```

记忆方法为：

1. #是去掉左边
2. %是去掉右边
3. 单一符号是最小匹配；两个符号是最大匹配。



## 获取参数值长度(Parameter length)

`${#parameter}`

这个拓展很简单，就是返回`parameter`值的长度值。



## 大小写转换(Case modification)

```shell
${parameter^}
${parameter^^}
${parameter,}
${parameter,,}
```

字符`^`的意思是将第一个字符转换成大写字母，`^^`的意思是将所有的字符转化为大写字母。

字符`,`的意思是将第一个字符转换成小写字母，`,,`的意思是将所有的字符转化为小写字母。