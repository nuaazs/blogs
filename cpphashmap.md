## 1，map简介

`map` 是STL的一个关联容器，它提供一对一的`hash`。

第一个可以称为关键字(key)，每个关键字只能在map中出现一次；第二个可能称为该关键字的值(value)；

map以**模板(泛型)**方式实现，可以存储任意类型的数据，包括使用者自定义的数据类型。Map主要用于资料**一对一映射(one-to-one)**的情況，map內部的实现自建一颗**红黑树**，这颗树具有对数据**自动排序**的功能。在map内部所有的数据都是有序的，后边我们会见识到有序的好处。比如一个班级中，每个学生的学号跟他的姓名就存在著一对一映射的关系。

## 2，map的功能

自动建立`key－ value`的对应。key 和 value可以是任意你需要的类型。

## 3，使用map

使用map得包含map类所在的头文件

```cpp
#include <map>  //注意，STL头文件没有扩展名.h
```

map对象是模板类，需要关键字和存储对象两个模板参数：

```cpp
std:map<int, string> personnel;
```

这样就定义了一个用`int`作为索引,并拥有相关联的指向`string`的指针.

为了使用方便，可以对模板类进行一下类型定义，

```cpp
typedef map<int,CString> UDT_MAP_INT_CSTRING;
UDT_MAP_INT_CSTRING enumMap;
```

## 4，map的构造函数

map共提供了6个构造函数，这块涉及到内存分配器这些东西，略过不表，在下面我们将接触到一些map的构造方法，这里要说下的就是，我们通常用如下方法构造一个map：

```
map<int, string> mapStudent;
```

## 5，插入元素
```cpp
// 定义一个map对象
map<int, string> mapStudent;

// 第一种 用insert函數插入pair
mapStudent.insert(pair<int, string>(000, "student_zero"));

// 第二种 用insert函数插入value_type数据
mapStudent.insert(map<int, string>::value_type(001, "student_one"));

// 第三种 用"array"方式插入
mapStudent[123] = "student_first";
mapStudent[456] = "student_second";
```
以上三种用法，虽然都可以实现数据的插入，但是它们是有区别的，当然了第一种和第二种在效果上是完成一样的，用`insert`函数插入数据，在数据的 插入上涉及到**集合的唯一性**这个概念，即当map中有这个关键字时，insert操作是不能在插入数据的，但是用**数组**方式就不同了，它可以**覆盖**以前该关键字对应的值，用程序说明如下：
```cpp
mapStudent.insert(map<int, string>::value_type (001, "student_one"));

mapStudent.insert(map<int, string>::value_type (001, "student_two"));
```
上面这两条语句执行后，map中`001`这个关键字对应的值是“student_one”，第二条语句并没有生效，那么这就涉及到我们怎么知道insert语句是否插入成功的问题了，**可以用pair来获得是否插入成功**，程序如下
```cpp
// 构造定义，返回一个pair对象
pair<iterator,bool> insert (const value_type& val);

pair<map<int, string>::iterator, bool> Insert_Pair;

Insert_Pair = mapStudent.insert(map<int, string>::value_type (001, "student_one"));

if(!Insert_Pair.second)
    cout << ""Error insert new element" << endl;
```
我们通过pair的第二个变量来知道是否插入成功，它的第一个变量返回的是一个map的迭代器，如果插入成功的话```Insert_Pair.second```应该是`true`的，否则为`false`。

## 6， 查找元素

当所查找的关键key出现时，它返回数据所在对象的位置，如果沒有，返回`iter`与`end`函数的值相同。
```cpp
// find 返回迭代器指向当前查找元素的位置否则返回map::end()位置
iter = mapStudent.find("123");

if(iter != mapStudent.end())
    cout<<"Find, the value is"<<iter->second<<endl;
else
    cout<<"Do not Find"<<endl;
```
## 7， 刪除与清空元素
```cpp
//迭代器刪除
iter = mapStudent.find("123");
mapStudent.erase(iter);

//用关键字刪除
int n = mapStudent.erase("123"); //如果刪除了會返回1，否則返回0

//用迭代器范围刪除 : 把整个map清空
mapStudent.erase(mapStudent.begin(), mapStudent.end());
//等同于mapStudent.clear()
```
## 8，map的大小

在往map里面插入了数据，我们怎么知道当前已经插入了多少数据呢，可以用size函数，用法如下：
```cpp
int nSize = mapStudent.size();
```


##  9，map的基本操作函数：

C++ maps是一种关联式容器，包含“关键字/值”对
```cpp
     begin()         返回指向map头部的迭代器

     clear(）        删除所有元素

     count()         返回指定元素出现的次数

     empty()         如果map为空则返回true

     end()           返回指向map末尾的迭代器

     equal_range()   返回特殊条目的迭代器对

     erase()         删除一个元素

     find()          查找一个元素

     get_allocator() 返回map的配置器

     insert()        插入元素

     key_comp()      返回比较元素key的函数

     lower_bound()   返回键值>=给定元素的第一个位置

     max_size()      返回可以容纳的最大元素个数

     rbegin()        返回一个指向map尾部的逆向迭代器

     rend()          返回一个指向map头部的逆向迭代器

     size()          返回map中元素的个数

     swap()           交换两个map

     upper_bound()    返回键值>给定元素的第一个位置

     value_comp()     返回比较元素value的函数
```

# hash_map原理简介
这是一节让你深入理解`hash_map`的介绍，如果你只是想囫囵吞枣，不想理解其原理，你倒是可以略过这一节，但我还是建议你看看，多了解一些没有坏处。

`hash_map`基于hash table（哈希表）。 哈希表最大的优点，就是**把数据的存储和查找消耗的时间大大降低，几乎可以看成是常数时间**；而**代价仅仅是消耗比较多的内存**。然而在当前可利用内存越来越多的情况下，用空间换时间的做法是值得的。另外，编码比较容易也是它的特点之一。

其基本原理是：使用一个下标范围比较大的数组来存储元素。可以设计一个函数（**哈希函数**，也叫做**散列函数**），**使得每个元素的关键字都与一个函数值（即数组下标，hash值）相对应**，于是用这个数组单元来存储这个元素；也可以简单的理解为，按照关键字为每一个元素“分类”，然后将这个元素存储在相应“类”所对应的地方，称为**桶**。

但是，不能够保证每个元素的关键字与函数值是一一对应的，因此极有可能出现对于不同的元素，却计算出了相同的函数值，这样就产生了“冲突”，换句话说，就是把不同的元素分在了相同的“类”之中。 总的来说，“**直接定址**”与“**解决冲突**”是哈希表的两大特点。

`hash_map`，首先分配一大片内存，形成许多**桶**。是利用`hash函数`，对`key`进行映射到不同区域（桶）进行保存。
```cpp
	其插入过程是：
得到key
通过hash函数得到hash值
得到桶号(一般都为hash值对桶数求模)
存放key和value在桶内。


	其取值过程是:
得到key
通过hash函数得到hash值
得到桶号(一般都为hash值对桶数求模)
比较桶的内部元素是否与key相等，若都不相等，则没有找到。
取出相等的记录的value。
```

hash_map中直接地址用hash函数生成，解决冲突，用比较函数解决。这里可以看出，如果每个桶内部只有一个元素，那么查找的时候只有一次比较。当许多桶内没有值时，许多查询就会更快了(指查不到的时候).

由此可见，要实现哈希表, 和用户相关的是：**hash函数**和**比较函数**。这两个参数刚好是我们在使用hash_map时需要指定的参数。


例子
```cpp
#include <hash_map>
#include <string>
using namespace std;
int main(){
    hash_map<int, string> mymap;
    mymap[9527]="唐伯虎点秋香";
    mymap[1000000]="百万富翁的生活";
    mymap[10000]="白领的工资底线";
    ...
        if(mymap.find(10000) != mymap.end()){
            ...
        }
```
你没有指定`hash`函数和`比较函数`的时候，你会有一个**缺省**的函数，看看`hash_map`的声明，你会更加明白。下面是SGI STL的声明：
```cpp
template <class _Key, class _Tp, class _HashFcn = hash<_Key>,
class _EqualKey = equal_to<_Key>,
class _Alloc = __STL_DEFAULT_ALLOCATOR(_Tp) >
class hash_map
{
    ...
}
```
也就是说，在上例中，有以下等同关系：
```cpp
hash_map<int, string> mymap;
//等同于:
hash_map<int, string, hash<int>, equal_to<int> > mymap;
```
## hash_map 的hash函数
`hash< int>`到底是什么样子？看看源码:

```cpp
struct hash<int> {
    size_t operator()(int __x) const { return __x; }
};
```
原来是个函数对象。在SGI STL中，提供了以下hash函数：
```cpp
    struct hash<char*>
    struct hash<const char*>
    struct hash<char> 
    struct hash<unsigned char> 
    struct hash<signed char>
    struct hash<short>
    struct hash<unsigned short> 
    struct hash<int> 
    struct hash<unsigned int>
    struct hash<long> 
    struct hash<unsigned long>
```
也就是说，如果你的`key`使用的是以上类型中的一种，你都可以使用**缺省的hash**函数。当然你自己也可以定义自己的hash函数。对于自定义变量，你只能如此，例如对于**string**，就必须**自定义hash函数**。例如：
```cpp
struct str_hash{
    size_t operator()(const string& str) const
    {
        unsigned long __h = 0;
        for (size_t i = 0 ; i < str.size() ; i ++)
            __h = 5*__h + str[i];
        return size_t(__h);
    }
};

//如果你希望利用系统定义的字符串hash函数，你可以这样写：
struct str_hash{
    size_t operator()(const string& str) const
    {
        return __stl_hash_string(str.c_str());
    }
};
```
在声明自己的哈希函数时要注意以下几点：

1、使用`struct`，然后重载`operator().
2、返回是size_t`
3、参数是你要hash的key的类型。
4、函数是`const`类型的。

现在可以对开头的string 进行哈希化了 . 直接替换成下面的声明即可：
```cpp
map<string, string> namemap; 
//改为：
hash_map<string, string, str_hash> namemap;
```