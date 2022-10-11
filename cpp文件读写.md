# 写在前面：不要忘了 int main()！！

### 读取文本

使用头文件`#include <fstream>`，包含`ifstream-输入流`,`ofstream-输出流`,`iostream-输入输出流`三种类型的文件流。

```cpp
ifstream iFile;
int a;
iFile.open(filename);//ifstream默认以读方式打开
//do your job
iFile>>a;
iFile.close();
```

### 逐行读取

```cpp
char buf[300000];
while (!iFile.eof()){// find the end of the file
    iFile.getline(buf,300000);}
```

### 分割字符串

#### 方法1：使用strtok函数

`strtok`会在分割的位置添加一个`\0`，返回每一个**分割部分的头指针**。所以它返回的是**buf上的地址**，当buf失效了，它返回的指针也失效了。其次，因为它需要**改变原字符串**，所以buf不能传入const char*类型

```cpp
const char* d=" *"; //token 分隔符，支持多个分隔符，如这里空格和*
char* p=strtok(buf,d);
while(p){
    cout<<p<<" ";
    p=strtok(NULL,d);//循环过程要把buf赋为NULL
    }
```

#### 方法2：使用STL find和substr函数

涉及到`string`类的两个函数`find`和`substr`：

##### find函数

**原型：**`size_t find ( const string& str, size_t pos = 0 ) const;`
**功能：**查找子字符串第一次出现的位置。
**参数说明：**str为子字符串，pos为初始查找位置。
**返回值：**找到的话返回第一次出现的位置，否则返回`string::npos`

##### substr函数

**原型：**`string substr ( size_t pos = 0, size_t n = npos ) const;`
**功能：**截取从pos开始到n结束这段字符串。
**参数说明：**pos为起始位置（默认为0），n为结束位置（默认为npos）
**返回值：**子字符串

```cpp
std::vector<std::string> split(std::string str, std::string pattern){
    std::string::size_type pos;
    std::vector<std::string> result;
    str+=pattern; //扩展字符串以方便操作
    int size = str.size();
    for(int i=0; i<size; i++){
        pos = str.find(pattern,i);
        if(pos<size){
            std::string s = str.substr(i,pos-i);
            result.push_back(s);
            i=pos+pattern.size()-1;
        }
    }
    return result;
}
```

C语言中的`read()`函数（linux管道中也是使用read）：

`ssize_t read(int fd,void *buf,int count)`，从文件描述符`fd`中读取`count`长度，存入`buf`中，返回实际读取的大小。如返回值少于count则文件已经读完，如等于则代表buf不足以放下整个文件。

如文件没读取完成，文件的指针会自动记录当前读取的地方，下次调用`read()`从未读处开始。

```cpp
int fd;
char buf[BUF_SIZE];
fd = open("f:/test.txt",0);
if(-1 == fd)
    return -1;
int n=read(fd,buf,2);
if(-1==n || 0 ==n){
    return -1;
    close(fd);
}
```









# C++文件读写详解（ofstream,ifstream,fstream）

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/7aJd0.jpg)

这里主要是讨论`fstream`的内容：

```cpp
#include <fstream>
ofstream //文件写操作 内存写入存储设备
ifstream //文件读操作，存储设备读取到内存中
fstream //读写操作，对打开的文件可进行读写操作
```



### 1. 打开文件

在`fstream`类中，成员函数`open()`实现打开文件的操作，从而将数据流和文件进行关联，通过`ofstream`,`ifstream`,`fstream`对象进行对文件的读写操作

函数：`open()`

```cpp
public member function
void open(const char * filename, ios_base::openmode mode = ios_base);
void open(const wchar_t * _Filename,
         ios_base::openmode mode=ios_base::in | ios_base::out),
		 int prot = ios_base::_Openprot);
```

参数：

- filename 操作文件名
- mode 打开文件的方式
- prot 打开文件的属性 , 基本很少用到

打开文件的方式在ios类(所以流式I/O的基类)中定义，有如下几种方式：

ios::in	为输入(读)而打开文件
ios::out	为输出(写)而打开文件
ios::ate	初始位置：文件尾
ios::app	所有输出附加在文件末尾
ios::trunc	如果文件已存在则先删除该文件
ios::binary	二进制方式

这些方式是可以**组合使用**的，用`|`

```cpp
ofstream out;
out.open("Hello.txt", ios::in|ios::out|ios::binary)
```

打开文件的属性同样在ios类中也有定义：

| 0    | 普通文件，打开操作 |
| ---- | ------------------ |
| 1    | 只读文件           |
| 2    | 隐含文件           |
| 4    | 系统文件           |

对于文件的属性也可以使用“`或`”运算和“`+`”进行组合使用，这里就不做说明了。
很多程序中，可能会碰到`ofstream out("Hello.txt")`, `ifstream in("...")`,`fstream foi("...")`这样的的使用，并没有显式的去调用`open()`函数就进行文件的操作，直接调用了其默认的打开方式，因为在`stream`类的构造函数中调用了`open()`函数,并拥有同样的构造函数，所以在这里可以直接使用流对象进行文件的操作，默认方式如下：

```cpp
ofstream out("...", ios::out);
ifstream in("...", ios::in);
fstream foi("...", ios::in|ios::out);
```

当使用默认方式进行对文件的操作时，你可以使用成员函数`is_open()`对文件是否打开进行验证

### 2.关闭文件

当文件读写操作完成之后，我们必须将文件关闭以使文件重新变为可访问的。成员函数`close()`，它负责将缓存中的数据排放出来并关闭文件。这个函数一旦被调用，原先的流对象就可以被用来打开其它的文件了，这个文件也就可以重新被其它的进程所访问了。为防止流对象被销毁时还联系着打开的文件，析构函数将会自动调用关闭函数`close`。

### 3.文本文件的读写

类`ofstream`, `ifstream` 和`fstream` 是分别从`ostream`,`istream` 和`iostream` 中引申而来的。这就是为什么 `fstream` 的对象可以使用其父类的成员来访问数据。

一般来说，我们将使用这些类与同控制台(`console`)交互同样的成员函数(`cin` 和 `cout`)来进行输入输出。如下面的例题所示，我们使用重载的插入操作符`<<`：

```cpp
// writing on a text file
#include <fiostream.h>
int main(){
    ofstream out("out.txt");
    if (out.is_open()){
        out << "This is a lint.\n";
        out << "This is another line.\n";
        out.close();
    }
    return 0;
}
```

在文件中读入数据也可以用`cin>>`

```cpp
//reading a text file
#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
int main(){
    char buffer[256];
    ifstream in("test.txt");
    if(!in.is_open()){
        cout << "Error optning file";
        exit(1);
    }
    while(!in.eof()){
        in.getline(buffer,100);
        cout << buffer << endl;
    }
    return 0;
}
```

上面的例子读入一个文本文件的内容，然后将它打印到屏幕上。注意我们使用了一个新的成员函数叫做`eof` ，它是`ifstream` 从类 `ios` 中继承过来的，当到达文件末尾时返回`true` 。

### 状态标志符的验证(Verification of state flags)

除了`eof()`以外，还有一些验证流的状态的成员函数（所有都返回`bool`型返回值）：

`bad()`
如果在读写过程中出错，返回 `true` 。例如：当我们要对一个不是打开为写状态的文件进行写入时，或者我们要写入的设备没有剩余空间的时候。

`fail()`
除了与`bad()` 同样的情况下会返回 `true` 以外，加上**格式错误**时也返回`true` ，例如当想要读入一个整数，而获得了一个字母的时候。

`eof()`
如果读文件到达文件末尾，返回`true`。

`good()`
这是最通用的：如果调用以上任何一个函数返回`true` 的话，此函数返回 `false` 。

要想重置以上成员函数所检查的状态标志，你可以使用成员函数`clear()`，没有参数。

### 获得和设置流指针(get and put stream pointers)

所有输入/输出流对象(i/o streams objects)都有至少一个流指针：

- **ifstream**， 类似`istream`, 有一个被称为`get pointer`的指针，指向下一个将被读取的元素。
- **ofstream**, 类似 `ostream`, 有一个指针 `put pointer` ，指向写入下一个元素的位置。
- **fstream**, 类似 `iostream`, 同时继承了`get` 和 `put`

我们可以通过使用以下成员函数来读出或配置这些指向流中读写位置的流指针：

`tellg()` 和 `tellp()`
这两个成员函数不用传入参数，返回`pos_type` 类型的值(根据ANSI-C++ 标准) ，就是一个整数，代表当前`get`流指针的位置 (用`tellg`) 或 `put` 流指针的位置(用`tellp`).

`seekg()` 和`seekp()`
这对函数分别用来**改变**流指针`get` 和`put`的位置。两个函数都被重载为两种不同的原型：

`seekg ( pos_type position );`
`seekp ( pos_type position );`
使用这个原型，流指针被改变为指向从文件开始计算的一个绝对位置。要求传入的参数类型与函数 `tellg` 和`tellp` 的返回值类型相同。

`seekg ( off_type offset, seekdir direction );`
`seekp ( off_type offset, seekdir direction );`
使用这个原型可以指定由参数`direction`决定的一个具体的指针开始计算的一个位移(`offset`)。它可以是：

`ios::beg`	从流开始位置计算的位移
`ios::cur`	从流指针当前位置开始计算的位移
`ios::end`	从流末尾处开始计算的位移
流指针 `get` 和 `put` 的值对文本文件(text file)和二进制文件(binary file)的计算方法都是不同的，因为文本模式的文件中某些特殊字符可能被修改。由于这个原因，建议对以文本文件模式打开的文件总是使用`seekg` 和 `seekp`的第一种原型，而且不要对`tellg` 或 `tellp` 的返回值进行修改。对二进制文件，你可以任意使用这些函数，应该不会有任何意外的行为产生。

以下例子使用这些函数来获得一个二进制文件的大小：

```cpp
//obtaining file size
#include <iostream.h>
#include <fstream.h>
const char * filename = "test.txt";
int main(){
    long l,m;
    ifstream in(filename,ios::in|ios::binary);
    l = in.tellg();
    in.seekg(0,ios::end);
    m = in.tellg();
    in.close();
    cout<< "size of " << filename;
    cout<< " is " << (m-;) << "bytes.\n";
    return 0;
}
// size of example.txt is 40 bytes.
```



### 二进制文件

在二进制文件中，使用`<<` 和`>>`，以及函数（如`getline`）来操作符输入和输出数据，没有什么实际意义，虽然它们是符合语法的。

文件流包括两个为顺序读写数据特殊设计的成员函数：`write` 和 `read`。第一个函数 (write) 是ostream 的一个成员函数，都是被ofstream所继承。而read 是istream 的一个成员函数，被ifstream 所继承。类 fstream 的对象同时拥有这两个函数。它们的原型是：

`write ( char * buffer, streamsize size );
read ( char * buffer, streamsize size );`
这里 `buffer` 是一块**内存的地址**，用来存储或读出数据。参数`size` 是一个整数值，表示要从缓存（buffer）中读出或写入的字符数。

```cpp
//reading binary file
#include <iostream>
#include <fstream.h>
const char * filename = "test.txt";
int main(){
    char * buffer;
    long size;
    ifstream in (filename,ios::in|ios::binary|ios::ate);
    size = in.tellg();
    in.seekg(0,ios::beg);
    buffer = new char [size];
    in.read(buffer,size);
    in.close();
    cout<< "The complete file is in a buffer";
    delete[] buffer;
    return 0;
}
// The complete file is in a buffer
```



### 缓存和同步（Buffers and Synchronization）

当我们对文件流进行操作的时候，它们与一个`streambuf` 类型的缓存(`buffer`)联系在一起。这个缓存（`buffer`）实际是一块内存空间，作为流(`stream`)和物理文件的媒介。例如，对于一个输出流， 每次成员函数`put`(写一个单个字符)被调用，这个字符不是直接被写入该输出流所对应的物理文件中的，而是首先被插入到该流的缓存（`buffer`）中。

当缓存被排放出来(`flush`)时，它里面的所有数据或者被写入物理媒质中（如果是一个输出流的话），或者简单的被抹掉(如果是一个输入流的话)。这个过程称为同步(`synchronization`)，它会在以下任一情况下发生：

- 当文件被关闭时: 在文件被关闭之前，所有还没有被完全写出或读取的缓存都将被同步。
- 当缓存buffer 满时:缓存Buffers 有一定的空间限制。当缓存满时，它会被自动同步。
- 控制符明确指明:当遇到流中某些特定的控制符时，同步会发生。这些控制符包括：flush 和endl。
- 明确调用函数sync(): 调用成员函数sync() (无参数)可以引发立即同步。这个函数返回一个int 值，等于-1 表示流没有联系的缓存或操作失败。