## 一、What is C++

本贾尼·斯特劳斯特卢普，与1979年4月份贝尔实验室的本贾尼博士在分析UNIX系统分布内核流量分析时，希望有一种有效的更加模块化的工具。1979年10月完成了预处理器Cpre，为C增加了类机制，也就是面向对象，1983年完成了C++的第一个版本，C with classes也就是C++。
### C++与C的不同点：
1. C++基本兼容C的语法（内容）
2. 支持面向对象的编程思想
3. 支持运算符重载
4. 支持泛型编程、模板
5. 支持异常处理
6. 类型检查严格



### 二、 第一个C++程序

### 1.文件扩展名
`.cpp`,`.cc` , `.C` ,`.cxx`
### 2. 编译器
g++ 大多数系统需要额外安装，Ubuntu系统下的安装命令：

```shell
sudo apt-get update
sudo apt-get install g++
```


gcc也可以继续使用，但需要增加参数 -xC++ -lstdc++

> gcc and g++ are compiler-drivers of the GNU Compiler Collection (which was once upon a time just 
> the GNU C Compiler).
> Even though they automatically determine which backends (cc1 cc1plus ...) to call depending on the 
> file-type, unless overridden with -x language, they have some differences.
> The probably most important difference in their defaults is which libraries they link against 
> automatically.
> According to GCC's online documentation link options and how g++ is invoked, g++ is equivalent to 
> gcc -xc++ -lstdc++ -shared-libgcc (the 1st is a compiler option, the 2nd two are linker options). 
> This can be checked by running both with the -v option (it displays the backend toolchain commands 
> being run).
> gcc和g++是GNU编译器的编译器驱动程序。收藏(很久以前就是GNUC编译器).
> 即使它们自动确定哪个后端(cc1 cc1plus.)根据文件类型进行调用，除非-x language他们有一些不同之处。
> 它们的默认值中最重要的区别可能是它们自动链接到哪个库。
> 根据GCC的在线文件链接选项和如何调用g++, g++等于gcc -xc++ -lstdc++ -shared-libgcc(第一个是编译器选项，
> 第二个是链接器选项)。可以通过使用-v选项(它显示正在运行的后端工具链命令)。
>
> 引自 
> https://stackoverflow.com/questions/172587/what-is-the-difference-between-g-and-gcc



### 3.g++和gcc的区别的回答如下：
- GCC：GNU编译集
  引用GNU编译器支持的所有不同语言。
  GCC：GNU C编译器
- G++：GNU C++编译器
- 主要区别是：
  GCC将编译：*C/*cpp文件，分别作为C和C++。
  G++将编译：*.c/*.cpp文件，但它们都将被视为C++文件。
- 如果使用g++链接对象文件，它将自动链接到STD C++库中(GCC不会这样做)
  GCC编译C文件的预定义宏较少。
  GCC编译*.cpp和g++编译*.c/*.cpp文件有一些额外的宏。
  编译*.cpp文件时的额外宏：

```c++
#define __GXX_WEAK__ 1
#define __cplusplus 1
#define __DEPRECATED 1
#define __GNUG__ 4
#define __EXCEPTIONS 1
#define __private_extern__ extern
```

可以参考：https://www.zhihu.com/answer/536826078

### 4.头文件

```c++
 #include <iostream>
 #include <stdio.h> 
```

可以继续使用，但C++建议使用 #include <cstdio>

### 5.输入/输出

```c++
cin >> 输入数据;
cout << 输出数据;
//cin/cout会自动识别类型
//scanf/printf可以继续使用
//cout和cin是类对象，而scanf/printf是标准库函数。
```

### 6.增加了namespace
```c++
std::cout
using namespace std;
```