## 一、用途

`auto`是c++程序设计语言的关键字。用于两种情况

（1）声明变量时根据初始化表达式自**动推断该变量的类型**

（2）声明函数时**函数返回值的占位符**



## 二、简要理解

`auto`可以在声明变量时根据变量初始值的类型自动为此变量选择匹配的类型。

举例：对于值`x=1`既可以声明：` int x=1 `或 `long x=1`，也可以直接声明 `auto x=1`



## 三、用法

根据初始化表达式自动推断被声明的变量的类型，如：

```cpp
auto f = 3.14;  //double
auto s("hello");  //const char*
auto z = new auto(9);  //int *
auto x1 = 5, x2 = 5.0, x3 = 'r';   //错误，必须是初始化为同一类型
```

但是，这么简单的变量声明类型，不建议用`auto`关键字，而是应更清晰地直接写出其类型。

`auto`关键字更适用于类型冗长复杂、变量使用范围专一时，使程序更清晰易读。如：

```cpp
 std::vector<int> vect; 
 for(auto it = vect.begin(); it != vect.end(); ++it)
 {  
    //it的类型是std::vector<int>::iterator
    std::cin >> *it;
  }
```

或者保存`lambda`表达式类型的变量声明：

```cpp
auto ptr = [](double x){return x*x;}; //类型为std::function<double(double)>函数对象
```



## 四、优势

### 1. 拥有初始化表达式的复杂类型变量声明时简化代码。

比如：

```cpp
#include <string>  
#include <vector>  
void loopover(std::vector<std::string>&vs)  
{  
    std::vector<std::string>::iterator i=vs.begin();  
    for(;i<vs.end();i++)  
    {  
      
    }  
  
}
```

变为：

```cpp
#include <string>  
#include <vector>  
void loopover(std::vector<std::string>&vs)  
{  
    for(  auto i=vs.begin();;i<vs.end();i++)  
    {  
      
    }  
  
} 
```

使用`std::vector<std::string>::iterator`来定义`i`是C++常用的良好的习惯，但是这样长的声明带来了代码可读性的困难，因此引入`auto`，使代码可读性增加。并且使用STL将会变得更加容易

### 2. 可以避免类型声明时的麻烦而且避免类型声明时的错误。

但是auto不能解决所有的精度问题。比如：

```cpp
#include <iostream>  
using namespace std;  
int main()  
{  
   unsigned int a=4294967295;//最大的unsigned int值  
   unsigned int b=1；  
   auto c=a+b;  
   cout<<"a="<<a<<endl;  
   cout<<"b="<<b<<endl;  
   cout<<"c="<<c<<endl;  
}  
```

上面代码中，程序员希望通过声明变量c为auto就能解决a+b溢出的问题。而实际上由于a+b返回的依然是`unsigned int`的值，姑且c的类型依然被推导为unsigned int，auto并不能帮上忙。这个跟动态类型语言中数据自动进行拓展的特性还是不一样的。

## 五、注意的地方

### 1. 可以用valatile，pointer（*），reference（&），rvalue reference（&&） 来修饰auto

```cpp
auto k = 5;  
auto* pK = new auto(k);  
auto** ppK = new auto(&k);  
const auto n = 6;  
```

### 2. 用auto声明的变量必须初始化

### 3. auto不能与其他类型组合连用

### 4. 函数和模板参数不能被声明为auto

### 5. 定义在堆上的变量，使用了auto的表达式必须被初始化

```cpp
int* p = new auto(0); //fine  
int* pp = new auto(); // should be initialized  
auto x = new auto(); // Hmmm ... no intializer  
auto* y = new auto(9); // Fine. Here y is a int*  
auto z = new auto(9); //Fine. Here z is a int* (It is not just an int)  
```

### 6. 以为auto是一个占位符，并不是一个他自己的类型，因此不能用于类型转换或其他一些操作，如sizeof和typeid

### 7. 定义在一个auto序列的变量必须始终推导成同一类型

```cpp
auto x1 = 5, x2 = 5.0, x3='r';  //错误，必须是初始化为同一类型
```

### 8. auto不能自动推导成CV-qualifiers (constant & volatile qualifiers)

### 9. auto会退化成指向数组的指针，除非被声明为引用
