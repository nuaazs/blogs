A namespace is a scope.

C++ provides namespaces to prevent name conflicts.

A namespace is a mechanism for expressing logical grouping. That is, if some declarations logically belong together to some criteria(准则), they can be put in a common namespace to express that fact.

That is, the namespace is the mechanism(机制) for supporting module programming paradigm.

C++命名空间是一种描述逻辑分组的机制。也就是说，如果有一些声明按照某种准则在逻辑上属于同一个模块，就可以将它们放在同一个命名空间。

```cpp
// x.h
namespace MyNamespace1{
    int i;
    void func();
    class CHello{
        public:
        void print();
    }
};
```

```cpp
// y.h
namespace MyNamespace2{
    class CHello{
        public:
        void print();
    }
};
```

```cpp
// z.cpp
#include "x.h"
#include "y.h"
int main(){
    MyNamespace1::CHello x;
    MyNamespace2::CHello y;
    x.print();
    y.print();
    return 0;
}
```

命名空间中可以定义变量，函数和类等自定义数据类型，它们具有相同的作用范围。对于不同的命名空间，可以定义相同的变量名，函数名，类名等等。在使用的时候，只要在成员前区分开不同的命名空间就可以了。命名空间实质上是一个作用域，上例通过引入命名空间解决了命名冲突。

### 命名空间的定义

namespace是命名空间，可以防止多个文件有重复定义成员。命名空间是一个作用域，其形式以关键字namespace开始，后接命名空间的命名，然后一对大括号内写上命名空间的内容。

```cpp
//point.h
namespace spacepoint
{
        struct point
        {
            int x;
            int y;
        };
        void set(point& p, int i, int j)
        {
            p.x = i;
            p.y = j;
        }
} 
//point.cpp
using namespace spacepoint;
int main()
{
        point p;
        set(p,1,2);
        return 0;
}
```

在头文件`point.h`中定义了一个命名空间`spacepoint`，在`point.cpp`文件中若要访问`spacepoint`中的成员，就必须加：`using namespace spacepoint`，表示要使用命名空间`spacepoint`，若没加这句代码，则会有编译错误：

```cpp
//point.cpp
int main()
{
        point p;  //error C2065: “point”: 未声明的标识符
        set(p,1,2); // error C3861: “set”: 找不到标识符
        return 0;
}
```

命名空间的成员，是在命名空间定义中的花括号内声明了的名称。可以在命名空间的定义内，定义命名空间的成员（内部定义）。也可以只在命名空间的定义内声明成员，而在命名空间的定义之外，定义命名空间的成员（外部定义）。命名空间成员的外部定义的格式为：

**命名空间名::成员名 ……**

```cpp
// space.h
namespace Outer // 命名空间Outer的定义
{ 
        int i; // 命名空间Outer的成员i的内部定义
        namespace Inner // 子命名空间Inner的内部定义
        { 
          void f() { i++; } // 命名空间Inner的成员f()的内部定义，其中的i为Outer::i
          int i;
          void g() { i++; } // 命名空间Inner的成员g()的内部定义，其中的i为Inner::i
          void h(); // 命名空间Inner的成员h()的声明
        }
        void f(); // 命名空间Outer的成员f()的声明
}
void Outer::f() {i--;}   // 命名空间Outer的成员f()的外部定义
void Outer::Inner::h() {i--;}  // 命名空间Inner的成员h()的外部定义
```



### 域操作符

**The scope resolution operator (域解析符)**`::`occurs between the namespaces name and the variable.

“`::`”就是作用域操作符，首先要有命名空间的概念。用户声明的命名空间成员名自动被加上前缀，命名空间名后面加上域操作符“`::`”，命名空间成员名由该命名空间名进行限定修饰。命名空间成员的声明被隐藏在其命名空间中，除非我们为编译器指定查找的声明的命名空间，否则编译器将在**当前域及嵌套包含当前域的域中查找该命名的声明**。

```cpp
#include "space.h"
int main ( ) 
{
    Outer::i = 0;
    Outer::f();        // Outer::i = -1;
    Outer::Inner::f();   // Outer::i = 0;
    Outer::Inner::i = 0;
    Outer::Inner::g();   // Inner::i = 1;
    Outer::Inner::h();   // Inner::i = 0;
    std::cout << "Hello, World!" << std::endl;
    std::cout << "Outer::i = " << Outer::i << ", Inner::i = " <<     Outer::Inner::i << std::endl;
}
```

