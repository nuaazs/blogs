

程序运行时常会碰到一些异常情况，例如：

- 做除法的时候除数为 0；
- 用户输入年龄时输入了一个负数；
- 用 new 运算符动态分配空间时，空间不够导致无法分配；
- 访问数组元素时，下标越界；打开文件读取时，文件不存在。


这些异常情况，如果不能发现并加以处理，很可能会导致程序崩溃。

所谓“处理”，可以是给出错误提示信息，然后让程序沿一条不会出错的路径继续执行；也可能是不得不结束程序，但在结束前做一些必要的工作，如将内存中的数据写入文件、关闭打开的文件、释放动态分配的内存空间等。

一发现异常情况就立即处理未必妥当，因为在一个函数执行过程中发生的异常，在有的情况下由该函数的调用者决定如何处理更加合适。尤其像库函数这类提供给程序员调用，用以完成与具体应用无关的通用功能的函数，执行过程中贸然对异常进行处理，未必符合调用它的程序的需要。

此外，将异常分散在各处进行处理不利于代码的维护，尤其是对于在不同地方发生的同一种异常，都要编写相同的处理代码也是一种不必要的重复和冗余。如果能在发生各种异常时让程序都执行到同一个地方，这个地方能够对异常进行集中处理，则程序就会更容易编写、维护。

鉴于上述原因，C++ 引入了异常处理机制。其基本思想是：函数 A 在执行过程中发现异常时可以不加处理，而只是“拋出一个异常”给 A 的调用者，假定为函数 B。

拋出异常而不加处理会导致函数 A 立即中止，在这种情况下，函数 B 可以选择捕获 A 拋出的异常进行处理，也可以选择置之不理。如果置之不理，这个异常就会被拋给 B 的调用者，以此类推。

如果一层层的函数都不处理异常，异常最终会被拋给最外层的 main 函数。main 函数应该处理异常。如果main函数也不处理异常，那么程序就会立即异常地中止。

### C++异常处理基本语法

C++ 通过`throw`语句和 `try...catch` 语句实现对异常的处理。`throw` 语句的语法如下：

```cpp
throw 表达式;
```

该语句抛出一个异常。异常是一个表达式，其值的类型可以是基本类型也可以是类。

`try... catch`语句的语法如下：

```cpp
try{
    语句组
}
catch(异常类型){
    异常处理代码
}
...
catch(异常类型){
    异常处理代码
}
```

`catch`可以有多个，但至少要一个。

不妨把`try`和其后`{}`中的内容称作`try块`，把`catch`和其后`{}`中的内容称作`catch块`。

`try...catch`语句的执行过程是：

- 执行try块中的语句，如果执行的过程中没异常抛出，那么执行完后就执行最后一个`catch块`后面的语句
- 如果`try块`执行过程中抛出了异常，那么抛出异常后立即跳转到第一个“异常类型”和抛出异常类型匹配的`catch块`中执行（称作异常被该`catch块`捕获），执行完后再跳转到最后一个`catch块`后面继续执行。

```cpp
#include <iostream>
using namespace std;
int main(){
    double m, n;
    cin >> m >> n;
    try{
        cout << " before dividing." << endl;
        if (n == 0)
            throw -1; //抛出int类型异常
        else
            cout << m/n << endl;
        cout << "after dividing." << endl;
    }
    catch(double d){
        cout << "catch(double)" << d << endl;
    }
    
    catch(int e){
        cout << "catch(int)" << e << endl;
    }
    cout << "finished" << endl;
    return 0;
}
```

程序的运行结果如下：

```cpp
9 6↙
before dividing.
1.5
after dividing.
finished
```

说明当 n 不为 0 时，`try 块`中不会拋出异常。因此程序在` try 块`正常执行完后，越过所有的 `catch 块`继续执行，**catch 块一个也不会执行**。

程序的运行结果也可能如下：

```cpp
9 0↙
before dividing.
catch(int) -1
finished
```

当 n 为 0 时，try 块中会拋出一个整型异常。拋出异常后，`try块`立即停止执行。该整型异常会被类型匹配的第一个 `catch 块`捕获，即进入`catch(int e)`块执行，**该 catch 块执行完毕后，程序继续往后执行，直到正常结束**。

如果拋出的异常没有被 `catch块`捕获，例如，将`catch(int e)`，改为`catch(char e)`，当输入的 n 为 0 时，拋出的整型异常就没有 `catch块`能捕获，**这个异常也就得不到处理，那么程序就会立即中止**，`try...catch` 后面的内容都不会被执行。

### 能够捕获任何异常的 catch 语句

如果希望不论拋出哪种类型的异常都能捕获，可以编写如下 `catch 块`：

```cpp
catch(...){
    ...
}
```

这样的 catch 块能够捕获任何还没有被捕获的异常。例如下面的程序：

```cpp
#include <iostream>
using namespace std;
int main(){
    double m, n;
    cin >> m >> n;
    try{
         cout << "before dividing." << endl;
        if (n == 0)
            throw - 1;  //抛出整型异常
        else if (m == 0)
            throw - 1.0;  //拋出 double 型异常
        else
            cout << m / n << endl;
        cout << "after dividing." << endl;
    }
    catch (double d) {
        cout << "catch (double)" << d << endl;
    }
    catch (...) {
        cout << "catch (...)" << endl;
    }
    cout << "finished" << endl;
    return 0;
}
```

程序的运行结果如下：

```cpp
9 0↙
before dividing.
catch (...)
finished
```

当 n 为 0 时，拋出的整型异常被`catchy(...)`捕获。

程序的运行结果也可能如下：

```cpp
0 6↙
before dividing.
catch (double) -1
finished
```

当 m 为 0 时，拋出一个 `double`类型的异常。虽然`catch (double)`和`catch(...)`都能匹配该异常，但是`catch(double)`是第一个能匹配的 catch 块，因此会执行它，而不会执行`catch(...)`块。

由于`catch(...)`能匹配任何类型的异常，它后面的 `catch块`实际上就不起作用，**因此不要将它写在其他 catch 块前面。**

### 异常的再拋出

如果一个函数在执行过程中拋出的异常在本函数内就被 catch 块捕获并处理，那么该异常就不会拋给这个函数的调用者（也称为“上一层的函数”）；如果异常在本函数中没有被处理，则它就会被拋给上一层的函数。例如下面的程序：

```cpp
#include <iostream>
#include <string>
using namespace std;
class CException{
    public:
    string msg:
    CException(string s):msg(s){}    
};

double Devide(double x, double y){
    if(y == 0)
        throw CException("devided by zero");
    cout << "in Devide " << endl;
    return x/y;
}

int CountTax(int salary){
    try{
        if (salary < 0)
            throw -1;
        cout << "counting tax" << endl;
    }
    catch(int){
        cout << "salary < 0" << endl;
    }
    cout << "tax counted " << endl;
    return salary * 0.15;
}
int main(){
    double f = 1.2
        try{
            CountTax(-1);
            f = Devide(3,0);
            cout << " end of try block" << endl;
        }
    catch(CException e){
        cout << e.msg << endl;
    }
    cout << "f= " << f << endl;
    cout << "finished" << endl;
    return 0;
}
```

程序的输出结果如下：

```cpp
salary < 0
tax counted
    divided by zero
    f=1.2
    finished
```

CountTax函数抛出异常后自行处理，这个异常就不会继续被抛给调用者，即main函数。因此在main函数的try块中，CountTax之后的语句还能正常执行，即会执行` f=Devide(3,0);`.

第35行，Devide函数抛出了异常却不处理，该异常就会被抛给Devide函数的调用者，即main函数。抛出此异常后，Devide函数立即结束执行，第14行不被执行，函数也不会返回一个值，这从第35行f的值不会被修改可以看出。

Devide函数中抛出的异常被main函数中类型匹配的catch块捕获。第38行中的e对象是用复制构造函数初始化的。

如果抛出的异常是派生类的对象，而catch块的异常类型是基类，那么这两者也能够匹配，因为派生类对象也是基类对象。

虽然函数也可以通过返回值或者传引用的参数通知调用者发生了异常，但采用这种方式的话，每次调用函数时都要判断是否发生了异常，这在函数被多处调用时比较麻烦。有了异常处理机制，可以将多出函数调用写在一个try块中，任何一处调用发生异常都会被匹配的catch块捕获并处理，也就不需要每次调用后都判断是否发生了异常。

有时虽然在函数中对异常进行了处理，但是还是希望能通过通知调用者，以便让调用者知道发生了异常，从而可以进一步处理，在catch块中抛出异常可以满足这种需求。例如：

```cpp
#include <iostream>
#include <string>
using namespace std;
int CountTax(int salary){
    try{
        if(salary < 0)
            throw string("zero salary");
        cout << "counting tax" << endl;
    }
    catch (string s){
        couot << "CountTax error:"<<s<<end;
        throw; //继续抛出捕获的异常
    }
    cout << "tax counted" << endl;
    return salary*0.15;
}
int main(){
    double f = 1.2;
    try{
        CountTax(-1);
        cout << "end of try block" << endl;
	}
    catch(string s){
        cout << s << endl;
    }
    cout << "finished" << endl;
    return 0;
}
```

程序的输出结果如下：

```cpp
CountTax error:zero salary
zero salary
finished
```

第 14 行的`throw;`没有指明拋出什么样的异常，因此拋出的就是 catch 块捕获到的异常，即 string("zero salary")。这个异常会被 main 函数中的 catch 块捕获。



### 函数的异常声明列表

为了增强程序的可读性和可维护性，使程序员在使用一个函数时就能看出这个函数可能会拋出哪些异常，C++ 允许在函数声明和定义时，加上它所能拋出的异常的列表，具体写法如下：

```cpp
void func() throw (int, double, A, B, C);
```

或

```cpp
void func() throw (int, double, A, B, C){...}
```

上面的写法表明 `func` 可能拋出 `int` 型、`double` 型以及 `A`、`B`、`C` 三种类型的异常。异常声明列表可以在函数声明时写，也可以在函数定义时写。如果两处都写，则两处应一致。

如果异常声明列表如下编写：

```cpp
void func() throw();
```

则说明 func 函数不会拋出任何异常。

一个函数如果不交待能拋出哪些类型的异常，就可以拋出任何类型的异常。

函数如果拋出了其异常声明列表中没有的异常，在编译时不会引发错误，但在运行时， Dev C++ 编译出来的程序会出错；用 Visual Studio 2010 编译出来的程序则不会出错，异常声明列表不起实际作用。



### C++标准异常类

C++ 标准库中有一些类代表异常，这些类都是从 exception 类派生而来的。常用的几个异常类如图所示。

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/IQ7R4.jpg)

`bad_typeid`、`bad_cast`、`bad_alloc`、`ios_base::failure`、`out_of_range` 都是 `exception` 类的派生类。C++ 程序在碰到某些异常时，即使程序中没有写 `throw` 语句，也会自动拋出上述异常类的对象。这些异常类还都有名为 `what` 的成员函数，返回字符串形式的异常描述信息。使用这些异常类需要包含头文件 `stdexcept`。

1. bad_typeid

   使用typeid运算符时，如果其操作数是一个多态类的指针，而该指针的值为NULL，则会抛出此异常。

2. bac_cast

   在用dynamic_cast进行从多态基类对象（或引用）到派生类的引用的强制类型转换时，如果转换是不安全的，则会抛出此异常：

   ```cpp
   #include <iostream>
   #include <stdexcept>
   using namespace std;
class Base{
       virtual void func(){}
   };
   
   class Derived:public Base{
       public:
       void Print(){}
       
   };
   
   void PrintObj(Base &b){
       try{
           Derived &rd = dynamic_cast<Derived  &>(b);
           //此转换若不安全，会抛出bad_cast异常
           rd.Print();
       }
       catch (bad_cast & e){
           cerr << e.what() << endl;
       }
   }
   
   int main(){
       Base b;
       PrintObj(b);
       return 0;
   }
   ```
   
   程序的输出结果如下：
   
   ```cpp
   Bad dynamic_cast!
   ```
   
   在 `PrintObj` 函数中，通过 `dynamic_cast` 检测 `b` 是否引用的是一个 `Derived` 对象，如果是，就调用其 `Print` 成员函数；如果不是，就拋出异常，不会调用 `Derived::Print`。



3. bad_alloc

   在用new运算符进行动态内存分配时，如果没有足够的内存，则会引发此异常。

   ```cpp
   #include <iostream>
   #include <stdexcept>
   using namespace std;
   int main()
   {
       try {
           char * p = new char[0x7fffffff];  //无法分配这么多空间，会抛出异常
       }
       catch (bad_alloc & e)  {
        cerr << e.what() << endl;
       }
       return 0;
   }
   ```
   
   程序的输出结果如下：
   
   ```cpp
   bad allocation
   ios_base::failure
   ```
   
   在默认状态下，输入输出流对象不会拋出此异常。如果用流对象的 `exceptions` 成员函数设置了一些标志位，则在出现打开文件出错、读到输入流的文件尾等情况时会拋出此异常。此处不再赘述。



4. out_of_range

   用 `vector` 或 `string` 的 `at` 成员函数根据下标访问元素时，如果下标越界，则会拋出此异常。例如：

```cpp
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
using namespace std;
int main()
{
    vector<int> v(10);
    try {
        v.at(100) = 100;  //拋出 out_of_range 异常
    }
    catch (out_of_range & e) {
        cerr << e.what() << endl;
    }
    string s = "hello";
    try {
        char c = s.at(100);  //拋出 out_of_range 异常
    }
    catch (out_of_range & e) {
        cerr << e.what() << endl;
    }
    return 0;
}
```

程序的输出结果如下：

```cpp
invalid vector <T> subscript
invalid string position
```

如果将`v.at(100)`换成`v[100]`，将`s.at(100)`换成`s[100]`，程序就不会引发异常（但可能导致程序崩溃）。因为 at 成员函数会检测下标越界并拋出异常，而 operator[] 则不会。operator [] 相比 at 的好处就是不用判断下标是否越界，因此执行速度更快。