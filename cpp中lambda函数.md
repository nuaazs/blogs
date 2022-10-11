# C++之Lambda表达式

## 1. 概述

C++ 11 中的 Lambda 表达式用于定义并创建匿名的函数对象，以简化编程工作。Lambda 的语法形式如下：

```cpp
[函数对象参数] (操作符重载函数参数) mutable 或 exception 声明 -> 返回值类型 {函数体}
```

可以看到，Lambda 主要分为五个部分：`[函数对象参数]`、`(操作符重载函数参数)`、`mutable 或 exception 声明`、`-> 返回值类型`、`{函数体}`。

## 2. Lambda 语法分析

### 2.1 `[函数对象参数]`

标识一个 Lambda 表达式的开始，这部分必须存在，不能省略。**函数对象参数是传递给编译器自动生成的函数对象类的构造函数的。**

函数对象参数只能使用那些到定义 Lambda 为止时 Lambda 所在作用范围内可见的局部变量(包括 Lambda 所在类的 this)。函数对象参数有以下形式：

- `空`。没有任何函数对象参数。
- `=`。函数体内可以使用 Lambda 所在范围内所有可见的局部变量（包括 Lambda 所在类的 this），并且是值传递方式（相当于编译器自动为我们按值传递了所有局部变量）。
- `&`。函数体内可以使用 Lambda 所在范围内所有可见的局部变量（包括 Lambda 所在类的 this），并且是引用传递方式（相当于是编译器自动为我们按引用传递了所有局部变量）。
- `this`。函数体内可以使用 Lambda 所在类中的成员变量。
- `a`。将 a 按值进行传递。按值进行传递时，函数体内不能修改传递进来的 a 的拷贝，因为默认情况下函数是 const 的，要修改传递进来的拷贝，可以添加 mutable 修饰符。
- `&a`。将 a 按引用进行传递。
- `a，&b`。将 a 按值传递，b 按引用进行传递。
- `=，&a，&b`。除 a 和 b 按引用进行传递外，其他参数都按值进行传递。
- `&，a，b`。除 a 和 b 按值进行传递外，其他参数都按引用进行传递。

### 2.2 `(操作符重载函数参数)`

标识重载的 () 操作符的参数，没有参数时，这部分可以省略。参数可以通过按值（如: `(a, b)`）和按引用 (如:`(&a, &b)`) 两种方式进行传递。

### 2.3 `mutable 或 exception 声明`

这部分可以省略。按值传递函数对象参数时，加上 mutable 修饰符后，可以修改传递进来的拷贝（注意是能修改拷贝，而不是值本身）。`exception` 声明用于指定函数抛出的异常，如抛出整数类型的异常，可以使用 `throw(int)`。

### 2.4 `-> 返回值类型`

标识函数返回值的类型，当返回值为 void，或者函数体中只有一处 return 的地方（此时编译器可以自动推断出返回值类型）时，这部分可以省略。

### 2.5 `{函数体}`

标识函数的实现，这部分不能省略，但函数体可以为空。

## 3. 示例

```cpp
[] (int x, int y){return x + y;} // 隐式返回类型
[] (int& x){++x;} // 没有 return 语句 -> Lambda 函数的返回类型是 'void'
[] (){++global_x;} // 没有参数，仅访问某个全局变量
[] {++global_x;} // 与上一个相同，省略了 (操作符重载函数参数)
```

可以像下面这样显示指定返回类型：

```cpp
[] (int x, int y) -> int { int z = x + y; return z; }
```

在这个例子中创建了一个临时变量 z 来存储中间值。和普通函数一样，这个中间值不会保存到下次调用。什么也不返回的Lambda 函数可以省略返回类型，而不需要使用` -> void` 形式。

Lambda 函数可以引用在它之外声明的变量. 这些变量的集合叫做一个闭包. 闭包被定义在 Lambda 表达式声明中的方括号 `[]` 内。这个机制允许这些变量被按值或按引用捕获。

```cpp
[] //未定义变量，试图在Lambda内使用任何外部变量都是错的。
[x,&y] //x按值捕获，y按引用捕获。
[&] //用到的任何外部变量都隐式按引用捕获。
[=] //用到的任何外部变量都隐式按值捕获。
[&,x] //x显示地按值捕获，其他变量按引用捕获。
[=,&z] //z按引用捕获，其他变量按值捕获。
```



### 3.1 示例 1

```cpp
std::vector<int> some_list;
int total = 0;
for (int i = 0; i < 5; ++i) some_list.push_back(i);
std::for_each(begin(some_list), end(some_list), [&total](int x)
{
    total += x;
});
```

此例计算 list 中所有元素的总和。变量 `total` 被存为 Lambda 函数闭包的一部分。因为它是**栈变量（局部变量）**total 引用，所以可以改变它的值。

### 3.2 示例 2

```cpp
std::vector<int> some_list;
int total = 0;
int value = 5;
std::for_each(begin(some_list), end(some_list), [&, value, this](int x)
{
    total += x * value * this->some_func();
});
```

此例中 `tota`l 会存为引用,`value` 则会存一份值拷贝。**对 this 的捕获比较特殊，它只能按值捕获。**

`this` 只有当包含它的最靠近它的函数不是静态成员函数时才能被捕获。对 `protect` 和 `private` 成员来说，这个 `Lambda` 函数与创建它的成员函数有相同的访问控制。如果 `this` 被捕获了，不管是显式还是隐式的，那么它的类的作用域对 Lambda 函数就是可见的。访问`this` 的成员不必使用 `this->` 语法，可以直接访问。



## 4. 总结

不同编译器的具体实现可以有所不同，但期望的结果是: 按引用捕获的任何变量，Lambda 函数实际存储的应该是这些变量在创建这个 Lambda 函数的函数的**栈指针**，而不是 Lambda 函数本身栈变量的引用。不管怎样，因为大多数 Lambda 函数都很小且在局部作用中，与候选的**内联函数**很类似，所以按引用捕获的那些变量不需要额外的存储空间。

如果一个闭包含有局部变量的引用，在超出创建它的作用域之外的地方被使用的话，这种行为是未定义的!   Lambda 函数是一个依赖于实现的函数对象类型,这个类型的名字只有编译器知道. 如果用户想把 lambda 函数做为一个参数来传递, 那么形参的类型必须是模板类型或者必须能创建一个 `td::function` 类似的对象去捕获 `lambda` 函数。使用 `auto` 关键字可以帮助存储 lambda 函数：

```cpp
auto my_lambda_func = [&](int x) { /* ... */ };
auto my_onheap_lambda_func = new auto([=](int x) { /* ... */ });
```

这里有一个例子, 把匿名函数存储在变量、数组或 vector 中,并把它们当做命名参数来传递:

```cpp
#include <functional>
#include <vector>
#include <iostream>
double eval(std::function<double(double)> f, double x = 2.0){
    return f(x);
}
int main(){
    std::function<double(double)> f0 = [](double x){return 1;};
    auto f1 = [](double x){return x;};
    decltype(f0) fa[3] = {f0,f1,[](double x){return x*x};};
    std::vector<decltype(f0)> fv = {f0,f1};
    fv.push_back ([](double x){returnx*x;});
    for (int i=0; i<fv.size() ; i++) std::cout<<fv[i](2.0)<<"\n";
    for (int i=0 ;i<3 ; i++) std::cout<<fa[i](2.0) << "\n";
    for (auto &f:fv) std::cout<<f(2.0) << "\n";
    for (auto &f:fa) std::cout<<f(2.0) << "\n";
    std::cout << eval(f0) << "\n";
    std::cout << eval(f1) << "\n";
    return 0
}
```

一个没有指定任何捕获的 lambda 函数,可以显式转换成一个具有相同声明形式函数指针.所以,像下面这样做是合法的:

```cpp
auto a_lambda_func = [](int x) { /* ... */ };
void (*func_ptr)(int) = a_lambda_func;
func_ptr(4); // calls the lambda
```