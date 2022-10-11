# C++左值引用和右值引用

### 左值与右值

**左值**是指既能出现在等号左边也能出现在等号右边的变量(或表达式)，**右值则只能出现在等号右边**。

返回左值引用的函数，连同赋值、下标、解引用和前置递增/递减运算符，都是返回左值的表达式。

返回非引用类型/右值引用的函数，连同算术、关系、位以及后置递增/递减运算符，都返回右值的表达式

**左值持久，右值短暂，左值有持久的状态，而右值要么是字面常量，要么是在表达式求值过程中创建的临时对象(将要被销毁的对象)。**



### 引用

引用是给一个存在的对象定义的别名， 一个变量可以有多个引用，引用必须初始化，引用只能在初始化的时候引用一次，不能更改引用其他变量。

### 左值引用

通过`&`获得获得左值引用，左值引用只能绑定左值。

```cpp
int intValue1=10;
//将intValue1绑定到intValue2和intValue3

int &intValue2 = intValue1, &intValue3 = intValue2;
intValue2 = 100;
std::cout << intValue1 << std::endl;//100
std::cout << intValue2 << std::endl;//100
std::cout << intValue3 << std::endl;//100
```

不能将左值引用绑定到一个右值，但是const的左值引用可以，常量引用不能修改绑定对象的值。

```cpp
int &intValue1 = 10; //错误
const int &intValue2 = 10; //正确
```

### 右值引用

通过`&&`获得右值引用，右值引用只能绑定右值。

右值引用的好处是减少右值作为参数传递时的复制开销。

```cpp
int intValue = 10;
int &&intValue2 = 10; //正确
int &&intValue3 = intValue; //错误
```

使用`std::move`可以获得绑定到一个左值的右值引用

```cpp
int intValue = 10;
int &&intValue3 = std::move(intValue);
```