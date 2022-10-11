# C++类中成员变量的初始化
C++类中成员变量的初始化有两种方式：

1. 构造函数初始化列表和构造函数体内赋值。下面看看两种方式有何不同。

2. 成员变量初始化的顺序是按照在类中定义的顺序。

## 1、内部数据类型（char，int……指针等）
```cpp
class Animal  
{  
public:  
    Animal(int weight,int height):       //A初始化列表  
      m_weight(weight),  
      m_height(height)  
    {  
    }  
    Animal(int weight,int height)       //B函数体内初始化  
    {  
        m_weight = weight;  
        m_height = height;  
    }  
private:  
    int m_weight;  
    int m_height;  
};  
```
对于这些内部类型来说，基本上是没有区别的，效率上也不存在多大差异。

当然A和B方式不能共存的。

## 2、无默认构造函数的继承关系中
```cpp
class Animal  
{  
public:  
    Animal(int weight,int height):        //没有提供无参的构造函数   
      m_weight(weight),  
      m_height(height)  
    {  
}  
private:  
    int m_weight;  
    int m_height;  
};  
  
class Dog: public Animal  
{  
public:  
    Dog(int weight,int height,int type)   //error 构造函数 父类Animal无合适构造函数  
    {  
    }  
private:  
    int m_type;  
};  
```
上面的子类和父类编译会出错：
因为子类`Dog`初始化之前要进行父类`Animal`的初始化，但是根据`Dog`的构造函数，没有给父类传递参数，使用了父类`Animal`的无参数构造函数。而父类`Animal`提供了有参数的构造函数，这样编译器就不会给父类`Animal`提供一个默认的无参数的构造函数了，所以编译时报错，说找不到合适的默认构造函数可用。要么提供一个无参数的构造函数，要么在子类的`Dog`的初始化列表中给父类`Animal`传递初始化参数，如下：
```cpp
class Dog: public Animal  
{  
public:  
    Dog(int weight,int height,int type):  
        Animal(weight,height)         //必须使用初始化列表增加对父类的初始化  
    {  
    }  
private:  
    int m_type;  
};  
```

## 3、类中const常量，必须在初始化列表中初始，不能使用赋值的方式初始化
```cpp
class Dog: public Animal  
{  
public:  
    Dog(int weight,int height,int type):  
        Animal(weight,height),   
        LEGS(4)                //必须在初始化列表中初始化  
    {  
        //LEGS = 4;           //error  
    }  
private:  
    int m_type;  
    const int LEGS;  
};  
```
## 4、包含有自定义数据类型（类）对象的成员初始化
```cpp
class Food  
{  
public:  
    Food(int type = 10)  
    {  
        m_type = 10;  
    }  
    Food(Food &other)                 //拷贝构造函数  
    {  
        m_type = other.m_type;  
    }  
    Food & operator =(Food &other)      //重载赋值=函数  
    {  
        m_type = other.m_type;  
        return *this;  
    }  
private:  
    int m_type;  
};  
```
  
（1）构造函数赋值方式 初始化成员对象`m_food`
```cpp
class Dog: public Animal  
{  
public:  
    Dog(Food &food)  
      //:m_food(food)    
    {  
        m_food = food;               //初始化 成员对象  
    }  
private:  
    Food m_food;  
};  
//使用  
Food fd;  
Dog dog(fd);
```
结果：  
先执行了**对象类型构造函数Food**(int type = 10)——>然后在执行**对象类型构造函数Food** & operator =(Food &other)  
想象是为什么？  
  
（2）构造函数初始化列表方式
```cpp
class Dog: public Animal  
{  
public:  
    Dog(Food &food)  
      :m_food(food)    //初始化 成员对象  
    {  
        //m_food = food;                 
    }  
private:  
    Food m_food;  
};  
//使用  
Food fd;  
Dog dog(fd);
```
结果：执行Food(Food &other)拷贝构造函数完成初始化


不同的初始化方式得到不同的结果：
明显构造函数初始化列表的方式得到更高的效率。

