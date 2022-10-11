## C++ using用法总结

### **配合命名空间，对命名空间权限进行管理**

```cpp
using namespace std;
using std::cout;
```



### 类型重命名

作用等同typedef，但逻辑上更直观。

```cpp
#include <iostream>
using namespace std;

#define DString std::string // !不建议使用
typedef std::string TString; //!使用typedef的方式
using Ustring = std::string; //！使用using typeName_self = stdtypename;
    
//更直观
typedef void (tFunc*)(void);
using uFunc = void(*)(void);

int main(int argc, char *argv[]){
    TString ts("String!");
    Ustring us("Ustring!");
    string s("dfkjf");
    
    cout << ts << endl;
    cout << us <<endl;
    cout << s << endl;
    return 0;
}
```



### 继承体系中，改变部分接口的继承权限

比如我们需要继承一个基类，然后又将基类中的某些public接口在子类对象实例化后对外开放使用。

```cpp
#include <iostream>
#include <typeinfo>

using namespace std;
class Base{
    public:
    Base(){}
    ~Base(){}
    void dis1(){
        cout<<"dis1"<<endl;
    }
    void dis2(){
        cout<<"dis2"<<endl;
    }
};
class BaseA:private Base{
public:
    using Base::dis1;//需要在BaseA的public下释放才能对外使用
    void dis2show(){
		this->dis2();
    }
};

int main(int argc, char *argv[]){
    BaseA ba;
    ba.dis1();
    ba.dis2show();
    return 0;
}
```

