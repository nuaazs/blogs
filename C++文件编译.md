## g++

通过下面命令可查看g++版本

```
g++ -v
```

结果如下：

![image-20211212104810288](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211212104810288.png)

也可以通过`g++ --help` 查看更多的可用命令。



## 编译单个文件

编写单个文件的可执行程序代码**hello.cpp**如下

```cpp
1 #include <iostream>
2 using namespace std;
3 
4 int main(){
5     cout << "Hello World!" << endl;
6 }
```

用cmd打开该文件所在的相应文件夹，并输入：**g++ hello.cpp**

默认情况下，在该文件夹中将产生：a.exe, 此时在cmd中输入a,就可以看到输出结果。

我们也可以自定义产生的可执行程序名，如test.exe, 我们只要输入：**g++ hello.cpp -o test**

然后就得到test.exe文件，在cmd中输入test就能够得到结果。



## 编译多个文件

定义头文件**header.h**, 头文件包含3个函数声明：

```cpp
int fact(int n);
int static_val();
int mabs(int);
```

定义函数定义文件**func.cpp**：

```cpp
#include "header.h"

int fact(int n)
{
    int ret = 1;
    while(n > 1)
        ret *= n--;
    return ret;
}

int static_val()
{
    static int count = 1;
    return ++count;

}

int mabs(int n)
{
    return (n > 0) ? n : -n;
}
```

定义主函数文件**main.cpp**：

```cpp
#include <iostream>
#include "header.h"
using namespace std;


int main()
{
    int j = fact(5);
    cout << "5! is " << j << endl;
    for(int i=1; i<=5; ++i)
    {
        cout << static_val() << " ";
    }
    cout << endl;
    cout << "mabs(-8) is " << mabs(-8) << endl;
    return 0;
}
```

在同一个文件夹下编辑**header.h**，**func.cpp**，**main.cpp**后，就可以进行多个文件编译，注意到在命令行编译中似乎没有头文件什么事，头文件只是起到声明的作用，因此只需编译两个***.cpp**文件并链接就可以。

输入下面两行分别编译两个文件：

```shell
g++ -c func.cpp
g++ -c main.cpp
```

上面编译完成后生成两个文件：**func.o**，**main.o**

之后通过链接就可以得到最终的可执行程序，输入下面命令：

```shell
g++ main.o func.o -o test
```

### tips

```shell
-o：指定生成可执行文件的名称。使用方法为：g++ -o afile file.cpp file.h ... （可执行文件不可与待编译或链接文件同名，否则会生成相应可执行文件且覆盖原编译或链接文件），如果不使用-o选项，则会生成默认可执行文件a.out。
-c：只编译不链接，只生成目标文件。
-g：添加gdb调试选项。
```

