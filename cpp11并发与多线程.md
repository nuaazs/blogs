## 1. 并发基本概念及实现，进程和线程基本概念

### 1.1 并发、进程、线程的基本概念和综述

#### 1.1.1 并发

两个或多个的任务（独立的活动）同时发生（进行）；一个程序同时执行多个独立的任务。

以往计算机，单核CPU某一个时刻只能执行一个任务，由操作系统调度，每秒钟进行多次所谓的“任务切换”。这是一个中并发的假象（不是真正的并发），这种切换也称为上下文切换，是要有时间开销的，比如操作系统保存切换时的各种状态，执行进度等信息，都需要时间，一会还需切换回来的时间复原这些信息。

硬件发展，出现了多处理器计算机，用户服务器和高性能计算机领域。台式机：在一块芯片上有多核（多个）CPU：双核、4核、8核、10核…，能够实现真正的并行执行多个任务（硬件并发）。

对于任务多于CPU核数，如果有4个任务，当时CPU只有2个核，这样还需要切换任务。

使用并发的原因：同时可以做多个事情，提高性能。

#### 1.1.2 并行

当系统有一个以上CPU时,则线程的操作有可能非并发。当一个CPU执行一个线程时，另一个CPU可以执行另一个线程，两个线程互不抢占CPU资源，可以同时进行，这种方式我们称之为并行

> 并发就是你有个伙计在干活，他做事能见缝插针。并行就是你有多个这样的伙计。

#### 1.1.3 可执行文件

磁盘上的一个文件，windows下，一个扩展名为.exe的文件，linux下，文件权限带x的文件（rwxrwxrwx,x为执行权限）。

#### 1.1.4 进程

可执行程序能够运行，window下，直接双击可执行程序来运行；linux下，./文件名就可以执行程序。

进程，就是一个可执行程序运行起来了，就创建了一个进程。

进程就是运行起来的可执行程序。

#### 1.1.5 线程

每个进程（执行起来的可执行程序），都有一个主线程，这个主线程是唯一的，也就是一个进程中只能有一个主线程。

当你执行可执行该程序，产生一个进程后，这个主线程就随这个进程默默的启动起来。

`ctrl+f5`运行测试程序时 ，实际上是进程的主线程来执行（调用）这个main函数中的代码，main中的代码执行完，主线程也就执行了，整个进程也运行结束了。主线程执行完main中return后，表示整个进程执行完毕。

**进程和主线程唇齿相依，有你必然有我，有我必然有你，没有我必然没有你。主线程结束，进程同样也结束；进程结束，主线程同样也结束。**

**线程：**是用来执行代码的。可以理解成一条代码的执行道路：从main开始到main结束。

除了主线程外，我们可以通过自己写代码来创建其他线程，其他线程走的是别的道路，甚至去不同的地方。

**每创建一个新线程，我就可以在同一个时刻，多干一个不同的事（多走一条不同的代码执行路径）。**

线程并不是越多越好，每个线程，都需要一个独立的堆栈空间（大约1M），线程之间的切换要保存很多中间状态，切换会耗费本该属于程序运行的时间。总之，**线程太多会浪费内存并且线程上下文切换会浪费本该执行任务的时间。**

**多线程（并发）:**实际中，必须使用多线程的案例。开发一个游戏服务器。游戏服务器需要给多个玩家提供服务，多个玩家同时连到服务器上。就如玩家一需要充值，游戏服务器要请求充值服务器来完成充值任务，在一个过程中需要耗费10s左右。在这10s内，玩家二发起了抽卡的操作，不可能让玩家二等待10s。所以需要多线程，一个线程处理充值服务，一个线程处理抽卡服务。玩家一般多于线程数，但是某些线程执行非常快。

#### 1.1.6 总结

- 线程是用来执行代码的。
- 线程可以理解成一条代码执行的道路，一个新线程代表一条新的通道。
- 一个进程自动包含一个主线程，主线程随着进程默默的启动并运行，我们可以通过编码来创建多个其他线程（非主线程）。
- 创建进程的数量最大不建议超过200~300，至于到底多少个合适 ，大家在实际的项目中可以不断调整和优化。
- 有时候太多的线程，效率反而会降低，上下文切换会占用太多没有意义的时间。
- 因为主线程是自动启动的，所以一个进程最少有一个线程（主线程）。
- 多线程程序可以同时做多件事，所以运行效率高。但是到底有多高，并不是一个很容易评估和量化的东西。大家仍旧需要在实际编程中，进行体会和调整优化。
- 有时候，多个线程访问同一个资源，多线程效率的提升不明显。

#### 1.1.7 启发

- 开发多线程程序：实力的体现，一个是商用的必须需求；
- 线程开发有一定难度，实现代码更复杂，理解更困难，需要一定的学习时间。
- C++线程会涉及很多新概念，对于C++道路上的成长特别关键，不要急于求成。
- 网络通讯，网络服务器，网线方向必须学习多线程，



### 1.2 并发的实现方法

#### 1.2.1 实现并发的手段

- 通过多个进程实现并发。
- 在单独的进程中，创建多个线程来实现并发，通过自己写带amll创建除主线程之外的其他线程。

#### 1.2.2 多进程并发

word启动后就是一个进程，ie浏览器启动后就是一个进程。

进程之间需要通信：

- 同一个电脑上： 管道，文件，消息队列，共享内存
- 不在同一个电脑上：socket通信技术

#### 1.2.3 多线程并发

单个进程中，创建了多个线程。

线程：类似于轻量级的进程，每个线程都有自己独立的运行路径，但是一个进程中的所有线程共享地址空间（共享内存）。

全局变量，指针，引用可以在线程之间传递，所以使用多线程的开销远远小于多进程。

共享内存带来了新问题：数据一致性问题。比如有两个线程：线程A和线程B可能同时往一个内存块中写数据，这样将会出现问题。

**多进程和多线程并发虽然可以混合使用，但是建议优先考虑多线程技术手段，而不是多进程。**

**注意：**本课程只讲多线程开发技术，后续所讲的并发，都是多线程并发。

#### 1.2.4 总结

和进程相比，线程有如下优点：

- 线程启动速度更快，更轻量级；
- 系统资源开销更少，执行速度更快，比如共享内存这种通信方式比任何其他的通信方式都快。

缺点：

- 使用有一定的难度，要更小心处理数据的一致性问题。

  

### 1.3 C++新标准线程库

windows: `CreateThread()`，`_beginthread()`,`_beginthreadexe()`创建线程

Linux: `pthread_create()`创建线程

以往的多线程代码不能跨平台。

有人使用POSIX thread(pthread)库，这是一个跨平台的库，但是需要做配置，比较复杂，用起来不方便。

C++11新标准，C++本身增加了多线程的支持，意味着可移植性（跨平台），这就大大减少开发人员工作量了。以前都是使用操作系统提供的一些接口来创建多线程。





## 2. 线程启动、结束、创建线程多种方法、join、detach

### 2.1 范例演示线程运行的开始和结束

程序运行起来，生成一个进程。同时进程的运行，该进程所属的主线程开始主动运行。

主线程从main()开始执行，那么自己创建的线程，也需要从一个函数开始执行（初始函数）。一旦初始函数运行完毕，表示创建的线程运行结束。

整个进程是否执行完毕的标志是：主线程是否执行完毕，如果主线程执行完毕了，就代表整个进程执行完毕了。

此时，**一般情况**，如果其他子线程还没有执行完毕，那么这些子线程也会被操作系统**强行终止**。

所以，一般情况下，我们得到一个结论，如果我们想保持子线程（自己用代码创建的线程）的运行状态的话，那么必须让**主线程一直保持运行，不要让主线程运行完毕**（有特殊情况，后面讲解）。

#### 2.1.1 thread

`thread` 是一个标准库里面的类，用于创建线程，并且线程开始执行。线程执行的起点是可执行函数。

```cpp
thrad mythread(myfun);  //myfun是定义的初始函数
```

#### 2.1.2 join()

`join()` 就是阻塞主线程，让主线程等待子线程执行完毕，然后子线程和主线程汇合，然后主线程往后执行。

如果主线程执行完毕了，但子线程没有执行完毕，这种程序是不合格的，写出来的程序也是不稳定的。所以需要使用`join()` 来阻塞主线程，等待子线程执行完成。一个良好的程序，应该是主线程等待子线程执行完毕后，主线程才能最终退出。

#### 2.1.3 detach()

传统的多线程程序主线程需要等待子线程执行完毕，然后自己再最后退出。`detach()` 实现主线程和子线程不汇合（分离），主线程执行自己的，子线程执行自己的，主线程也不必要等待子线程执行完毕。

我们创建了很多子线程，让主线程逐个等待子线程结束，这种编程方法不太好，所以引入了`detach()`。（其实不然，了解就可以）

一旦使用`detach()`之后，与这个主线程关联的`thread`对象就会失去与这个主线程的关联，此时这个子线程就会驻留在后台执行（主线程与该子线程失去联系）。这个子线程就相当于被C++运行时库接管，当这个子线程执行完成后，由运行时库负责清理该线程相关的资源（守护线程）。

`detach()`使子线程失去我们的控制。一旦调用`detach()`，子线程就会进入后台执行，就不能再用`join()`，否则系统会报告异常。

#### 2.1.4 joinable()

`joinable()`判断是否可以成功使用`join()`或者`detach()`。`joinable()`返回`true`或`false`，`true`表示可以使用`join`或`detach`；`false`表示不能使用`join`或`detach`。

#### 2.1.5 创建线程

创建线程的步骤：

- a) 包含一个头文件`thread`
- b) 创建初始函数
- c) `main`中开始写代码，创建线程和执行线程
- d) `join`阻塞主线程，等待子线程执行完毕

代码：

```cpp
#include <iostream>
#include <thread>

using namespace std;
//线程的初始函数
void myprint()
{
    cout << "我的线程开始执行" << endl;

    cout << "我的线程执行结束了" << endl;
}
int main()
{
    std::cout << "Hello World!\n";
    thread mytobj(myprint); //创建线程
    mytobj.join();//主线程阻塞，当子线程执行完毕，join()执行完毕。
    cout << "I love China!" << endl;

    return 0;
} 
```

这个代码有两个线程在跑，相当于整个程序的执行有两条线同时走，所以可以同时执行两个事。即使一条线被堵住了，另外一条线还可以执行。

join导致线程阻塞，主线程得等它执行完才能继续。

删除join后，执行程序后的输出：

```
Hello World!
我的线程开始执行I love China!

我的线程执行结束了
```

可以看出输出比较混乱，而且会出现异常。

原因主要是：主线程执行完成，但是创建的子线程还没有执行完成，这样就会出现异常。

使用`detach()`示例

代码：

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
//线程的初始函数
void myprint()
{
    cout << "我的线程开始执行" << endl;
    cout << "我的线程开始执行1" << endl;
    cout << "我的线程开始执行2" << endl;
    cout << "我的线程开始执行3" << endl;
    cout << "我的线程开始执行4" << endl;
    cout << "我的线程开始执行5" << endl;
    cout << "我的线程开始执行6" << endl;
    cout << "我的线程执行结束了" << endl;
}
int main()
{
    std::cout << "Hello World!\n";
    thread mytobj(myprint);
    mytobj.detach();
    cout << "I love China1!" << endl;
    cout << "I love China2!" << endl;
    cout << "I love China3!" << endl;
    cout << "I love China4!" << endl;
    cout << "I love China5!" << endl;
    return 0;
}
```

输出：

```
Hello World!
我的线程开始执行I love China1!

I love China2!我的线程开始执行1

I love China3!我的线程开始执行2

我的线程开始执行3I love China4!

我的线程开始执行4I love China5!

我的线程开始执行5
```

不会出现异常，但是发现子线程的内容还没有全部输出，就结束了。这是由于主线程执行完毕，进程就结束了，子线程后面的信息就无法输出。一开始（`detach()`），子线程会转入后台执行，由C++运行时库接管，使用cout就会往屏幕打印信息，但是主线程结束了，就无法显示打印的信息。

`joinable()`使用示例：

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
//线程的初始函数
void myprint()
{
    cout << "我的线程开始执行" << endl;
    cout << "我的线程开始执行1" << endl;
    cout << "我的线程开始执行2" << endl;
    cout << "我的线程开始执行3" << endl;
    cout << "我的线程开始执行4" << endl;
    cout << "我的线程开始执行5" << endl;
    cout << "我的线程开始执行6" << endl;
    cout << "我的线程执行结束了" << endl;
}
int main()
{
    std::cout << "Hello World!\n";
    thread mytobj(myprint);
    if (mytobj.joinable())
    {
        mytobj.join();
    }

    cout << "I love China1!" << endl;
    cout << "I love China2!" << endl;
    cout << "I love China3!" << endl;
    cout << "I love China4!" << endl;
    cout << "I love China5!" << endl;
    return 0;
}
```

输出：

```
Hello World!
我的线程开始执行
我的线程开始执行1
我的线程开始执行2
我的线程开始执行3
我的线程开始执行4
我的线程开始执行5
我的线程开始执行6
我的线程执行结束了
I love China1!
I love China2!
I love China3!
I love China4!
I love China5!
```



### 2.2 其他创建线程的手法

`thread`接收一个可调用对象，来创建线程。可调用对象包括：**函数**、**类对象（重载`()`）**和**lambda**。

#### 2.2.1 类对象

创建一个类，重载`()`。

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
//类
class TA
{
public:
    void operator()()
    {
        cout << "我的线程operator()开始执行" << endl;

        cout << "我的线程operator()执行结束" << endl;
    }

};
int main()
{
    std::cout << "Hello World!\n";
    TA ta;
    thread mytobj(ta);
    if (mytobj.joinable())
    {
        mytobj.join();
    }
    cout << "I love China1!" << endl;
    cout << "I love China2!" << endl;
    cout << "I love China3!" << endl;
    cout << "I love China4!" << endl;
    cout << "I love China5!" << endl;
    return 0;
}
```

**问题分析**:

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
//线程的初始函数
class TA
{
public:
    int & m_i;
    TA(int &i) : m_i(i) {}
    void operator()()
    {
        cout << "我的线程operator()开始执行" << endl;
        cout << "m_i1的值为：" << m_i << endl;
        cout << "m_i2的值为：" << m_i << endl;
        cout << "m_i3的值为：" << m_i << endl;
        cout << "m_i4的值为：" << m_i << endl;
        cout << "m_i5的值为：" << m_i << endl;
        cout << "我的线程operator()执行结束" << endl;
    }

};
int main()
{
    std::cout << "Hello World!\n";
    int myi = 6;
    TA ta(myi);
    thread mytobj(ta);
    if (mytobj.joinable())
    {
        mytobj.detach();
    }
    cout << "I love China1!" << endl;
    cout << "I love China2!" << endl;
    cout << "I love China3!" << endl;
    cout << "I love China4!" << endl;
    cout << "I love China5!" << endl;
    return 0;
}
```

子线程中的对象的引用变量`m_i`绑定了`i`，而`i`绑定的是主线程中的`myi`局部变量。如果使用`detach`，主线程结束了，`myi`局部变量就在内存中被销毁了，子线程没有执行完成，`m_i`找不到对应的值了，而还在打印内容，就会产生不可预料的结果。

如果使用`join()`就不会出现这种情况，因为主线程必须等待子线程执行完毕就会结束，所以`myi`局部变量在内存空间中一直存在。

还有一个问题，使用detach()时，主线程结束，ta对象也是一个局部变量，但是子线程还可以执行。

一旦调用了`detach()`，那主线程执行结束了，这里用的`ta`对象还在吗？

**这个对象不在了。但是没有关系。**

这个对象实际上是被**复制**到线程中去。所以执行完主线程后，ta被销毁，但是所复制的TA对象依旧存在。所以，只要这个TA类对象里没有引用，或指针，那么就不会产生问题。

分析对象如何被复制到线程中入的：

通过调用对象的拷贝构造函数实现的，具体看如下代码。

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
//线程的初始函数
class TA
{
public:
    int & m_i;
    TA(int &i) : m_i(i) 
    {
        cout << "TA构造函数被执行" << endl;
    }
    TA(const TA & ta) : m_i(ta.m_i) 
    {
        cout << "TA拷贝构造函数被执行" << endl;
    }
    ~TA()
    {
        cout << "TA析构函数被执行" << endl;
    }
    void operator()()
    {
        cout << "我的线程operator()开始执行" << endl;
        cout << "m_i1的值为：" << m_i << endl;
        cout << "m_i2的值为：" << m_i << endl;
        cout << "m_i3的值为：" << m_i << endl;
        cout << "m_i4的值为：" << m_i << endl;
        cout << "m_i5的值为：" << m_i << endl;
        cout << "我的线程operator()执行结束" << endl;
    }

};
int main()
{
    std::cout << "Hello World!\n";
    int myi = 6;
    TA ta(myi);
    thread mytobj(ta);
    if (mytobj.joinable())
    {
        mytobj.detach();
    }
    cout << "I love China1!" << endl;
    cout << "I love China2!" << endl;
    cout << "I love China3!" << endl;
    cout << "I love China4!" << endl;
    cout << "I love China5!" << endl;
    return 0;
}
```

输出：

```
Hello World!
TA构造函数被执行
TA拷贝构造函数被执行
I love China1!
I love China2!
I love China3!我的线程operator()开始执行
m_i1的值为：
I love China4!
6
I love China5!
TA析构函数被执行m_i2的值为：6
```

**在创建ta对象后，马上就会出现一次拷贝构造函数调用，将主线程中的ta对象复制到子线程中。**

通过输出可以看出，这里只有一次析构函数调用，该次析构函数调用是主线程中的ta被析构，而子线程中的析构没有显示出来，因为主线程已经提前结束了，没有显示，在后台执行。

将`detach()`改为`join()`时，输出为：

```
Hello World!
TA构造函数被执行
TA拷贝构造函数被执行
我的线程operator()开始执行
m_i1的值为：6
m_i2的值为：6
m_i3的值为：6
m_i4的值为：6
m_i5的值为：6
我的线程operator()执行结束
TA析构函数被执行
I love China1!
I love China2!
I love China3!
I love China4!
I love China5!
TA析构函数被执行
```

注意第一次调用析构函数，是子线程中复制进入的那个TA对象调用的。而第二次是主线程中的ta调用的析构函数。

### 2.3 lambda表达式

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;

int main()
{
    auto mylamthread = [] {
        cout << "我的线程开始执行" << endl;
        cout << "我的线程结束执行" << endl;
    };
    thread mytobj(mylamthread);
    if (mytobj.joinable())
    {
        mytobj.join();
    }
    cout << "I love China1!" << endl;
    cout << "I love China2!" << endl;
    cout << "I love China3!" << endl;
    cout << "I love China4!" << endl;
    cout << "I love China5!" << endl;
    return 0;
}
```





## 3. 线程传参详解，detach()大坑，成员函数做线程函数

### 3.1 传递临时对象作为线程参数

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
void myprint(const int &i, char * pmybuf)
{
    cout << i << endl;
    cout << pmybuf << endl;
    return;
}
int main()
{

    int mvar = 1;
    int & mvary = mvar;
    char mybuf[] = "this it a test1";
    thread mytobj(myprint, mvar, mybuf);
    if (mytobj.joinable())
    {
        mytobj.join();
    }
    cout << "I love China1!" << endl;
    cout << "I love China2!" << endl;
    cout << "I love China3!" << endl;
    cout << "I love China4!" << endl;
    cout << "I love China5!" << endl;
    return 0;
}
```

#### 3.1.1 陷阱一

`myprint`函数中的i虽然是引用，但实际上是**值传递**，并不是`mvar`的引用，即使主线程`detach`了子线程，那么子线程中用`i`值仍是安全的。

通过调式发现，mvar和i的地址是不同的：

```
0x008ff9e4 {1}
0x006ade64 {1}
```

所以，子线程传输参数是引用时，实际上是值传递。这样是安全的，但是不推荐使用。

通过调式发现，pmybuf和mybuf的地址相同：

```cpp
0x005af814 "this it a test1"

0x005af814 "this it a test1"
```

所以在使用detach()是不安全的，主线程结束了，子线程找不对对应地址的值。

**而对于传递指针参数时，`detach`子线程时，绝对会出现问题。**

**所以不推荐使用引用，绝对不能使用指针。**

将`char * pmybuf`改成`const string &pmybuf`，pmybuf的地址和mybuf的地址不相同：

```
0x001dfe54 "this it a test1"
0x0054f45c "this it a test1"
```

所以目前来看是安全的。

#### 3.1.2 陷阱二

```cpp
thread mytobj(myprint, mvar, mybuf);
```

mybuf在什么时候转换成string的呢？

如果主线程执行完成了，`mybuf`都被回收了，系统再将`mybuf`字符数组转换成`string`，这样就会存在问题。实际上是存在这种可能性的。

在创建线程时，直接将`mybuf`转换成`string`对象，这是一个保证在线程肯定有效的对象。

```cpp
thread mytobj(myprint, mvar, string(mybuf));
```

为什么这样写就不会有问题呢？难道就不会再没有转换之前，主线程结束，mybuf被回收吗？下面来证明，这样是一定安全的。

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
class A
{
public:
    int m_i;
    A(int a) : m_i(a) {//类型转换构造函数，可以将一个int整型转换成一个A类对象
        cout << "[A::A(int a)构造函数执行]" << this << endl;
    }
    A(const A & a) : m_i(a.m_i) {
        cout << "[A::A(int a)拷贝构造函数执行]" << this << endl;
    }
    ~A(){
        cout << "[~A::A()析构函数执行]" << this << endl;
    }
};
void myprint(const int &i, const A &pmybuf)
{
    cout << &pmybuf << endl;  //打印pmybuf对象的地址

    return;
}
int main()
{

    int mvar = 1;
    int mysecondpar = 12;
    //thread mytobj(myprint, mvar, mysecondpar); //希望mysecondpar转换成A类型对象传递给myprint的第二个参数
    thread mytobj(myprint, mvar, A(mysecondpar)); //创建临时对象作为参数传递参数
    if (mytobj.joinable())
    {
        mytobj.detach();
        //mytobj.join();
    }

    return 0;
}
```

输出：

```
[A::A(int a)构造函数执行]012FFD34
[A::A(int a)拷贝构造函数执行]0146E2F0
0146E2F0
[~A::A()析构函数执行]012FFD34
[~A::A()析构函数执行]0
```

创建线程的同时，构造临时对象的方法传递参是可行的。临时对象的构造一定会在主线程(main函数)执行完之前，构造出来。

只要用临时构造的对象作为参数传递给子线程，那么就一定能够在主线程执行完毕之前把线程函数的第二个参数构建出来，从而确保即便detach()了，子线程也安全运行。

如果不构造临时对象，而是使用隐式类型转换，这是不安全的。

#### 3.1.3 总结

- 若传递int这种简单的类型，建议都是值传递，不要使用引用

- 如果传递类对象，**避免隐式类型转换，应该全部在创建线程时，构造出临时对象，然后在函数参数中用引用来接收，否则系统还会构建临时对象**，浪费，即构造三个对象。

- 终极结论：**建议不使用detach()，只使用join()，这样就不存局部变量失效导致线程对内存的非法引用问题**。

  

### 3.2 临时对象作为线程参数详解

#### 3.2.1 线程id概念

id是一个数字，每个线程（不管是主线程，还是子线程）实际上都对应着一个id，而且每个id都是不同的。即，不同的线程，它的线程id必然不同。

线程id可以使用C++标准库里面的函数来获取：`std::this_thread::get_id()`来获取。

#### 3.2.2 临时对象构造时机摸索

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
class A
{
public:
    int m_i;
    A(int a) : m_i(a) {//类型转换构造函数，可以将一个int整型转换成一个A类对象
        cout << "[A::A(int a)构造函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    A(const A & a) : m_i(a.m_i) {
        cout << "[A::A(int a)拷贝构造函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    ~A(){
        cout << "[~A::A()析构函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
};
void myprint( const A &pmybuf)
{
    cout << "子线程myprint2的参数地址是" << &pmybuf << "threadid = " << std::this_thread::get_id() << endl;
    return;
}
int main()
{
    int mvar = 12;
    cout << "主线程id是" << std::this_thread::get_id() << endl;
    thread mytobj(myprint, mvar); //希望mysecondpar转换成A类型对象传递给myprint的第二个参数
    if (mytobj.joinable())
    {
        //mytobj.detach();
        mytobj.join();
    }

    return 0;
}
```

输出：

```
主线程id是13572
[A::A(int a)构造函数执行]00E3F74Cthradid = 15000
子线程myprint2的参数地址是00E3F74Cthreadid = 15000
[~A::A()析构函数执行]00E3F74Cthradid = 15000
```

从输出可以发现，**对象A的构造是在子线程中进行的。所以，如果使用detach()就会出现问题，即主线程已经结束了，mvar变量已经被回收，将无法进行对象A的构造。**

致命的问题是在子线程中构造的A类对象。

使用临时对象来作为参数传递给线程：

```cpp
thread mytobj(myprint, A(mvar)); //希望mysecondpar转换成A类型对象传递给myprint的参数
```

输出：

```
主线程id是16836
[A::A(int a)构造函数执行]00BAFA38thradid = 16836
[A::A(int a)拷贝构造函数执行]0061E2A0thradid = 16836
子线程myprint2的参数地址是[~A::A()析构函数执行]00BAFA380061E2A0thradid = 16836
threadid = 10444
[~A::A()析构函数执行]0061E2A0thradid = 10444
```

从输出可以看出，**使用临时对象，所有的A类对象都在main()函数中就已经构建完毕了。所以这是安全的。即使使用detach()，子线程也能正常运行。**



### 3.2 传递类对象、智能指针作为线程参数

下面程序，判断子线程对一个变量值进行修改，会不会影响到主线程中对应的值。答案是：**不会。**

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
class A
{
public:
    mutable int m_i; //m_i是可以修改的，即使在常量对象中
    A(int a) : m_i(a) {//类型转换构造函数，可以将一个int整型转换成一个A类对象
        cout << "[A::A(int a)构造函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    A(const A & a) : m_i(a.m_i) {
        cout << "[A::A(int a)拷贝构造函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    ~A(){
        cout << "[~A::A()析构函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
};
void myprint( const A &pmybuf)
{
    pmybuf.m_i = 199;  //修改该值不会影响main函数
    cout << "子线程myprint2的参数地址是" << &pmybuf << "threadid = " << std::this_thread::get_id() << endl;
    return;
}
int main()
{
    int mvar = 12;
    A myobj(10);

    cout << "主线程id是" << std::this_thread::get_id() << endl;
    thread mytobj(myprint, myobj); //将类对象作为线程参数
    if (mytobj.joinable())
    {
        //mytobj.detach();
        mytobj.join();
    }

    return 0;
}
```

`mutable` 修饰一个变量，表明该变量是可以改变的，即使该变量位于`const`对象中。通过调式上面程序发现，子线程和主线程中的两个对象地址不相同，所以修改子线程中对象不会影响主线程中的对象。

```
0x006ffbb0 {m_i=10 }
0x009ade28 {m_i=199 }
```

两者的地址不同。

如果想做到，子线程中对象的修改影响主线程中的对象，需要使用`std::ref()`函数，使用方式：

```cpp
thread mytobj(myprint, std::ref(myobj)); //将类对象作为线程参数
```

表示子线程对myobj对象的修改将会影响主线程。所有代码：

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
class A
{
public:
    mutable int m_i; //m_i是可以修改的，即使在常量对象中
    A(int a) : m_i(a) {//类型转换构造函数，可以将一个int整型转换成一个A类对象
        cout << "[A::A(int a)构造函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    A(const A & a) : m_i(a.m_i) {
        cout << "[A::A(int a)拷贝构造函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    ~A(){
        cout << "[~A::A()析构函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
};
void myprint( const A &pmybuf)
{
    pmybuf.m_i = 199;  //修改该值不会影响main函数
    cout << "子线程myprint2的参数地址是" << &pmybuf << "threadid = " << std::this_thread::get_id() << endl;
    return;
}
int main()
{
    int mvar = 12;
    A myobj(10);

    cout << "主线程id是" << std::this_thread::get_id() << endl;
    thread mytobj(myprint, std::ref(myobj)); //将类对象作为线程参数
    if (mytobj.joinable())
    {
        //mytobj.detach();
        mytobj.join();
    }

    return 0;
}
```

使用`std::ref`来作用到类对象，这样就不会调用拷贝构造函数，这样可以做到子线程和主线程中作用的对象是同一个对象。

输出：

```
[A::A(int a)构造函数执行]009FFA88thradid = 5080
主线程id是5080
子线程myprint2的参数地址是009FFA88threadid = 12584
[~A::A()析构函数执行]009FFA88thradid = 5080
```

输出中没有调用拷贝构造函数。而且根据调式，子线程和主线程中对象地址相同：

```cpp
0x001ef960 {m_i=10 }  //主线程

0x001ef960 {m_i=199 } //子线程

0x001ef960 {m_i=199 } //主线程
```

传递智能指针：

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;

void myprint(unique_ptr<int> pzn)
{
    cout << *pzn << endl;
}
int main()
{
    unique_ptr<int> myp(new int(100));
    thread mytobj(myprint, std::move(myp)); //智能指针当作参数
    if (mytobj.joinable())
    {
        mytobj.join();
    }
    return 0;
}
```

`std::move()`将独占式智能指针转换到另一个独占式智能指针。

调式发现：pzn和myp的地址相同。

```cpp
0x009bf1a8 {100}
0x009bf1a8 {100}
```

而且执行子线程后，发现`myp`为空指针，也是因为`std::move`将智能指针转换给子线程中的`pzn`，子线程执行完毕，智能指针被释放，`myp`成为一个空指针。

注意这里不能使用`detach()`，只能使用`join()`。因为使用`detach()`，`main()`函数执行完成了，子线程还在执行中，`main()`执行完成会将`myp`的内存被系统回收，而此时子线程中的智能指针指向被系统回收的内存，会导致内存泄露。



### 3.4 成员函数指针做线程函数

使用成员函数指针做线程函数时，需要至少两个参数：

```cpp
thread mytobj(&A::thread_work, myobj, 15);
```

第一个参数是类的**成员函数的地址**，第二个**参数是类对象**，第三个参数是**类的成员函数的参数**。 其中可以使用`std::ref()`作用`myobj`，这样就不会调用拷贝拷贝函数，但是这样不能使用`detach()`，因为这是不安全的。

```cpp
#include "pch.h"
#include <iostream>
#include <thread>

using namespace std;
class A
{
public:
    mutable int m_i; //m_i是可以修改的，即使在常量对象中
    A(int a) : m_i(a) {//类型转换构造函数，可以将一个int整型转换成一个A类对象
        cout << "[A::A(int a)构造函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    A(const A & a) : m_i(a.m_i) {
        cout << "[A::A(int a)拷贝构造函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    ~A() {
        cout << "[~A::A()析构函数执行]" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    void thread_work(int num) //成员函数形式创建子线程
    {
        cout << "子线程thread_work执行了" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
    void operator()(int num) //类对象创建子线程
    {
        cout << "子线程()执行了" << this << "thradid = " << std::this_thread::get_id() << endl;
    }
};
int main()
{
    int mvar = 12;
    A myobj(10);
    thread mytobj(&A::thread_work, myobj, 15); //类成员函数指针当作线程函数
    thread mytobj(&A::thread_work, std::ref(myobj), 15); //不调用拷贝构造函数
    //thread mytobj(std::ref(myobj), mvar); //不调用拷贝构造函数，这样使用detach()就不安全了
    if (mytobj.joinable())
    {
        //mytobj.detach();
        mytobj.join();
    }

    return 0;
}
```



## 4 创建多个线程、数据共享问题分析、案例代码

### 4.1 创建和等待多个线程

创建多个线程的代码示例：

```cpp
#include "pch.h"
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
void myprint(int inum) //线程入口函数
{
    cout << "myprint线程开始执行了，线程编号=" << inum << endl;
    cout << "myprint线程结束执行了，线程编号=" << inum << endl;
}
int main()
{
    int mvar = 12;
    vector<thread> mythreads;
    for (int i = 0; i < 10; i++)
    {
        mythreads.push_back(thread(myprint, i)); //创建了10个线程，并开始执行
    }
    for (auto iter = mythreads.begin(); iter != mythreads.end(); iter++)
    {
        iter->join(); //等待10个线程都返回
    }
    cout << "I love china!" << endl;

    return 0;
}
```

输出：

```
myprint线程开始执行了，线程编号=0
myprint线程结束执行了，线程编号=0
myprint线程开始执行了，线程编号=1
myprint线程结束执行了，线程编号=1
myprint线程开始执行了，线程编号=2
myprint线程开始执行了，线程编号=myprint线程结束执行了，线程编号=2myprint线程开始执行了，线程编号=34
myprint线程结束执行了，线程编号=myprint线程开始执行了，线程编号=3

5
myprint线程结束执行了，线程编号=
5myprint线程结束执行了，线程编号=4
myprint线程开始执行了，线程编号=
6
myprint线程结束执行了，线程编号=6
myprint线程开始执行了，线程编号=7
myprint线程开始执行了，线程编号=8myprint线程结束执行了，线程编号=
myprint线程结束执行了，线程编号=myprint线程开始执行了，线程编号=97
8

myprint线程结束执行了，线程编号=9
I love china!
```

从输出可知，多个线程执行顺序是乱的，跟操作系统内部对线程的运行调度机制有关；主线程等待所有子线程运行结束，最后主线程结束。推荐使用`join`写法，这样程序更稳定。

将`thread`对象放入到容器里管理，看起来像个`thread`对象数组，这对我们一次创建大量的线程并对大量线程进行管理很方便。



### 4.2 数据共享问题分析

#### 只读数据

只读数据是安全稳定的，不需要特别的处理手段，直接读就可以。

#### 有读有写

一个共享容器，2个线程写，8个线程读。如果没有特别的处理，那程序肯定崩溃。

最简单的不崩溃处理，读的时候不能写，写的时候不能读。2个线程不能同时写，8个线程不能同时读。

假设写的动作分10小步，由于任务切换，导致各种诡异事情发生（最可能的诡异事情是崩溃）。

#### 其他案例

北京-深圳的火车票，10个售票窗口，同时都要订同一个座位。



### 4.3 共享数据的保护案例代码

网络游戏服务器，两个自己创建的线程，一个线程收集玩家命令，并把命令数据写到一个队列中。另外一个线程用来从队列中取出玩家发送来的命令，执行玩家需要的动作。假定玩家发送的命令是一个数字。

list用链表实现，频繁的顺序插入数据和删除数据时效率高。`vector`对于随机插入和删除数据效率高。

```cpp
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>

using namespace std;

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {
        for (int i = 0; i < 100000; ++i)
        {
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
        }
    }
    //线程二，把数据从消息队列取出
    void outMsgRecvQueue()
    {
        for (int i = 0; i < 100000; ++i)
        {
            if (!msgRecvQueue.empty())
            {
                //消息队列不为空
                command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
                msgRecvQueue.pop_front(); //移除第一个元素，但不返回
                //这里就考虑数据处理
                //....
            }
            else
            {
                cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << i << endl;
            }
        }
        cout << "end" << endl;
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

程序运行时崩溃。

保护共享数据，操作时，某个线程用代码把共享数据锁住，其他想操作共享数据的线程必须等待解锁。

为了解决多线程保护共享数据问题，引入互斥量概念。



## 5 互斥量概念、用法、死锁演示及解决详解

### 5.1 互斥量(mutex)的基本概念

互斥量就是一个类对象，理解成一个锁。多个线程尝试用`lock()`成员函数来加锁，只有一个线程可以锁成功（成功的标志是`lock()`函数返回），如果没有锁成功，那么流程会卡在`lock()`这里不断的尝试去加锁。

解锁`unlock()`。

互斥量使用要小心，保护数据不多也不少，少了没有到达保护效果，多了影响效率。



### 5.2 互斥量的使用

#### 5.2.1 lock()和unlock()

步骤：先`lock()`，操作共享数据，`unlock()`。

`lock()`和`unlock()`必须成对使用，有`lock()`必然有`unlock()`。调用一个`lock()`，必然应该只调用一次`unlock()`。否则会导致程序崩溃。

```cpp
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {
        for (int i = 0; i < 100; ++i)
        {        
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
            my_mutex.lock();
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            my_mutex.unlock();
        }
    }
    //线程二，把数据从消息队列取出
    bool outMsgLULProc(int & command)
    {
        my_mutex.lock();
        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            my_mutex.unlock();
            return true;
        }
        my_mutex.unlock();
        return false;
    }
    void outMsgRecvQueue()
    {
        int command = 0;
        for (int i = 0; i < 100; ++i)
        {
            bool result = outMsgLULProc(command);
            if (result == true)
            {
                cout << "outMsgRecvQueue()执行" << command << endl;
            }
            else
            {
                cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << command << endl;
            }
        }
        cout << "end" << endl;
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    std::mutex my_mutex; //创建一个互斥量
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

为了防止大家忘记`unlock()`，引入了一个叫`std::lock_guard`的类模板，如果忘记`unlock`，其会自动替我们`unlock()`（跟智能指针`unique_ptr<>`类似）。

#### 5.2.2 std::lock_guard类模板：直接取代lock()和unlock()

如果使用了`lock_guard`之后，再不能使用`lock`和`unlock()`。

```cpp
bool outMsgLULProc(int & command)
{
    std::lock_guard<std::mutex> sbguard(my_mutex);
    if (!msgRecvQueue.empty())
    {
        command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
        msgRecvQueue.pop_front(); //移除第一个元素，但不返回
        return true;
    }
    return false;
}
```

原理：`lock_guard`的构造函数中执行了`mymutex.lock()`，而在析构函数中执行了`mymutex.unlock()`。在创建对象时，自动调用构造函数。这是一个局部变量，return的时候自动调用析构函数。

可以通过给代码段加`{}`，达到解锁的目的，即退出代码段，就会调用析构函数。

```cpp
void inMsgRecvQueue()
{
    for (int i = 0; i < 100; ++i)
    {        
        cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
        {
            std::lock_guard<std::mutex> sbguard(my_mutex);
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
        }
    }
}
```



### 5.3 死锁

张三：站在北京等李四，不挪窝

李四：站在深圳等张三，不挪窝

C++中：

一个互斥量是一个把锁。

比如我有两把锁（死锁这个问题前提条件是：由至少两个锁头也就是两个互斥量才能产生）；金锁（JinLock），银锁(YinLock)；

两个线程A、B

(1) 线程A执行的时候，这个线程先锁金锁，把金锁lock()成功了，它然后去lock银锁；出现了上下文切换

(2) 线程B执行了，这个线程先锁银锁，因为银锁没有被锁，所以银锁会lock()成功，线程B要去lock银锁

此时此刻，死锁产生了

(3) 线程A因为拿不到银锁头，流程走不下去（所有后边代码有解锁金锁头的但是流程走不下去了，所以金锁头解不开）

(4) 线程B因为拿不到金锁头，流程走不下去（所有后边代码有解锁银锁头的但是流程走不下去了，所以银锁头解不开）

线程A、B相互等待。

#### 5.3.1 演示

死锁示例代码：

```cpp
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {
        for (int i = 0; i < 100; ++i)
        {        
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
            {
                my_mutex1.lock();
                my_mutex2.lock();//实际工程中，这两个锁头不一定挨着的，可能他们需要保护不同的数据共享块
                msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
                my_mutex2.unlock();
                my_mutex1.unlock();
            }
        }
    }
    //线程二，把数据从消息队列取出
    bool outMsgLULProc(int & command)
    {
        my_mutex2.lock();
        my_mutex1.lock();
        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            my_mutex1.unlock();
            my_mutex2.unlock();
            return true;
        }
        my_mutex1.unlock();
        my_mutex2.unlock();
        return false;
    }
    void outMsgRecvQueue()
    {
        int command = 0;
        for (int i = 0; i < 100; ++i)
        {
            bool result = outMsgLULProc(command);
            if (result == true)
            {
                cout << "outMsgRecvQueue()执行" << command << endl;
            }
            else
            {
                cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << command << endl;
            }
        }
        cout << "end" << endl;
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    std::mutex my_mutex1; //创建一个互斥量
    std::mutex my_mutex2;
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

程序执行一段时候后，卡死，即出现死锁。

使用lock_gurad的方式类似：

```cpp
std::lock_guard<std::mutex> subguard1(my_mutex2);
std::lock_guard<std::mutex> subguard2(my_mutex1);
```

#### 5.3.2 死锁的一般解决方案

只要保证两个互斥量上锁的顺序一致，就不会出现死锁。解锁的顺序可以随意。

#### 5.3.3 std::lock()函数模板

作用：一次锁住两个或者两个以上的互斥量（至少两个）；它不存在因为多个线程中锁顺序的问题，而引起死锁风险。

std::lock()：如果互斥量中有一个没锁住，它就在那里等着，等所有的互斥量锁住，它才能往下走。

要么两个互斥量都锁住，要么两个互斥量都没锁住。如果一个锁住失败，就会将其他锁住成功解锁（释放）。

`my_mutex1`被锁住了，`my_mutex2`而锁住失败，则会将`my_mutex1`释放。

使用范例：

```cpp
void inMsgRecvQueue()
{
    for (int i = 0; i < 100; ++i)
    {        
        cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
        {
            std::lock(my_mutex1,my_mutex2);//相当于每个互斥量都调用了.lock()，两个参数顺序随意
            //实际工程中，这两个锁头不一定挨着的，可能他们需要保护不同的数据共享块
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            my_mutex1.unlock();
            my_mutex2.unlock();
        }
    }
}
```

#### 5.3.4 `std::lock_guard()`和`std::adopt_lock`

使用范例：

```cpp
//线程二，把数据从消息队列取出
bool outMsgLULProc(int & command)
{

    std::lock(my_mutex1, my_mutex2);//相当于每个互斥量都调用了.lock()
    std::lock_guard<std::mutex> subguard1(my_mutex1, std::adopt_lock);
    std::lock_guard<std::mutex> subguard2(my_mutex2, std::adopt_lock);

    if (!msgRecvQueue.empty())
    {
        command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
        msgRecvQueue.pop_front(); //移除第一个元素，但不返回
        return true;
    }
    return false;
}
void outMsgRecvQueue()
{
    int command = 0;
    for (int i = 0; i < 100; ++i)
    {
        bool result = outMsgLULProc(command);
        if (result == true)
        {
            cout << "outMsgRecvQueue()执行" << command << endl;
        }
        else
        {
            cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << command << endl;
        }
    }
    cout << "end" << endl;
}
```

`std::adopt_lock`是个结构体，起一个标记作用：就是表示这个互斥量已经`lock()`,不需要在`std::lock_guad<std::mutex>`构造函数里面不再对`mutex`对象进行lock()了。

总结： std::lock()一次锁住多个互斥量，谨慎使用（建议一个一个锁）；

## 6. unique_lock详解

`unique_lock`是一个类模板，可以取代`lock_guard`，在实际中，推荐使用`lock_guard`。`lock_guard`取代了mutex的lock()和unlock()。

`unique_lock`也是对mutex对象进行加锁和解锁，但是比`lock_guard`灵活，但是效率低，占用内存大。

通常使用，`unique_lock`可以直接替代`lock_guard`，如下代码：

```
bool outMsgLULProc(int & command)
{

    std::unique_lock<std::mutex> subguard(my_mutex1);
    if (!msgRecvQueue.empty())
    {
        command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
        msgRecvQueue.pop_front(); //移除第一个元素，但不返回
        return true;
    }
    return false;
}
```

#### `unique_lock`第二个参数

```
std::lock_guard<std::mutex> subguard1(my_mutex1, std::adopt_lock);
```

`std::adopt_lock`:表示这个互斥量已经被lock了（你必须要将互斥量提前lock，否则会报异常），通知`lock_guard`不需要在构造函数中lock互斥量了。

- std::adopt_lock

`unique_lock`也可以带`std::adopt_lock`标记，作用相同，就是不希望在`unique_lock`的构造函数中lock互斥量。

```cpp
void inMsgRecvQueue()
{
    for (int i = 0; i < 100; ++i)
    {        
        cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
        {
            my_mutex1.lock(); //先将互斥量lock，后面才能使用unique_lock的std::adopt_lock参数
            std::unique_lock<std::mutex> subguard(my_mutex1,std::adopt_lock);
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
        }
    }
}
```

- `std::try_to_lock`

`std::try_to_lock`：尝试用`mutex`的`lock()`去锁定`mutex`，但如果没有锁定成功，也会立即返回，并不会阻塞在那里。使用`std::try_to_lock`参数时，不能使用`lock()`。

当一个进程睡眠等待时，而且该进程已经锁住了，此时另一个进程需要等待上一个进程解锁才能往下执行。如果使用`std::try_to_lock`，另一个进程就不会阻塞了，而是可以去做别的事。

```cpp
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {
        for (int i = 0; i < 1000; ++i)
        {        
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
            {
                std::unique_lock<std::mutex> subguard(my_mutex1,std::try_to_lock);
                if (subguard.owns_lock()) //拿到了锁
                    msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
                else
                    cout << "inMsgRecvQueue()执行，但没有拿到锁头，只能干点别的事" << i << endl;
            }
        }
    }
    //线程二，把数据从消息队列取出
    bool outMsgLULProc(int & command)
    {
        std::unique_lock<std::mutex> subguard(my_mutex1);

        std::chrono::milliseconds dura(200); //1s = 10000ms 
        std::this_thread::sleep_for(dura); //让程序休息200ms

        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            return true;
        }
        return false;
    }
    void outMsgRecvQueue()
    {
        int command = 0;
        for (int i = 0; i < 1000; ++i)
        {
            bool result = outMsgLULProc(command);
            if (result == true)
            {
                cout << "outMsgRecvQueue()执行" << command << endl;
            }
            else
            {
                cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << command << endl;
            }
        }
        cout << "end" << endl;
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    std::mutex my_mutex1; //创建一个互斥量
    //std::mutex my_mutex2;
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

- std::defer_lock

前提不能提前自己lock互斥量，跟`try_to_lock`的前提相同。作用就是，并没有给mutex加锁，初始化了一个没有加锁的mutex。

我们借着`defer_lock`的话题，引出`unique_lock`的重要成员函数

```
//线程一，把收到的消息放入到一个队列的线程
void inMsgRecvQueue()
{
    for (int i = 0; i < 1000; ++i)
    {        
        cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
        {
            std::unique_lock<std::mutex> subguard(my_mutex1,std::defer_lock); //拿到一个没有加锁的my_mutex1
            subguard.lock(); //调用的是unique_lock对象的成员函数lock()，不用自己unlock，unique_lock能自动解锁
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
        }
    }
}
```

#### unique_lock的成员函数

- lock()
- unlock()：有时候需要解锁，处理非共享代码，处理完共享代码，再加锁，处理共享代码

```
//线程一，把收到的消息放入到一个队列的线程
void inMsgRecvQueue()
{
    for (int i = 0; i < 1000; ++i)
    {        
        cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
        {
            std::unique_lock<std::mutex> subguard(my_mutex1,std::defer_lock); //拿到一个没有加锁的my_mutex1

            subguard.lock(); //调用的是unique_lock对象的成员函数lock()，不用自己unlock，unique_lock能自动解锁
            //处理共享代码
            subguard.unlock();
            //非共享代码处理，不许希望一直锁住，希望先解锁，
            subguard.lock();
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
        }
    }
}
```

- try_lock():尝试给互斥量加锁，如果拿不到锁，则返回false, 如果拿到锁，返回true，该函数不阻塞。

  void inMsgRecvQueue()
  {

  ```
  for (int i = 0; i < 1000; ++i)
  {        
      cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
      {
          std::unique_lock<std::mutex> subguard(my_mutex1,std::defer_lock); //拿到一个没有加锁的my_mutex1
          if (subguard.try_lock() == true) //拿到锁了
              msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
          else
              cout << "inMsgRecvQueue执行，没有拿到锁" << i << endl;
  
      }
  }
  ```

  }

为什么有时候需要unlock()?

因为lock()锁住的代码段越少，执行越快，整个程序运行效率越高。有人将把锁头锁住的代码行数，称为锁的粒度，力度一般用粗细来描述。

```
- 锁住的代码少，粒度细，执行效率高。
- 锁住的代码多，粒度粗，执行效率低。
```

要学会尽量选择合适粒度的代码进行保护，粒度太细，可能漏掉共享数据的保护；粒度太粗，影响效率。

- release() ：返回它所管理的mutex对象指针，并释放所有权；即`unique_lock`和mutex不再有关系。

如果原来mutex对象处于加锁状态，有责任接管过来，并解锁。release()返回原始的mutex指针。

```
//线程一，把收到的消息放入到一个队列的线程
void inMsgRecvQueue()
{
    for (int i = 0; i < 1000; ++i)
    {        
        cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
        {
            std::unique_lock<std::mutex> subguard(my_mutex1); //拿到一个没有加锁的my_mutex1
            std::mutex * ptr = subguard.release(); //现在需要自己解锁

            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            ptr->unlock(); //解锁
        }
    }
}
```

#### unique_lock所有权的传递

所有权概念：

```
std::unique_lock<std::mutex> subguard1(my_mutex1);
```

subguard1拥有`my_mutex1`的所有权。subguard1可以把自己拥有的mutex（`my_mutex1`）的所有权转移给其他的`unique_lock`对象。 与智能指针`unique_ptr`类似。

所有权不能进行复制，只能进行转移。

```
std::unique_lock<std::mutex> subguard2(std::move(subguard1)); //移动语义，现在相等于subguard2和my_mutex1绑定在一起，subguard1指向空，subguard2指向了my_mutex1
```

std::move()返回的右值（左值转右值），这样就会调用移动构造函数，所以就是转移，而不是复制。

```
//线程一，把收到的消息放入到一个队列的线程
void inMsgRecvQueue()
{
    for (int i = 0; i < 1000; ++i)
    {        
        cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
        {
            std::unique_lock<std::mutex> subguard(my_mutex1); //拿到一个没有加锁的my_mutex1
            std::unique_lock<std::mutex> subguard2(std::move(subguard));
            std::mutex * ptr = subguard2.release(); //现在需要自己解锁

            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            ptr->unlock(); //解锁

        }
    }
}
```

所有权转移的方法：

- std::move
- return std::unique_lock[std::mutex](std::mutex) : 通过函数返回`std::unique_lock`对象。

代码实现：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

class A
{
public:
    std::unique_lock<std::mutex> rtn_unique_lock()
    {
        std::unique_lock<std::mutex> tmpguard(my_mutex1);
        return tmpguard; //从函数返回一个局部的unique_lock对象是可以的，移动构造函数，返回这种局部
        //对象tempguard会导致系统生成临时unique_lock, 并调用unique_lock的移动构造函数
    }
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {
        for (int i = 0; i < 1000; ++i)
        {        
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
            {
                std::unique_lock<std::mutex> subguard = rtn_unique_lock();
                msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            }
        }
    }
    //线程二，把数据从消息队列取出
    bool outMsgLULProc(int & command)
    {
        std::unique_lock<std::mutex> subguard(my_mutex1);

        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            return true;
        }
        return false;
    }
    void outMsgRecvQueue()
    {
        int command = 0;
        for (int i = 0; i < 1000; ++i)
        {
            bool result = outMsgLULProc(command);
            if (result == true)
            {
                cout << "outMsgRecvQueue()执行" << command << endl;
            }
            else
            {
                cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << command << endl;
            }
        }
        cout << "end" << endl;
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    std::mutex my_mutex1; //创建一个互斥量
    //std::mutex my_mutex2;
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

## 7 单例设计模式共享数据分析、解决，call_once

### 7.1 设计模式讨论

设计模式：开发中，代码的一些写法，程序灵活，维护起来可能方便，但是别人接管、阅读比较困难。

用设计模式理念，写出来的代码比较晦涩的。

老外为了应付特别大的项目时，把项目的开发经验、模块划分经验总结整理成设计模式（现有开发需求，后有理论总结和整理）。设计模式被应用到中国，不太一样，拿着一个程序（项目）往设计模式上套。导致一个小小的项目，非要弄几个设计模式进去，本末倒置。

设计模式肯定有它独特的优点，要活学活用，不要深陷其中，生搬硬套。

### 7.2 单例设计模式

单例设计模式使用频率比较高。

单例：整个项目中，有某个或者某些特殊的类，属于该类的对象，我们只能创建1个，创建多个时编译器报错。

单例类：保证只生成一个对象。

单例类范例：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

//声明一个单例类
class MyCAS {
private:
    MyCAS() {} //私有化构造函数

private:
    static MyCAS *m_instance; //静态成员变量

public:
    static MyCAS * GetInstance()
    {
        if (m_instance == nullptr)
        {
            m_instance = new MyCAS();
            static CGar cl; //静态变量的生成周期到程序结束，所以程序结束时会调用该对象的析构函数
        }
        return m_instance;
    }
    class CGar //类中套类，用来释放对象MyCAS
    {
    public:
        ~CGar() {
            if (MyCAS::m_instance != nullptr)
            {
                delete m_instance;
                MyCAS::m_instance = nullptr;
            }
        }
    };
    void func()
    {
        cout << "测试..." << endl;
    }
};

//类静态变量初始化
MyCAS *MyCAS::m_instance = nullptr;

int main()
{
    MyCAS * p_a = MyCAS::GetInstance(); //创建一个对象，返回该类(MyCAS)对象的指针
    MyCAS * p_b = MyCAS::GetInstance(); //跟p_a是同一个对象
    p_a->func();
    return 0;
}
```

### 7.3 单例设计模式共享数据问题分析、解决

面临的问题：需要在自己创建的线程中创建单例类对象，这种线程可能不止一个。

我们可能会面临GetInstance()这种成员函数要互斥。

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

//互斥量
std::mutex resource_mutex;
//声明一个单例类
class MyCAS {
private:
    MyCAS() {} //私有化构造函数

private:
    static MyCAS *m_instance; //静态成员变量

public:
    static MyCAS * GetInstance()
    {
        //a)如果if(m_instance != nullptr)条件成立，则肯定表示m_instance已经被new过了；
        //b)如果if(m_instance == nullptr)条件成立，则不表示m_instance一定没有被new过。（没有添加互斥量）
        //因为没有互斥量的话，可能刚要new MyCAS()时，被切换出去，进程二还是new MyCAS()，
        //然后第一个进程被切换回去，又new MyCAS()
        if (m_instance == nullptr)  //双重锁定（双重检查），提高效率，
            //除了第一创建对象时，会进入互斥量，其他时候都不会进入互斥量
        {
            std::unique_lock<std::mutex> subguard(resource_mutex); //自动加锁和解锁
            if (m_instance == nullptr)
            {
                m_instance = new MyCAS();
                static CGar cl;
                //静态变量的生成周期到程序结束，所以程序结束时会调用该对象的析构函数
            }
        }
        return m_instance;
    }
    class CGar //类中套类，用来释放对象MyCAS
    {
    public:
        ~CGar() {
            if (MyCAS::m_instance != nullptr)
            {
                delete m_instance;
                MyCAS::m_instance = nullptr;
            }
        }
    };
    void func()
    {
        cout << "测试..." << endl;
    }
};

//类静态变量初始化
MyCAS *MyCAS::m_instance = nullptr;

void mythread()
{
    cout << "我的线程开始执行了" << endl;
    MyCAS * p_a = MyCAS::GetInstance();
    p_a->func();
    cout << "我的线程执行完了" << endl;
}

int main()
{
    std::thread mytobj1(mythread);
    std::thread mytobj2(mythread);
    //虽然这两个线程同一个入口函数，但千万要技术，这是两个线程，
    //所以这里会有两个流程同时执行mythread函数
    mytobj1.join();
    mytobj2.join();
    return 0;
}
```

为什么需要两个if?

```
static MyCAS * GetInstance()
{
    if (m_instance == nullptr)  //双重锁定（双重检查），提高效率，
        //除了第一创建对象时，会进入互斥量，其他时候都不会进入互斥量
    {
        std::unique_lock<std::mutex> subguard(resource_mutex); //自动加锁和解锁
        if (m_instance == nullptr)
        {
            m_instance = new MyCAS();
            static CGar cl;
            //静态变量的生成周期到程序结束，所以程序结束时会调用该对象的析构函数
        }
    }
    return m_instance;
}
```

有人会提问为什么需要两个if，里层的if可以删除啊。

其实一定需要两个if，才能提高效率，保证对象创建一次。因为第一次创建时，进程一可能执行到创建刚要创建对象时，被切换出去。此时，进程二被切换进来，此时判断m_instance仍然为nullptr，但是发现被加锁了，所以会阻塞。等待线程一执行完毕，此时对象已经创建完成。如果没有里层的if，进程二切换进来时，仍然会再次创建一个对象。这样就出现了问题。

### 7.4 std::call_once()

`call_once()`是C++11引入的函数，该函数的第二个参数是一个函数名，`call_once()`能够保证第二个参数对应的函数只被调用一次。

`call_once()`具有互斥量这种更耐穿，而且效率上，比互斥量消耗的资源更少。`call_once()`需要与一个标记结合使用，这个标记是`std::once_flat`，其中`once_flag`是一个结构。`call_once()`就是通过这个标记来决定对应的函数是否执行，调用`call_once()`成功后，`call_once()`就把这个标记设置成为一种已调用的状态。后续再次调用`call_once()`，只要`once_flag`被设置为了“已调用”状态，那么对应的函数就不会再被执行了。

能够解决单例对象在多线程下初始化时，需要使用互斥量。

使用`call_once()`来修改上面的代码：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

//互斥量
std::mutex resource_mutex;
std::once_flag g_flag; //这个一个系统定义的标记


//声明一个单例类
class MyCAS {
    static void CreateInstance() //只被调用一次的函数
    {
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura); //休息2s
        cout << "CreateInstance()执行完毕" << endl;
        m_instance = new MyCAS();
        static CGar cl;
    }
private:
    MyCAS() {} //私有化构造函数

private:
    static MyCAS *m_instance; //静态成员变量

public:
    static MyCAS * GetInstance()
    {
        std::call_once(g_flag, CreateInstance);
        //假设两种线程同时执行到上行代码，其中一个线程要等待另一个线程执行完毕CreateInstance()，
        //才会判断是否能够再次执行
        cout << "call_once执行完毕" << endl;
        return m_instance;
    }
    class CGar //类中套类，用来释放对象MyCAS
    {
    public:
        ~CGar() {
            if (MyCAS::m_instance != nullptr)
            {
                delete m_instance;
                MyCAS::m_instance = nullptr;
            }
        }
    };
    void func()
    {
        cout << "测试..." << endl;
    }
};

//类静态变量初始化
MyCAS *MyCAS::m_instance = nullptr;

void mythread()
{
    cout << "我的线程开始执行了" << endl;
    MyCAS * p_a = MyCAS::GetInstance();
    p_a->func();
    cout << "我的线程执行完了" << endl;
}

int main()
{
    std::thread mytobj1(mythread);
    std::thread mytobj2(mythread);
    //虽然这两个线程同一个入口函数，但千万要技术，这是两个线程，
    //所以这里会有两个流程同时执行mythread函数
    mytobj1.join();
    mytobj2.join();
    return 0;
}
```

## 8 `condition_variable`, `waite=`, `notify_one`, `notify_all`

### 8.1 条件变量`std::condition_variable`, `wait()`, `nofity_one()`

线程A和线程B

在线程A等待一个条件满足，比如消息队列中有消息，线程B专门往消息队列中添加消息，此时触发线程A，这就是条件变量。

`std::condition_variable`是一个类，和一个条件相关的一个类，即就是等待一个条件满足，这个类需要和互斥量配合工作。使用的时候，需要生成这个类的对象。

wait()是成员函数，第一个参数是互斥量(被unique_lock绑定的互斥量)，第二参数的为可调用对象(这里使用lambda表达式)。如果lambda表达式返回是false，那么wait()将解锁互斥量，并堵塞在本行。如果lambda表达式返回true，wait()直接返回，向下执行。

```
std::unique_lock<std::mutex> subguard1(my_mutex1);
my_cond.wait(subguard1, [this] { //一个lambda就是一个可调用对象(函数)
    if (!msgRecvQueue.empty())
        return true;
    else
        eturn false;
});
```

堵塞到什么时候呢？堵塞到其他某个线程调用notify_one()成员函数。

如果wait()没有第二个参数，那么就跟第二个参数lambda表达式返回false效果一样（即wait()解锁互斥量，并堵塞在本行，堵塞到其他某个线程调用`notify_one()`）。

`my_cond.notify_one()`尝试把wait()的线程唤醒，执行完该代码，那么outMsgRecvQueue()里边的wait就会被唤醒。

当其他线程用`notify_one()`将本线程中wait()（原来是阻塞的）唤醒后，wait()将恢复执行：

a) 首先wait()不断尝试重新获取互斥量锁（阻塞之前将锁解开了），如果获取不到，那么线程还是会阻塞在wait()处，等着获取互斥量锁（获取互斥量锁，就等于上锁了）。如果获取到，wait()就继续执行b)。

b) 如果wait()有第二个参数（lambda表达式,或者其他函数），就判断lambda表达，如果表达式为false，那么wait()又对互斥量进行解锁，然后继续阻塞，等待再一次被`notify_one`唤醒。如果表达式为true，则wait()返回true，继续往下执行其他代码（此时互斥量被上锁）。

如果waite()没有第二个参数，就wait返回，继续往下执行其他代码。

示例代码

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {
        for (int i = 0; i < 1000; ++i)
        {
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
            std::unique_lock<std::mutex> subguard(my_mutex1);
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            my_cond.notify_one();//尝试把wait()的线程唤醒，执行完这行，那么outMsgRecvQueue()里边的wait就会被唤醒
        }
    }
    //线程二，把数据从消息队列取出
    void outMsgRecvQueue()
    {
        int command = 0;
        while (true)
        {
            std::unique_lock<std::mutex> subguard1(my_mutex1);
            my_cond.wait(subguard1, [this] { //一个lambda就是一个可调用对象(函数)
                if (!msgRecvQueue.empty())
                    return true;
                else
                    return false;
            });
            command = msgRecvQueue.front();
            msgRecvQueue.pop_front();
            subguard1.unlock(); //因为unique_lock的灵活性，可以随时的unlock()，以免锁住太长时间
            cout << "outMsgRecvQueue执行，取出一个元素" << command << endl;
        }
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    std::mutex my_mutex1; //创建一个互斥量
    std::condition_variable my_cond; //生成一个条件变量对象
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

如果没有阻塞在wait()时，`notify_one()`尝试去欢迎wait()时，不会有任何反应。

假如outMsgRecvQueue()正在处理一个事务，需要一段时间，而不是正卡在wait()那里等待`notify_one()`唤醒，那么此时`notify_one()`将不会有任何效果。

```
void outMsgRecvQueue()
{
    int command = 0;
    while (true)
    {
        std::unique_lock<std::mutex> subguard1(my_mutex1);
        my_cond.wait(subguard1, [this] { //一个lambda就是一个可调用对象(函数)
            if (!msgRecvQueue.empty())
                return true;
            else
                return false;
        });
        command = msgRecvQueue.front();
        msgRecvQueue.pop_front();
        subguard1.unlock(); //因为unique_lock的灵活性，可以随时的unlock()，以免锁住太长时间
        cout << "outMsgRecvQueue执行，取出一个元素" << command << endl;

        //需要处理的事务
        //程序执行到这里，另一个线程调用notify_one()时，不会有效果
        //需要一段时间
    }
}
```

上述的程序还不完美，存在一些问题。

### 8.2 上述代码的深入思考

(1) msgRecvQueue队列中数据可能不止一条，这样outMsgRecvQueue()来不及处理，所以应该考虑对inMsgRecvQueue往队列中添加消息时，进行限制。

(2) `notify_one()`所在的线程可能先执行，那么此时它没有wait()可以唤醒，所以执行`notify_one()`不会有任何效果，等于没有执行。

(3) 如果没有理解清除`condition_variable()`的工作原理，不建议使用，宁愿使用效率差点，但是有把握是稳定的代码。

### 8.3 `notify_all()`

`notify_one()`只能通知一个线程，即只能唤醒一个线程的wait()。

`notify_all()`同时能够唤醒多个线程。

本范例中使`notify_one()`和`notify_all()`的作用一样：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {
        for (int i = 0; i < 1000; ++i)
        {

            std::unique_lock<std::mutex> subguard(my_mutex1);
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            my_cond.notify_all();//尝试把wait()的线程唤醒，执行完这行，那么outMsgRecvQueue()里边的wait就会被唤醒
        }
    }
    //线程二，把数据从消息队列取出
    void outMsgRecvQueue()
    {
        int command = 0;
        while (true)
        {
            std::unique_lock<std::mutex> subguard1(my_mutex1);
            my_cond.wait(subguard1, [this] { //一个lambda就是一个可调用对象(函数)
                if (!msgRecvQueue.empty())
                    return true;
                else
                    return false;
            });
            command = msgRecvQueue.front();
            msgRecvQueue.pop_front();
            cout << "outMsgRecvQueue执行，取出一个元素" << command << "threadid = " << std::this_thread::get_id() << endl;
            subguard1.unlock(); //因为unique_lock的灵活性，可以随时的unlock()，以免锁住太长时间
        }
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    std::mutex my_mutex1; //创建一个互斥量
    std::condition_variable my_cond; //生成一个条件变量对象
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myOutMsgObj2(&A::outMsgRecvQueue, &myobja);
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myOutMsgObj2.join();
    myInMsgObj.join();

    return 0;
}
```

## 9 async、future、package_task、promise

### 9.1 std::async、std::future创建后台任务并返回值

希望线程返回一个结果。

std::async是一个函数模板，是用来启动一个异步任务，启动一个异步任务后，返回一个std::future对象，std::future是一个类模板。

启动一个异步任务，就是自动创建一个线程，并开始执行对应的线程入口函数，返回一个std::future对象。std::future对象中包含线程入口函数所返回的结果（线程返回的结果）。我们可以通过调用一个future对象的成员函数get来获取结果。

std::future提供了一种访问异步操作结果的机制。即这个结果没有办法马上得到，但是在不就的将来线程执行完毕的时候，就能得到结果。所以大家可以这么理解，future对象里会保存一个值，在将来的某个时刻拿到这个值。

具体实现代码：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

//线程入口函数
int mythread()
{
    cout << "mythread() start threadid " << this_thread::get_id() << endl;
    //程序休息
    std::chrono::milliseconds dura(5000);
    std::this_thread::sleep_for(dura);
    cout << "mythread() end threadid " << this_thread::get_id() << endl;
    return 5;
}

int main()
{
    cout << "main " << this_thread::get_id() << endl;
    std::future<int> result = std::async(mythread);
    cout << "continue.... !" << endl;
    cout << result.get() << endl; //主线程阻塞在本行，等待mythread()执行完毕，并返回结果
    cout << "I love China!" << endl;
    return 0;
}
```

输出：

```
main 10288
continue.... !
mythread() start threadid 19484
mythread() end threadid 19484
5
I love China!
```

程序通过std::future对象的get()成员函数等待线程执行结束并返回结果。如果线程没有执行完毕，得不到返回值，主线程将会阻塞在get()代码处。

std::future对象还有一个wait()成员函数，其只能等待线程执行结束，但是无法获取线程的返回值。

注意get成员函数只能调用一次，调用多次会报异常。

使用成员函数当作线程的入口函数，代码如下：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;



class A {
public:
    //线程入口函数
    int mythread(int mypara)
    {
        cout << mypara << endl;
        cout << "mythread() start threadid " << this_thread::get_id() << endl;
        //程序休息
        std::chrono::milliseconds dura(5000);
        std::this_thread::sleep_for(dura);
        cout << "mythread() end threadid " << this_thread::get_id() << endl;
        return 5;
    }
};

int main()
{
    A myobj;
    int mypara = 12;
    cout << "main " << this_thread::get_id() << endl;
    std::future<int> result = std::async(&A::mythread, &myobj, mypara); //成员函数当作线程入口函数
    //第二参数是对象引用，才能保证线程中使用同一个对象
    cout << "continue.... !" << endl;
    cout << result.get() << endl; //主线程阻塞在本行，等待mythread()执行完毕，并返回结果
    cout << "I love China!" << endl;
    return 0;
}
```

还可以额外先std:async传递一个参数，该参数类型为std::launch类型（枚举类型）来达到一些特殊目的。比如：

1) std::launch::deferred 表示线程入口函数调用被延迟到std::future的wait()或者get()被调用时才执行。如果wait()或者get()没有被调用，那么线程会执行吗？不会执行，实际上线程都没有被创建。

```
std::future<int> result = std::async(std::launch::deferred, &A::mythread, &myobj, mypara);
```

输出：

```
main 14200
continue.... !
12
mythread() start threadid 14200
mythread() end threadid 14200
5
I love China!
```

我们发现，主线程和子线程id相同，所以根本就没有创建子线程。延迟调用，没有创建子线程，是在主线程中调用的线程入口函数。

2） std::launch::async 表示在调用async时就创建线程，不需要等到调用get()，子线程就开始执行了。（通过加断定进行调式可以发现子线程执行的过程）

```
std::future<int> result = std::async(std::launch::async, &A::mythread, &myobj, mypara);
```

std:async()系统默认使用std::launch::async | std::launch::defered标记，它可以异步运行或不运行，这取决于系统的负载，但我们无法控制它。

### 9.2 std::packaged_task

`std::packaged_task`是一个类模板，模板参数是各种可调用对象。通过std::packaged_task将各种可调用对象包装起来，方便将来作为线程入口函数使用。

`packaged_task`包装起来的可调用对象还是可以直接调用，所以从这个角度来讲，`packaged_task`对象，也是一个可调用对象。

```
std::packaged_task<int(int)> mypt([](int mypara) {
    cout << mypara << endl;
    cout << "mythread() start threadid " << this_thread::get_id() << endl;
    //程序休息
    std::chrono::milliseconds dura(5000);
    std::this_thread::sleep_for(dura);
    cout << "mythread() end threadid " << this_thread::get_id() << endl;
    return 5;
});
mypt(105); //直接调用
```

lambda表达式做可用对象，并将`packaged_task`对象放入容器中，然后从容器中取出，并直接调用`packaged_task`对象。代码如下：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

vector<std::packaged_task<int(int)> > mytasks; //容器


int main()
{
    cout << "main " << this_thread::get_id() << endl;
    /*lambd表达式做入口函数*/
    std::packaged_task<int(int)> mypt([](int mypara) {
        cout << mypara << endl;
        cout << "mythread() start threadid " << this_thread::get_id() << endl;
        //程序休息
        std::chrono::milliseconds dura(5000);
        std::this_thread::sleep_for(dura);
        cout << "mythread() end threadid " << this_thread::get_id() << endl;
        return 5;
    });
    mytasks.push_back(std::move(mypt)); //需要使用移动语义，放入容器后，mypt变成空
    std::packaged_task<int(int)> mypt2;
    auto iter = mytasks.begin();
    mypt2 = std::move(*iter); //移动语义
    mytasks.erase(iter); //删除第一个元素，迭代器已经失效，后续代码不可以再使用iter
    mypt2(123);
    std::future<int> result = mypt2.get_future();
    cout << result.get() << endl;

    return 0;
}
```

输出：

```
main 10048
123
mythread() start threadid 10048
mythread() end threadid 10048
5
```

### 9.3 std::promise

std::promise是一个类模板。我们能够在某个线程中给它（std::promise）赋值时，然后我们可以在其他线程中将这个值取出来使用。

通过promise保存一个值，在将来某个时刻，可以通过将一个future绑定到这个promise，然后通过future就可以这个得到promise中保存的值。具体代码如下：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

//线程入口函数    
void mythread(std::promise<int> &tmpp, int calc) //注意第一个参数
{
    //执行一系列计算
    calc++;
    calc *= 5;
    //让程序休息5s，来代替其中一系列复杂的计算
    std::chrono::milliseconds dura(5000);
    std::this_thread::sleep_for(dura);
    int result = calc; //假设通过一系列计算得到一个结果result
    tmpp.set_value(result); //将结果保证到tmmp对象中
}


int main()
{
    cout << "main " << this_thread::get_id() << endl;
    std::promise<int> myprom; //声明一个std::promise对象，该对象可以保存值得类型为int
    std::thread t1(mythread, std::ref(myprom), 10);
    t1.join();
    //获取结果
    std::future<int> result = myprom.get_future(); //promise和future绑定，用于获取线程返回值
    cout << "result = " << result.get() << endl;
    cout << "I love China!" << endl;
    return 0;
}
```

输出：

```
main 10600
result = 55
I love China!
```

通过两个子线程来处理，即将一个子线程的值传递给另一个子线程。即实现了两个线程之间的通信。

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

//线程入口函数
void mythread(std::promise<int> &tmpp, int calc) //注意第一个参数
{
    //执行一系列计算
    calc++;
    calc *= 5;
    //让程序休息5s，来代替其中一系列复杂的计算
    std::chrono::milliseconds dura(5000);
    std::this_thread::sleep_for(dura);
    int result = calc; //假设通过一系列计算得到一个结果result
    tmpp.set_value(result); //将结果保证到tmmp对象中
}

void mythread2(std::future<int> &tmpf)
{
    auto result = tmpf.get();
    cout << "mythread2 result " << result << endl;

}

int main()
{
    cout << "main " << this_thread::get_id() << endl;


    std::promise<int> myprom; //声明一个std::promise对象，该对象可以保存值得类型为int
    std::thread t1(mythread, std::ref(myprom), 10);
    t1.join();
    //获取结果
    std::future<int> result = myprom.get_future(); //promise和future绑定，用于获取线程返回值

    std::thread t2(mythread2, std::ref(result));
    t2.join();
    cout << "I love China!" << endl;

    return 0;
}
```

输出：

```
main 12916
mythread2 result 55
I love China!
```

总结：

到底怎么用？什么时候用？

学习这些内容的目的，并不是要把它们用在实际开发中。相反，如果我们能够用最少的内容，写出一个稳定高效的多线程程序，更佳。

为了成长，必须要阅读高手的代码，从而实现自己代码的积累。现在的学习，是为了更好的看懂高手的代码，而不是要在开发中使用。

## 10. future其他成员函数、shared_future、atomic

### 10.1 std::future的其他成员函数

`wait_for`成员函数表示等待子线程执行一段时间。

```
std::future_status status = result.wait_for(std::chrono::seconds(6)); 
```

上面代码表示等在子线程执行6s，此时会在此行代码处阻塞6s，如果子线程6s之内执行完成，则返回`std::future_status::ready`；如果子线程在6s内未执行完毕，则返回`std::future_stauts::timeout`，即子线程执行时间超时了。

还有一种情况就是，将async()的第一个参数设置为`std::launch::deferred`，此时子线程会延迟执行。

```
std::future<int> result = std::async(std::launch::deferred, mythread);
```

执行到`wait_for()`时，直接过去，不会等待一段时间(6s)，因为此时线程还没有开始执行，并且`wait_for()`返回`std::future_status::deferred`。注意使用std::launch::deferred时，不会创建子线程，而是在主线程中执行线程的入口函数代码（前面已经介绍），而且需要调用result.get()，才会执行。

范例代码：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

//线程入口函数
int mythread()
{
    cout << "mythread() start threadid " << this_thread::get_id() << endl;
    //程序休息
    std::chrono::milliseconds dura(5000);
    std::this_thread::sleep_for(dura);
    cout << "mythread() end threadid " << this_thread::get_id() << endl;
    return 5;
}

int main()
{
    cout << "main threadid = " << this_thread::get_id() << endl;
    std::future<int> result = std::async(std::launch::deferred, mythread);
    //std::future<int> result = std::async(mythread);
    cout << "continue.... !" << endl;
    //cout << result.get() << endl; //主线程阻塞在本行，等待mythread()执行完毕，并返回结果
    std::future_status status = result.wait_for(std::chrono::seconds(6)); //等待1/6s
    //futrue_status是一个枚举类型
    if (status == std::future_status::timeout)//超时表示线程还没有执行完,等待1s，希望线程返回
    {
        cout << "超时了，线程还没有执行完！" << endl;
    }
    else if (status == std::future_status::ready)//表示线程成功返回
    {
        cout << "线程执行完毕，成功返回！" << endl;
        cout << result.get() << endl;
    }
    else if (status == std::future_status::deferred)//延迟
    {//如果将async的第一个参数被设置为std::launch::deferred时，则本条件成立
        cout << "线程被延迟执行！" << endl;
        cout << result.get() << endl; //如果使用deferred,则此处调用子线程中的代码，但是不会创建子线程
    }
    cout << "I love China!" << endl;
    return 0;
}
```

### 10.2 std::shared_future

注意future对象result不能两次调用get()，只能调用一次。

```
auto result = tmpf.get();
```

原因是：get()的设计是一个移动语义，相等于将tmpf中的内容移动到result中，tmpf变为空，所以不能再次调用get()。

但是如果有多个线程都需要获取某个线程中的结果，那怎么解决呢？

这是出现了`std::shared_future`类模板。该对象的get()函数不是使用移动语义实现，而是使用复制实现的。所以可以多次调用get()来获取值。

`shared_future`类模板的使用范例：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

//线程入口函数

int mythread(int mypara)
{
    cout << mypara << endl;
    cout << "mythread() start threadid " << this_thread::get_id() << endl;
    //程序休息
    std::chrono::milliseconds dura(5000);
    std::this_thread::sleep_for(dura);
    cout << "mythread() end threadid " << this_thread::get_id() << endl;
    return 5;
}


void mythread2(std::shared_future<int> &tmpf)
{
    cout << "mythread2() start threadid " << this_thread::get_id() << endl;
    auto result = tmpf.get();
    cout << "mythread2 result " << result << endl;
}

int main()
{
    cout << "main " << this_thread::get_id() << endl;

    /*普通函数做入口函数*/
    std::packaged_task<int(int)> mypt(mythread);


    //将mythread函数通过packaged_task包装起来
    std::thread t1(std::ref(mypt), 1); //线程直接开始执行，第二参数是线程入口函数的参数
    t1.join(); //等待线程执行完毕
#if 0
    /*通过future转换成shared_future对象*/
    std::future<int>  result = mypt.get_future(); //线程的返回值赋给result

    //std::shared_future<int> result_s(std::move(result));
    bool ifcanget = result.valid(); //判断result是否为一个有效值
    cout << ifcanget << endl;
    std::shared_future<int> result_s(result.share()); //执行后reuslt为空
    ifcanget = result.valid();
    cout << ifcanget << endl;
#endif 
    /*通过get_future返回值直接构造一个shared_future对象*/
    std::shared_future<int> result_s(mypt.get_future()); 

    /*多次调用get()*/
    auto mythreadresult = result_s.get();
    mythreadresult = result_s.get();

    std::thread t2(mythread2, std::ref(result_s));
    t2.join();
    cout << "I love China!" << endl;

    return 0;
}
```

### 10.3 原子操作std::atomic

#### 原子操作概念引出范例

互斥量：多线程编程中，保护共享数据。首先上锁，操作共享数据，然后开锁。

两个线程，对一个变量进行操作，一个线程往变量中写值，另一个线程读取变量值。

注意C++的一行代码，在汇编中可能是多行代码。

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

int g_mycount = 0; //定义一个全局变量
//线程入口函数
void mythread()
{
    for (int i = 0; i < 1000000; i++)
    {
        g_mycount++;
    }
    return;
}

int main()
{
    std::thread mytobj1(mythread);
    std::thread mytobj2(mythread);

    mytobj1.join();
    mytobj2.join();

    cout << "两个线程执行完毕，最终的g_mycount的结果是： " << g_mycount << endl;
    return 0;
}
```

输出：

```
两个线程执行完毕，最终的g_mycount的结果是： 1893868
```

发现多次执行后会出现结果小于200 0000，这是由于多线程处理时，一个线程在执行加法时，可能被切换到另一个线程执行另外的加法，这样程序就混乱了。可以通过添加互斥量来进行临界区保护，以及使用原子操作来实现。

原子操作可以理解成一种不需要使用互斥量来加锁（无锁）技术的多线程并发编程方式。原子操作是在多线程中不会被打断的程序执行片段。

原子操作是一种无锁技术，所以原子操作的效率比互斥量的效率高。原子操作的局限性在于其针对的一般是一个变量，而互斥量针对的一段代码。

原子操作，一般指不可分隔的操作，即这种操作只存在两种状态，已执行完毕和未开始执行，不可能出现执行一半。

C++11中引入std::atomic来代表原子操作，std::atomic是一个类模板。std::atomic用来封装某个类型的值。

#### std::atomic用法

原子操作范例：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

std::atomic<int> g_mycount = 0; //类对象，封装了一个类型为int的atomic对象
//可以先操作一个int变量一样，操作g_mycount
//线程入口函数
void mythread()
{
    for (int i = 0; i < 10000000; i++)
    {
        /*原子操作*/
        g_mycount++; //对应的操作是一个原子操作，不会被打断
    }
    return;
}

int main()
{
    std::thread mytobj1(mythread);
    std::thread mytobj2(mythread);

    mytobj1.join();
    mytobj2.join();

    cout << "两个线程执行完毕，最终的g_mycount的结果是： " << g_mycount << endl;
    return 0;
}
```

输出：

```
两个线程执行完毕，最终的g_mycount的结果是： 20000000
```

这样输出结果一定是 200 0000。

bool类型的原子操作范例：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

std::atomic<bool> g_ifend = false; //判断线程是否退出，线程退出标记

void mythread()
{
    std::chrono::milliseconds dura(1000);
    while (g_ifend == false)
    {
        cout << "thread id = " << std::this_thread::get_id() << endl;
        cout << "运行中....." << endl;
        std::this_thread::sleep_for(dura);
    }
    cout << "thread id = " << std::this_thread::get_id() << endl;
    cout << "运行结束....." << endl;
    return;
}


int main()
{
    thread mytobj1(mythread);
    thread mytobj2(mythread);
    std::chrono::milliseconds dura(5000);
    std::this_thread::sleep_for(dura);
    g_ifend = true;
    mytobj1.join();
    mytobj2.join();
    cout << "线程执行完毕，退出！" << endl;
    return 0;
}
```

输出：

```
thread id = 4924
thread id = 780运行中.....
运行中.....

thread id = 4924thread id =
780
运行中.....运行中.....

thread id = thread id = 4924
780
运行中.....运行中.....

thread id = thread id = 780
4924
运行中.....运行中.....

thread id = thread id = 780
4924
运行中.....运行中.....

thread id = thread id = 4924
780
运行结束.....运行结束.....

线程执行完毕，退出！
```

#### 总结

原子操作只能针对一个变量进行操作，包括变量的读写。

原子操作一般用**计数或统计**，比如累计发出了多少数据包和接收了数据包等等问题。

## 11 std::atomic续谈、std::async深入谈

### 11.1 std::atomic续谈

atomic原子操作，针对++、–、+=、&=、|=是支持的，其他的操作符可能不支持。

比如：

```
g_mycount = 1 + g_mycount;
```

上述操作是不支持的。程序如下：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

std::atomic<int> g_mycount = 0; //类对象，封装了一个类型为int的atomic对象
//可以先操作一个int变量一样，操作g_mycount
//线程入口函数
void mythread()
{
    for (int i = 0; i < 10000000; i++)
    {
        /*原子操作*/
        //g_mycount++; //对应的操作是一个原子操作，不会被打断
        g_mycount = 1 + g_mycount;
    }
    return;
}

int main()
{
    std::thread mytobj1(mythread);
    std::thread mytobj2(mythread);

    mytobj1.join();
    mytobj2.join();

    cout << "两个线程执行完毕，最终的g_mycount的结果是： " << g_mycount << endl;
    return 0;
}
```

输出的结果不是2000 0000。

atomic还有许多复杂用法，可以自行研究。

### std::async深入谈

#### std::async参数详述

如果系统资源紧张，那么可能创建线程就会失败，那么执行std::thread()时整个程序可能崩溃。

std::async()一般不叫创建线程，而一般称之为创建一个异步任务。

std::async()与std::thread()的最明显区别：就是async有时候可能并不创建线程。

如果用std::launch::deferred来调用async会怎么样？

1. std::launch::deferred： 延迟调用，并且不会创建线程，延迟到调用get()或wait()的时候才执行mythread()，如果没有调用get()或wait()，就不会执行mythread()。
2. std::launch::async: 强制异步任务在新线程上执行，这意味着系统必须创建一个新线程来运行mythread()函数。
3. std::launch::async | std::launch::deferred : 这里`|`关系意味着，调用async()的行为可能是“创建新线程并立即执行入口函数”，或者“不创建新线程并延迟到调用get()才开始执行入口函数”，是两种行为中的一种，但是为那一种是不确定的，由系统确定。
4. 不带额外参数，只给async()函数一个入口函数名：默认值是std::launch::async | std::launch::deferred。系统会自动决定异步（创建新线程）还是同步（不创建新线程）。

但是系统是如何决定是异步还是同步执行呢？

#### std::async和std::thread的区别

std::thread创建线程，如果系统资源紧张，创建线程失败，那么整个程序会报异常并崩溃。

std::thread创建线程的方式，如果线程返回值，不是很容易，需要考虑全局量或引用。

std::async创建异步任务，可能创建线程也可能不创建线程，并且async调用方法可以容易得到线程入口函数的返回值。

由于系统资源限制：

(1) 如果用std::thread创建的线程太多，则可能创建失败，系统报告异常并崩溃。

(2) 如果用std::async，一般就不会报异常崩溃。因为如果系统资源紧张导致无法创建线程的时候，std::async这种不加额外参数的调用就不会创建新线程。而是后续谁调用了get()来请求结果，那么这个异步任务就运行在执行这条get()语句所在的线程上。

如果强制std::async创建新线程，那么就必须使用std::launch::async参数。此时，如果系统资源紧张时，程序崩溃。

(3) 经验：一个程序里面，线程数量不易超多100-200。

#### std::async不确定性问题解决

不加额外参数的std::async的调用，让系统自行决定是否创建新线程。

问题焦点在于：`std::future<int> result = std::async(mythread)`写法，这个异步任务到底有没有被延迟执行。

通过std::future对象的`wait_for()`函数来判断是否创建了新线程，具体代码如下。

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <future>

using namespace std;

//线程入口函数
int mythread()
{
    cout << "mythread() start threadid " << this_thread::get_id() << endl;
    //程序休息
    std::chrono::milliseconds dura(5000);
    std::this_thread::sleep_for(dura);
    cout << "mythread() end threadid " << this_thread::get_id() << endl;
    return 5;
}

int main()
{
    cout << "main threadid = " << this_thread::get_id() << endl;
    //std::future<int> result = std::async(std::launch::deferred, mythread);
    std::future<int> result = std::async(mythread);

    std::future_status status = result.wait_for(0s);//(std::chrono::seconds(0)); //0S
    //futrue_status是一个枚举类型
    if (status == std::future_status::deferred)//延迟
    {/*系统资源紧张，延迟执行*/
        cout << "线程被延迟执行！" << endl;
        cout << result.get() << endl; //如果使用deferred,则此处调用子线程中的代码，但是不会创建子线程
    }
    else
    {
        if (status == std::future_status::timeout)
        {
            cout << "超时，线程没有执行完！" << endl;
            cout << result.get() << endl;
        }
        else if (status == std::future_status::ready)
        {
            /*创建了新线程*/
            cout << "线程成功执行完毕并返回" << endl;
            cout << result.get() << endl;
        }
    }

    cout << "I love China!" << endl;
    return 0;
}
```

## 12 windows临界区、其他各种mutex互斥量

### 12.1 windows临界区

windows临界区跟mutex相似。具体用法：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

#include <windows.h>
using namespace std;

#define __WINDOWSJQ_

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {
        for (int i = 0; i < 100; ++i)
        {
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
#ifdef __WINDOWSJQ_
            EnterCriticalSection(&my_winsec);
            msgRecvQueue.push_back(i);
            LeaveCriticalSection(&my_winsec);        
#else
            my_mutex.lock();
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            my_mutex.unlock();
#endif
        }
    }
    //线程二，把数据从消息队列取出
    bool outMsgLULProc(int & command)
    {
#ifdef __WINDOWSJQ_
        EnterCriticalSection(&my_winsec);
        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            LeaveCriticalSection(&my_winsec);
            return true;
        }
        LeaveCriticalSection(&my_winsec);
#else
        my_mutex.lock();
        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            my_mutex.unlock();
            return true;
        }
        my_mutex.unlock();
#endif
        return false;
    }
    void outMsgRecvQueue()
    {
        int command = 0;
        for (int i = 0; i < 100; ++i)
        {
            bool result = outMsgLULProc(command);
            if (result == true)
            {
                cout << "outMsgRecvQueue()执行" << command << endl;
            }
            else
            {
                cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << command << endl;
            }
        }
        cout << "end" << endl;
    }
    A()
    {
#ifdef __WINDOWSJQ_
        InitializeCriticalSection(&my_winsec);
#endif
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    std::mutex my_mutex; //创建一个互斥量

#ifdef __WINDOWSJQ_
    CRITICAL_SECTION my_winsec;
#endif
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

### 12.2 多次进入临界区试验

在同一个线程中，windows中的“相同临界区变量”代表的临界区进入（EnterCriticalSection）可以被多次调用，但是调用几次临界区进入(EnterCriticalSectoin)，必须调用几次临界区离开（LeaveCriticalSection）。

而在同一个线程中，C++11的互斥量不允许lock多次，否则报异常。

### 12.3 自动析构技术

容器和智能指针就是RAII类。RAII类（Resource Acquistion is initialization）“资源获取及初始化”。

自动析构技术的核心就是设计一个CwinLock类，可以实现自动进入临界区和离开临界区，功能和C++11中的`std::lock_guard<std::mutex>`相似。这样的类称为RAII类。

具体代码：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

#include <windows.h>
using namespace std;

#define __WINDOWSJQ_

//用于自动释放windows下的临界区，防止忘记LeaveCriticalSection导致死锁情况的发生
//功能类似于C++11中的lock_guard
class CWinLock {//RAII类（Resource Acquistion is initialization）“资源获取及初始化”
public:
    CWinLock(CRITICAL_SECTION * pCritmp)
    {
        m_pCritical = pCritmp;
        EnterCriticalSection(m_pCritical);
    }
    ~CWinLock()
    {
        LeaveCriticalSection(m_pCritical);
    }
private:
    CRITICAL_SECTION * m_pCritical;
};

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {
        for (int i = 0; i < 100; ++i)
        {
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
#ifdef __WINDOWSJQ_
            //EnterCriticalSection(&my_winsec);//进入临界区
            //EnterCriticalSection(&my_winsec);
            CWinLock wlock(&my_winsec);
            CWinLock wlock2(&my_winsec);
            msgRecvQueue.push_back(i);
            //LeaveCriticalSection(&my_winsec);
            //LeaveCriticalSection(&my_winsec);//离开临界区
#else
            //my_mutex.lock();
            //my_mutex.lock();
            std::lock_guard<std::mutex> subguard(my_mutex);
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            //my_mutex.unlock();
            //my_mutex.unlock();
#endif
        }
    }
    //线程二，把数据从消息队列取出
    bool outMsgLULProc(int & command)
    {
#ifdef __WINDOWSJQ_
        EnterCriticalSection(&my_winsec);
        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            LeaveCriticalSection(&my_winsec);
            return true;
        }
        LeaveCriticalSection(&my_winsec);
#else
        my_mutex.lock();
        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            my_mutex.unlock();
            return true;
        }
        my_mutex.unlock();
#endif
        return false;
    }
    void outMsgRecvQueue()
    {
        int command = 0;
        for (int i = 0; i < 100; ++i)
        {
            bool result = outMsgLULProc(command);
            if (result == true)
            {
                cout << "outMsgRecvQueue()执行" << command << endl;
            }
            else
            {
                cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << command << endl;
            }
        }
        cout << "end" << endl;
    }
    A()
    {
#ifdef __WINDOWSJQ_
        InitializeCriticalSection(&my_winsec);
#endif
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    std::mutex my_mutex; //创建一个互斥量

#ifdef __WINDOWSJQ_
    CRITICAL_SECTION my_winsec;
#endif
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

### 12.4 recursive_mutex递归的独占互斥量

在实际中，可能出现如下代码：

```
void testfunc1()
{
    std::lock_guard<std::mutex> subguard(my_mutex);
    //执行各种工作
    testfunc2();
}

void testfunc2()
{
    std::lock_guard<std::mutex> subguard(my_mutex);
    //执行各种工作
}
```

这样就会出现异常，程序崩溃。

为了避免这种情况，就出现了`recursive_mutex`。

std::mutex是独占式互斥量，一个线程lock后，其他线程无法lock。

`recursive_mutex`是一个递归的独占互斥量，递归的意思是同一个线程中，同一个互斥量可以进行多次lock。

`recursive_mutex`使用范例：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

#include <windows.h>
using namespace std;

//#define __WINDOWSJQ_

//用于自动释放windows下的临界区，防止忘记LeaveCriticalSection导致死锁情况的发生
//功能类似于C++11中的lock_guard
class CWinLock {//RAII类（Resource Acquistion is initialization）“资源获取及初始化”
public:
    CWinLock(CRITICAL_SECTION * pCritmp)
    {
        m_pCritical = pCritmp;
        EnterCriticalSection(m_pCritical);
    }
    ~CWinLock()
    {
        LeaveCriticalSection(m_pCritical);
    }
private:
    CRITICAL_SECTION * m_pCritical;
};

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {

        for (int i = 0; i < 100; ++i)
        {
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
#ifdef __WINDOWSJQ_
            //EnterCriticalSection(&my_winsec);//进入临界区
            //EnterCriticalSection(&my_winsec);
            CWinLock wlock(&my_winsec);
            CWinLock wlock2(&my_winsec);
            msgRecvQueue.push_back(i);
            //LeaveCriticalSection(&my_winsec);
            //LeaveCriticalSection(&my_winsec);//离开临界区
#else
            //my_mutex.lock();
            //my_mutex.lock();
            std::lock_guard<std::recursive_mutex> subguard(my_mutex);
            testfunc1();
            msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            //my_mutex.unlock();
            //my_mutex.unlock();
#endif
        }
    }
    //线程二，把数据从消息队列取出
    bool outMsgLULProc(int & command)
    {
#ifdef __WINDOWSJQ_
        EnterCriticalSection(&my_winsec);
        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            LeaveCriticalSection(&my_winsec);
            return true;
        }
        LeaveCriticalSection(&my_winsec);
#else
        my_mutex.lock();
        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            my_mutex.unlock();
            return true;
        }
        my_mutex.unlock();
#endif
        return false;
    }
    void outMsgRecvQueue()
    {
        int command = 0;
        for (int i = 0; i < 100; ++i)
        {
            bool result = outMsgLULProc(command);
            if (result == true)
            {
                cout << "outMsgRecvQueue()执行" << command << endl;
            }
            else
            {
                cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << command << endl;
            }
        }
        cout << "end" << endl;
    }
    A()
    {
#ifdef __WINDOWSJQ_
        InitializeCriticalSection(&my_winsec);
#endif
    }

    void testfunc1()
    {
        std::lock_guard<std::recursive_mutex> subguard(my_mutex);
        //执行各种工作
        testfunc2();
    }

    void testfunc2()
    {
        std::lock_guard<std::recursive_mutex> subguard(my_mutex);
        //执行各种工作
    }
private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    //std::mutex my_mutex; //创建一个互斥量

    std::recursive_mutex  my_mutex;

#ifdef __WINDOWSJQ_
    CRITICAL_SECTION my_winsec;
#endif
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

使用`recursive_mutex`时，是否有优化空间？

`recursive_mutex`的效率上比mutex低。

递归次数有限制，递归太多可能报出异常。

### 12.5 带超时的互斥量`std::timed_mutex`和`std::recursive_timed_mutex`

`std::timed_mutex`带超时功能的独占互斥量。

`std::recursive_timed_mutex`带超时功能的递归独占互斥量。其的使用和`std::timed_mutex`一样。允许同一个线程，多次获取同一个互斥量。

`try_lock_for()`: 等待一段时间，如果等待超时没有拿到锁头，流程就走下去。参数是一段时间。

```
std::timed_mutex my_mutex;

std::chrono::milliseconds timeout(100);
if (my_mutex.try_lock_for(timeout)) //等待100毫秒来尝试获取锁
{
    //在100毫秒之内拿到了锁
    msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
    my_mutex.unlock(); 
}
else
{
    //100毫秒内没有拿到锁
    std::chrono::milliseconds sleeptime(100);
    std::this_thread::sleep_for(sleeptime);
}
```

`try_lock_for()`如果返回true，则表示在一段时间内拿到了锁，如果返回false，则表示没有拿到锁。

`try_lock_until()`：参数是未来一个时间点。在这个未来的时间没到的时间那，如果拿到了锁，那么往下执行；如果没有拿到锁，也能往下执行。

```
std::chrono::milliseconds timeout(100);
if (my_mutex.try_lock_until(chrono::steady_clock::now()+timeout)) //等待100毫秒来尝试获取锁
{
    //在100毫秒之内拿到了锁
    msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
    my_mutex.unlock(); 
}
else
{
    //100毫秒内没有拿到锁
    std::chrono::milliseconds sleeptime(100);
    std::this_thread::sleep_for(sleeptime);
}
```

测试代码：

```
#include "pch.h"
#include <iostream>
#include <thread>
#include <list>
#include <mutex>

#include <windows.h>
using namespace std;

//#define __WINDOWSJQ_

//用于自动释放windows下的临界区，防止忘记LeaveCriticalSection导致死锁情况的发生
//功能类似于C++11中的lock_guard
class CWinLock {//RAII类（Resource Acquistion is initialization）“资源获取及初始化”
public:
    CWinLock(CRITICAL_SECTION * pCritmp)
    {
        m_pCritical = pCritmp;
        EnterCriticalSection(m_pCritical);
    }
    ~CWinLock()
    {
        LeaveCriticalSection(m_pCritical);
    }
private:
    CRITICAL_SECTION * m_pCritical;
};

class A
{
public:
    //线程一，把收到的消息放入到一个队列的线程
    void inMsgRecvQueue()
    {

        for (int i = 0; i < 100; ++i)
        {
            cout << "inMsgRecvQueue()执行，插入一个元素" << i << endl;
#ifdef __WINDOWSJQ_
            //EnterCriticalSection(&my_winsec);//进入临界区
            //EnterCriticalSection(&my_winsec);
            CWinLock wlock(&my_winsec);
            CWinLock wlock2(&my_winsec);
            msgRecvQueue.push_back(i);
            //LeaveCriticalSection(&my_winsec);
            //LeaveCriticalSection(&my_winsec);//离开临界区
#else
            //my_mutex.lock();
            //my_mutex.lock();
            //std::lock_guard<std::recursive_mutex> subguard(my_mutex);
            std::chrono::milliseconds timeout(100);
            if (my_mutex.try_lock_until(chrono::steady_clock::now()+timeout)) //等待100毫秒来尝试获取锁
            {
                //在100毫秒之内拿到了锁
                msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
                my_mutex.unlock(); 
            }
            else
            {
                //100毫秒内没有拿到锁
                std::chrono::milliseconds sleeptime(100);
                std::this_thread::sleep_for(sleeptime);
            }
            //testfunc1();
            //msgRecvQueue.push_back(i); //假设i就是收到的命令，直接将其放入消息队列中
            //my_mutex.unlock();
            //my_mutex.unlock();
#endif
        }
    }
    //线程二，把数据从消息队列取出
    bool outMsgLULProc(int & command)
    {
#ifdef __WINDOWSJQ_
        EnterCriticalSection(&my_winsec);
        if (!msgRecvQueue.empty())
        {
            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            LeaveCriticalSection(&my_winsec);
            return true;
        }
        LeaveCriticalSection(&my_winsec);
#else
        my_mutex.lock();
        if (!msgRecvQueue.empty())
        {
            std::chrono::milliseconds sleeptime(1000000000000);
            std::this_thread::sleep_for(sleeptime);

            command = msgRecvQueue.front();//返回第一个元素，但不检查元素是否存在
            msgRecvQueue.pop_front(); //移除第一个元素，但不返回
            my_mutex.unlock();
            return true;
        }
        my_mutex.unlock();
#endif
        return false;
    }
    void outMsgRecvQueue()
    {
        int command = 0;
        for (int i = 0; i < 100; ++i)
        {
            bool result = outMsgLULProc(command);
            if (result == true)
            {
                cout << "outMsgRecvQueue()执行" << command << endl;
            }
            else
            {
                cout << "outMsgRecvQueue()执行，但目前消息队列为空！" << command << endl;
            }
        }
        cout << "end" << endl;
    }
    A()
    {
#ifdef __WINDOWSJQ_
        InitializeCriticalSection(&my_winsec);
#endif
    }


private:
    std::list<int> msgRecvQueue; //容器代表玩家发送过来的命令
    //std::mutex my_mutex; //创建一个互斥量

    //std::recursive_mutex  my_mutex;
    std::timed_mutex my_mutex;

#ifdef __WINDOWSJQ_
    CRITICAL_SECTION my_winsec;
#endif
};


int main()
{
    A myobja;
    std::thread  myOutMsgObj(&A::outMsgRecvQueue, &myobja);  //第二个参数是引用，才能保证线程里用的是同一个线程
    std::thread  myInMsgObj(&A::inMsgRecvQueue, &myobja);
    myOutMsgObj.join();
    myInMsgObj.join();

    return 0;
}
```

## 13 补充知识、线程池浅谈、数量谈、总结

### 13.1 虚假唤醒

虚假唤醒是针对条件变量`condition_variable`的，可能存在唤醒wait时，处理的无数据无效，这就导致虚假唤醒。

虚假唤醒的情形：

(1) `notify_one`多次调用；

(2) 存在多个线程处理数据，使用`notify_all`唤醒时，可能存在虚假唤醒。

处理虚假唤醒的措施：

wait()第二个参数(lambda表达)需要正确判断数据是否有效。

比如，第八节就正确避免了虚假唤醒，当多次唤醒wait()时，我们通过判断队列中是否为空，如果为空，就释放锁，继续等待。

```
my_cond.wait(subguard1, [this] { //一个lambda就是一个可调用对象(函数)
    if (!msgRecvQueue.empty())
        return true;
    else
        return false;
});
```

### 13.2 atomic

第10节和11节有介绍。

atmoic不允许定义时初始化操作，显示“尝试引用已删除的函数”，这是因为编译器内部没有声明拷贝构造函数，而不声明的原因这种赋值操作操作系统不好做成原子操作。

```
atomic<int> atm2 = atm;//不允许
```

同时，也不允许赋值操作：

```
atomic<int> atm2;
atm2 = atm;
```

同样因为尝试引用已删除的函数，拷贝赋值函数也没有声明。

但是原子操作atomic给出了load()函数，以原子方式读atomic对象的值。

```
atomic<int> atm;
atm    = 0;
atomic<int> atm2(atm.load());
auto atm2(atm.load());
```

同时atomic也给出了以原子方式写入内容，即store()函数：

```
atm2.store(12);
```

### 13.4 线程池

#### 场景设想

开发一个服务器程序，每来一个客户端，就创建一个新线程，为该客户提供服务。

网络游戏，玩家数量比较多，不可能给每个玩家创建个新线程，此程序写法在这种场景下不适合。

程序稳定性问题：编写的代码中，偶尔创建一个线程，这种写法，就让人感到不安。不知何时创建一个线程，创建失败。

线程池：把一堆线程放到一起，统一管理。这种统一管理调度，循环利用线程的方式，就称之为线程池。

#### 实现方式

在程序启动时，一次性创建好一定数量的线程。这样更让人放心，觉得程序更稳定。

### 13.5 线程创建数量谈

线程数量的极限问题： 2000个线程是极限，再创建线程程序会崩溃。

线程创建的数量建议:

a) 采用某些技术开发程序；接口提供商建议你创建线程数量= cpu数量， cpu数量 * 2等等；遵照专业建议ehe指示，确保程序高效执行。

b) 创建线程是为了完成一些业务，一个线程等于一条执行通路。如果100个业务堵塞充值线程，这里开110个线程，那是合适的。

c) 线程数量不是越多越好，线程之间需要切换，如果线程数量太多，切换消耗的内存和时间太大。建议，线程的数量不要超过500，控制在200个之内比较理想，具体通过实验来确定线程数量。

### 13.6 C++11多线程总结

windows，Linux都有自己的多线程开发接口。 不跨平台，所以C++11在语言层面提供了跨平台的多线程开发。

thread

join

detach: 参数的传递问题

mutex

lock_guard

unique_lock

call_one()

coindition_variable

windows临界区 EnterCriticalSection, LeaveCriticalSection

wait

notify_one

notify_all

packged_task

promise

async

future

future_shared

timed_mutex

recuresive_mutex

recuresive_timed_mutex

atomic