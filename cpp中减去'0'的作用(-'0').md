![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/Wcoag.jpg)

这种语法问题吧说简单它不那么简单，毕竟不好理解；但说难吧也不难，其实就是让代码更简洁更有逼格的途径而已。

它的作用就是减去0的ASCII值：48。可以方便的用来转换大小写或者数字和和字符。比如我们可以写这么一个函数：



```cpp
#include <cstdio>
#include <iostream>
using namespace std;

int change_chr_num( char x )
{
    return x - '0';
}

int main()
{
    char x;
    cin >> x;
    cout << change_chr_num(x) + 1;
    return 0;
}
/*
Input: 9
Output: 10
*/
```

　　输入的是**字符串类型下的9**，输出的是计算后的**整型10**。同时这个语句等价于`return x-48`，也就是说直接减去ACSII值48效果也是一样的。

```cpp
#include <cstdio>
#include <iostream>
using namespace std;

int change_chr_num( char x )
{
    return x - 48;
}

int main()
{
    char x;
    cin >> x;
    cout << change_chr_num(x) + 1;
    return 0;
}
/*
Input: 9
Output: 10
*/
```

修改一下代码，值还是一样的。

