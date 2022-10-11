## C\C++ 中 malloc、calloc、realloc 函数的用法

### 前言

C\C++提供了底层的内存操作，为程序提供了强大的能力。在使用 `malloc() calloc() realloc()` 进行**动态内存分配**时，内存区域中的这个空间称为`堆(heap)`，另一个内存区域，称为`栈(stack)`，其中的空间分配给函数的参数和本地变量，执行完该函数后，存储参数和本地变量的内存空间就会自动释放。而堆中的内存是由人控制的，在分配堆上的内存时，需要人自己来判断什么时候需要分配，什么时候需要释放。



### malloc

1. 函数原型

   ```cpp
   (TYPE *) malloc(SIZE)
   ```

2. 函数功能

   `malloc()`在内存的同台存储区中分配一块长度为SIZE字节的连续区域。参数SIZE为需要的内存空间的长度，返回该区域的地址。
   
   `malloc()`函数不能为所分配的空间初始化值，需要使用`memset()`，不然可能会出现内存中遗留的数据。在程序结束之前，需要使用`free()`进行内存释放。

3. 使用

   ```cpp
   int *p = NULL;
   p = (int *) malloc(sizeof(int)); //申请
   memset(p, 20, sizeof(int)/4); //赋初值（不可取）
printf("%d\n", *p); //20
   free(p); // 释放
   ```
   
   要注意的是，`memset()`是按照字节来进行赋值，而char类型的空间恰好是1字节，所以`memset()`最好用于char类型的赋值。当然，也可以用`sizeof(int)/4`来给int类型赋值，只不过只能赋`0x00-0xff`的值。不然的话就是按位赋值四次。
   
   ```cpp
   char *p = NULL;
   p = (char *)malloc(sizeof(char));
   memset("%c\n", *p); //g
   free(p);
   p=NULL;
   ```
   
   如上，`char` 类型可以完美的赋值。

### calloc

1. 函数原型

   ```cpp
   (TYPE *)calloc(int n,SIZE);
   ```

2. 函数功能

   `calloc()`函数功能和`malloc`类似，都是从堆内存中分配内存，不同的是，`calloc()`会自动进行初始化，每一位都初始化为零。`n`表示元素的个数，`SIZE`为单位元素的长度，从定义上看，`calloc()`适合为数组申请动态内存。与`malloc()`一样最后也需要使用`free()`进行内存释放。

3. 使用

   ```cpp
   #define SIZE 10
   int *p;
p = (int *) calloc(SIZE, sizeof(int));
   for (int i = 0; i< SIZE; ++i){
       p[i]=i;
   }
   for (int j =0; j<SIZE; ++j){
       printf("%d\n",p[j]);
   }
   free(p);
   p = NULL;
   ```
   
   

### realloc

1. 函数原型

   ```cpp
   (TYPE *) realloc(TYPE *ptr, NEW_SIZE);
   ```

2. 函数功能

   `realloc()`是给一个已经分配了地址的指针重新分配动态内存空间，`*ptr`为原有的空间地址，`NEW_SIZE`是重新申请的空间，若新的空间小于之前的空间，则不会进行改变，若新的空间大于原来的空间，则会从堆内存中为`*ptr`分配一个大小为`NEW_SIZE`的空间，同时将原来空间的内容以此复制进新的地址空间，`*ptr`之前的空间被释放，`realloc()`所分配的空间也是未初始化的。

3. 使用

   ```cpp
   #define SIZE 10
   int *p;
p = (int *) calloc(SIZE, sizeof(int));
   for (int i=0; i<SIZE; ++i){
       p[i]=i;
   }
   
   p = (int *) realloc(p, SIZE*2);
   for (int j=10; j<SIZE*2; ++j){
       p[j]=j;
   }
   for (int k=0; k<SIZE*2; ++k){
       printf("%d\n",p[k]); // 1, 2, 3, ... ,19
   }
   free(p);
   ```
   
   

