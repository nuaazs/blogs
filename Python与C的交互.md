# python与c互相调用

　　虽然python开发效率很高，但作为脚本语言，其性能不高，所以为了兼顾开发效率和性能，通常把性能要求高的模块用c或c++来实现或者在c或c++中运行python脚本来处理逻辑，前者通常是python中一些模块的实现方式，后者服务端程序（实现业务扩展或是Plugin功能）和游戏开发（脚本只处理逻辑）中比较常见。本文主要介绍通过在c中运行python脚本来实现python与c的相互调用，并通过c和python脚本设置同一段内存区域为例子来讲解。



## **准备工作**

　　为了在c中运行python脚本，需要在程序链接的时候将**python虚拟机库**链接进去，python虚拟机库是python安装目录下libs中的`python27.lib`文件，至于怎样将库链接进程序中可以自己google下。由于在c中使用了python的一些方法和数据结构，所以需要将python安装目录下的include目录添加到项目include目录中。好了，需要准备的就是这些，然后就可以开始实现一个设置内存区域的例子了。

 

## **内嵌python虚拟机**

　　在c中内嵌python虚拟机很简单，只需要在程序开头`include Python.h`头文件，然后调用下面两段来初始化python虚拟机实例就行了。

```c
Py_SetPythonHome("D:/Python27")
Py_Initialize();
```

`Py_SetPythonHome`函数是用来设置Python的库路径，也就是Python的安装路径，`Py_Initialize`函数真正实例化一个Python虚拟机，这样就把一个Python虚拟机内嵌到C中了。



## **调用python脚本**

 　将python虚拟机初始化后，其实就可以调用python脚本了。c中调用脚本模块中的方法分下面几个步骤：

　　1、使用`PyImport_ImportModule`导入脚步模块；

　　2、使用`PyObject_GetAttrString`获取模块特定方法信息；

　　3、使用`Py_VaBuildValue`转换输入参数；

　　4、使用`PyObject_CallObject`调用特定方法；

　　5、使用`PyArg_Parse`转换方法的返回结果。

　　由于上面流程在调用模块中的方法都是必须的，所以可以写个函数来封装上面的5个步骤，具体代码如下：

```c
int PyModuleRunFunction(const char *module, const char *function,
                        const char *result_format, void *result, const char *args_format, ...)
{
    Pyobject *pmodule, *pfunction, *args, *presult;
    pmodule = PyImport_ImportModule(const_cast<char *>(module));
    if (!pmodule)
    {
        PyObject *type = PyErr_Occurred();
        if (type == PyExc_NameError)
        {
            PyErr_Clear();
            return 0;
        }
        PyError("PyModuleRunFunction");
        return -1
    }
    
    pfunction = PyObject_GetAttrString(pmodule, const_cast<char *>(function));
    Py_DECREF(pmodule);
    if(!pfunction)
    {
        PyObject *type = PyErr_Occurred();
        if (type == PyExc_AttributeError)
        {
            PyErr_Clear();
            return 0;
        }
        PyError("PyModuleRunFunction");
        return -2;
    }
    
    if (pfunction == Py_None)
    {
        return 0;
    }
    
    va_list args_list;
    va_start(args_list, args_format);
    
    args = Py_VaBuildValue(const_cast<char *>(args_format), args_list);
    va_end(args_list);
    
    if (!args)
    {
        Py_DECREF(pfunction);
        return -3;
    }
    
    presult = PyObject_CallObject(pfunction, args);
    if (presult == 0)
    {
        PyError("PyModuleRunFunction");
        Py_XDECREF(pfunction);
        Py_XDECREF(args);
        return -1;
    }
    Py_XDECREF(pfuntion);
    Py_XDECREF(args);
    return ConvertResult(presult, result_format, result);
}
```

有了上面的调用python模块内方法的通用函数，我们就可以直接调用python脚本中的方法了，具体如下：

```c
PyModuleRunFunction("hello","test","",0,"()")
```



## **初始化c实现的python模块**

为了能在python脚本中调用到c中定义的方法，需要先在c中定义一个python模块，然后在脚本中import这个模块，最后通过这个模块来间接调用c中定义的方法。例如，我们通过c定义了一块内存区域data和对这个内存区域操作的函数SetData与GetData（代码如下），怎样在脚本中调用SetData与GetData函数来操作data呢？其实关键问题是怎么样在脚本中调用SetData和GetData函数，如果能在脚本中调用这两个函数，自然就能操作data了。python中通过模块的方式来解决这个问题。

```c
#define min(a,b) (((a) < (b)) ? (a) : (b))
char data[1024];

void SetData(const char *str)
{
    strncpy(data,str,min(strlen(str)+1,1024));
}
const char *GetData()
{
    return data;
}
```

在c中定义一个python模块有特定的步骤，具体代码如下：

```c
PyDoc_STRVAR(PySetData_doc__, "\
测试\n\
\n\
PySetData(str)\n\
str: 出入的字符串\n\
返回: \n\
null \n\
");
static PyObject* PySetData(PyObject *self, PyObject *args)
{
    const char* str = NULL;
    if ( !PyArg_ParseTuple(args, "s", &str) )
    {
        return 0;
    }
    SetData(str);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(PyGetData_doc__, "\
打印数据\n\
\n\
PyGetData()\n\
返回: \n\
data \n\
");
static PyObject* PyGetData(PyObject *self, PyObject *args)
{
    const char* str = NULL;
    return PyString_FromString(GetData());
}

static PyMethodDef module_methods[] = {
    {"py_set_data", PySetData, METH_VARARGS, PySetData_doc__},
    {"py_get_data", PyGetData, METH_VARARGS, PyGetData_doc__},
    {NULL}
    };
void InitCCallPy()
{
    PyObject *module = Py_InitModule3("pycallc", module_methods,
        "python call c");
}
```

`Py_InitModule3`用来定义一个python模块，**第一个参数是模块的名字**，**第二个参数是模块中的方法描述集合**，**第三个参数是模块的描述信息**。上面代码中我们定义了一个叫`pycallc`的模块，方法描述集合module_methods描述了两个方法py_set_data和py_get_data，这两个方法对应的函数地址是PySetData和PyGetData，这两个函数最终会分别调用前面定义的SetData和GetData。这样我们在python脚本中通过pycallc模块的py_set_data和py_get_data方法就可以设置和获取data数据了。看了上面的实现，**其实这个python模块的主要作用就是把c中定义的函数再封装一次，封装的函数能够被python识别。**



## **在python脚本中调用c实现的python模块**

 由于前面已经通过c代码初始化了一个python模块pycallc，那么在脚本中我们就可以通过import导入这个模块，并调用这个模块中的函数。具体代码如下：

```python
# -*- coding: utf-8 -*-

import pycallc

def test():
    print 'in python : ', pycallc.py_get_data()
    pycallc.py_set_data("change hello world!")
```

这样我们就实现了在python脚本中调用c中的方法。



## **总结**

从上面c调用python，python调用c，其实都是一些固定的步骤，知道就会用了，没有会不会的问题，只有想不想知道的问题。没有接触这个技术前可能觉得它很高深，但其实只要稍微花点心思去了解它，它也其实没有这么难。计算机很多技术不外乎都是这样，只有你想不想的问题，没有你会不会的问题，多问，多思考，多学习，总有一天你也能成为技术大牛。