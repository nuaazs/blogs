# VTK之基于Qt的VTK应用程序

友好的用户图形界面是应用程序必须的因素之一，对于VTK应用程序也是如此。VTK附带的程序示例大多数是基于控制台的，但是VTK也可以与很多流行的GUI开发工具整合。本文介绍如何把VTK（7.1.0）和GUI开发工具Qt(5.x)进行整合。

## 用CMake管理Qt工程

下图是Qt编译系统：

## 用CMake来管理 Qt工程的CMakeLists脚本程序：

```cmake
cmake_minimum_required( VERSION 2.8 )
project( YourProjectName )
find_package( Qt5Widgets REQUIRED)
include( ${QT_USE_FILE} )

# 程序所有源文件
# 定义变量 Project_SRCS, 其值为所列的文件列表
SET( Project_SRCS
     main.cpp
     )
  
# 程序所有的UI文件
# 定义变量Project_UIS,其值为所列的文件列表
SET( Project_UIS
     YourQtWindows.ui
     )
     
# 通过Qt的uic.exe生成UI文件对应的ui_xxxx.h文件
# 将生成的ui_xxxx.h文件放在变量Project_UIS_H里
# QT5_WRAP_UI就是干这个事情。
QT5_WRAP_UI( Project_UIS_H ${Project_UIS} )

# 通过Qt的moc.exe生成包含Q_OBJECT的头文件对应的
# moc_xxxx.cxx文件，将生成的moc_xxxx.cxx放在
# 变量Porject_MOC_SRCS里。QT5——WRAP_CPP就是干这个事情。
QT5_WRAP_CPP( Project_MOC_SRCS ${Project_MOC_HDRS} )

# Qt的MOC和UIC程序生成的moc_XXXX.cxx和ui_xxx.h
# 等文件是存放在CMake的“Where to build the binaries”
# 里指定的目录里，所以必须这些路径都包含进来
INCLUDE_DIRECTORIES( ${Project_SOURCE_DIR}
                     ${CMAKE_CURRENT_BINARY_DIR}
                   )
                   
# Qt程序如果有资源文件(*.qrc)，要包含资源文件，
# 然后用Qt的rcc.exe生成相应的qrc_XXXX.cpp文件。
# QT5_ADD_RESOURCES就是干这个事情。
SET( Project_RCCS YourProject.qrc)
QT5_ADD_RESOURCES( Project_RCC_SRCS ${Project_RCCS})

# 根据程序的cpp文件、头文件以及中间生成的ui_XXXX.h、
# moc_XXXX.cxx、qrc_XXXX.cxx等生成可执行文件，并链接
# Qt的动态库(Qt的动态库都定义在QT_LIBRARIES变量里了)
ADD_EXECUTABLE( YourProjectName
                ${Project_SRCS}
                ${Project_UIS_H}
                ${Project_MOC_SRCS}
                ${Project_RCC_SRCS}                           
              )
TARGET_LINK_LIBRARIES ( YourProjectName  ${Qt5Widgets_LIBRARIES} )
```



## 在Qt Designer里继承QVTKWidget控件

要实现QVTKWidget在Qt Designer里像Qt的其他标准控件的拖拽功能，需要将VTK中编译（必须是Release）生成的QVTKWidgetPlugin.dll和QVTKWidgetPlugin.lib复制到Qt安装目录中的plugins\designer下。复制完成后，Qt Designer界面如下图所示。

## 示例演示

本例先用Qt Designer生成ui文件，再整合VTK生成CT数据浏览器。