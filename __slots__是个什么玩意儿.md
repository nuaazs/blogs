

|python|魔法函数|

### \__slots__

在python中，每个类都有实例属性。默认情况下Python用一个字典来保存一个对象的实例属性。
然而，对于有着已知属性的小类来说，它可能是个瓶颈。这个字典浪费了很多内存。Python不能在对象创建时直接分配一个固定量的内存来保存所有的属性。因此如果你创建许多对象（我指的是成千上万个），它会消耗掉很多内存。
不过还是有一个方法来规避这个问题。这个方法需要使用`__slots__`来告诉Python不要使用字典，而且只给一个固定集合的属性分配空间。

#### 不使用
```python
class MyClass():
    def __init__(self,name, indentifier):
        self.name = name
        self.identifier = indentifier
        self.set_up()
```
#### 使用
```python
class MyClass():
    __slots__ = ['name','identifier']
    def __init__(self, name, identifier):
        self.name = name
        self.identifier = indentifier
        self.set_up()
```
第二段代码会为你的内存减轻负担。通过这个技巧，内存占用率几乎40%~50%的减少。

