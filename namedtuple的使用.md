##  基本概念

1. `namedtuple`是一个工厂函数，定义在python标准库中的`collections`中，使用这个函数可以创建一个可读性更强的元组。
2. `namedtuple`函数所创建（返回）的是一个**元组的子类**（python中基本数据类型都是类，且可以在`buildins`模块中找到）
3. `namedtuple`函数所创建元组，中文名称为 **具名元组**
4. 在使用普通元组的时候，我们只能通过`index`来访问元组中的某个数据
5. 使用具名元组，我们既可以使用`index`来访问，也可以使用具名元组中每个字段的名称来访问
6. 值得注意的是，具名元组和普通元组所需要的内存空间相同，所以 **不必使用性能来权衡是否使用具名元组**。



## 使用

`namedtuple`是一个函数

参数解析：

```python
def namedtuple(typename, field_names,*,rename=False,defaults=None,module=None)
```

有两个必填参数`typename`和`field_names`

- typename
  - 参数类型为字符串。
  - 具名元组返回一个元组子对象，我们要为这个对象命名，传入`typename`即可。
- field_names
  - 参数类型为字符串序列。
  - 用于为创建的元组的每个元素命名，可以传入像`['a','b']`这样的序列，也可以传入`'a b'`或`'a, b'`这种被**逗号**或**空格**分割的单字符串。
  - 必须是合法的标识符。不能是关键字。
- rename
  - 注意的参数中使用了`*`，其后的所有参数必须指定关键字。
  - 参数为布尔值。
  - 默认为`False`。当我们指定为`True`时，如果定义**`field_names`**参数时，出现非法参数时，会将其替换为位置名称。如`['abc', 'def', 'ghi', 'abc']`会被替换为`['abc', '_1', 'ghi', '_3']`
- defaults
  - 参数为`None`或可迭代对象。
  - 当此参数为`None`时，创建具名元组的实例时，必须要根据`field_names`传递指定数量的参数
  - 当设置`defaults`时，我们就为具名元组的元素赋予了默认值，被赋予默认值的元素在实例化的时候可以不传入
  - 当`defaults`传入的序列长度和`field_names`不一致时，函数默认会右侧优先
  - 如果`field_names`是`['x', 'y', 'z']`，`defaults`是`(1, 2)`，那么`x`是实例化必填参数，`y`默认为`1`，`z`默认为`2`



### 基本使用

理解了`namedtuple`函数的参数，我们就可以创建具名元组了

```python
Point = namedtuple("Point",["x","y"]) # 返回一个名为`Point`的类，并赋值给名为`Point`的变量
p = Point(11, y=22)
p[0]+p[1]  # 33

x,y = p
x,y # (11,22)

p.x+p.y #33
p # Point(x=11,y=22)
```

具名元组在存储[`csv`](https://docs.python.org/3.8/library/csv.html#module-csv)或者[`sqlite3`](https://docs.python.org/3.8/library/sqlite3.html#module-sqlite3)返回数据的时候特别有用。

```python
EmployeeRecord = namedtuple('EmployeeRecord','name,age,title,department,paygrade')

import csv
import sqlite3
for emp in map(EmployeeRecord._make, csv.reader(open("employees.csv","rb"))):
    print(emp.name, emp.title)

conn = squlite3.connect('/companydata')
cursor = conn.cursor()
cursor.execute('SELECT name,age,title,department,paygrad FROM employees')
for emp in map(EmployeeRecord._make,cursor.fetchall()):
    print(emp.name, emp.title)
```

### 特性

具名元组除了拥有继承自基本原子的所有方法之外，还额外提供三个方法和属性，为了防止命名冲突，这些方法都会以下划线开头

#### `_make(iterable)`

这是一个类函数，参数是一个迭代器，可以使用这个函数来构建具名元组实例

```python
t = [11,22]
Point._make(t)
# Point(x=11,y=22)
```

#### `_asdict()`

根据具名元组的名称和其元素值，构建一个`OrderedDict`返回

```python
p = Point(x=11,y=22)
p._asdict()
# OrderedDict([('x',11),('y',22)])
```

#### `_replace(**kwargs)`

实例方法，根据传入的关键词参数，替换具名元组的相关参数，然后返回一个新的具名元组。

```python
p=Point(x=11,y=22)
p._replace(x=33)
# Point(x=33,y=22)

for partnum,record in inventory.items():
    inventory[partnum]=record._replace(price=newprices[partnum],timestamp=time.now())
```

#### `_fields`

这是一个实例属性，储存了此具名元组的元素名称元组，在根据已经存在的具名元组创建新的具名元组的时候使用

```python
p.fields # view the field names
# ('x','y')

Color = namedtuple('Color','red green blue')
Pixel = namedtuple('Pixel',Point._fields+Color._fields)
Pixel(11,22,128,255,0)
# Pixel(x=11, y=22, red=128, green=255, blue=0)
```

#### `_fields_defaults`

查看具名元组类的默认值

```python
Account = namedtuple('Account',['type','balance'],defaults=[0])
Account._fields_defaults #{'balance': 0}  Attention, 0 is for 'balance' !!
Account('premium')
#Account(type='premium', balance=0)
```

### 使用技巧

#### 1. 使用`getattr`获取具名元组元素值

```python
getattr(p,'x')
# 11
```

#### 2. 将字典转换为具名元组

```python
d = {'x':11,'y':22}
Point(**d)
# Point(x=11,y=22)
```

#### 3. 既然具名元组是一个类，那么可以定制

```python
class Point(namedtuple('Point',['x','y'])):
    __slots__=()
    @property
    def hypot(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
    def __str__(self):
        return f"Point: x={self.x:6.3f} y={self.y:6.3f} hypot={self.hypot:6.3f}"
for p in Point(3,4), Point(14, 5/7):
    print(p)
```

`__slots__`值的设置可以保证具名元组保持最小的内存占用。



### 实例：namedtuple纸牌
```python
import collections
Card = collections.namedtuple('Card','rank suit')
class FrenchDeck:
  # 等级2 ~ A
  self.ranks = [str(n) for n in range(2,11)] + list('JQKA')
  # Suits of cards
  self.suits = 'spades diamonds clubs hearts'.split(' ')
  # cards
  def __init__(self):
    self._cards = [Card(rank,suit) for suit in self.suits for rank in self.ranks]
  def __getitem__(self,position):
    return self._cards[position]

french_deck = FrenchDeck()
french_deck[0] # Card(rank='2',suits='spades')
```