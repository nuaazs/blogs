# Interpret bytes as packed binary data

## 简介

[doc](https://docs.python.org/3/library/struct.html)

此模块在Python值和表示为Python字节对象的C结构之间执行转换。除其他来源外，它可用于处理文件或网络连接中存储的二进制数据。它使用格式字符串作为C结构布局的紧凑描述以及与Python值之间的预期转换。

几个`struct`函数（以及`Struct`的方法）带有一个`buffer`参数。这是指实现缓冲区协议并提供可读或可写缓冲区的对象。用于此目的的最常见类型是字节和字节数组，但是可以看作字节数组的许多其他类型实现了缓冲区协议，因此可以读取/填充它们，而无需从字节对象中进行额外的复制。


## 函数和异常

*exception* `struct.error`

Exception raised on various occasions; argument is a string describing what is wrong.



### `struct.pack(format, v1, v2, ...)`
返回一个字节对象，其中包含根据格式字符串格式打包的值v1，v2，…。参数必须与格式所需的值完全匹配。

### `struct.pack_into(format, buffer, offset, v1, v2, ...)`
Pack the values v1, v2 , ... according to the format string format and write the packed bytes into the writable buffer buffer starting at position offset. Note that *offset* is a required argument.
根据格式字符串格式打包值v1，v2，…，然后将打包的字节从位置偏移处开始写入可写缓冲区。
请注意，`offset`是必需的参数。

### `struct.unpack(format, buffer)`
Unpackfrom the buffer buffer(presumably packed by `pack(format, ...)`) according to the format string format. The result is a tuple even if it contains exactly one item. The buffer's size in bytes must match the size required by the format, as reflected by `calcsize()`.
根据格式字符串格式从缓冲区缓冲区解压缩（大概由pack（format，...）打包）。结果是一个元组，即使它只包含一个项目。缓冲区的大小（以字节为单位）必须与格式要求的大小匹配，如calcsize（）所反映。

### `struct.unpack_from(format, /, buffer, offset=0)
Unpack from buffer starting as position offset, according to the format string format. The result is a tuple even if it contains exactly one item. The buffer's size in bytes, starting at position offset, must be at least the size required by the format, as reflected by `calcsize()`.
根据格式字符串格式，从位置偏移处开始从缓冲区解压缩。结果是一个元组，即使它只包含一个项目。从位置偏移开始的缓冲区大小（以字节为单位）必须至少为格式要求的大小，如`calcsize()`所反映。

### `struct.iter_unpack(format, buffer)`
Iteratively unpackfrom the buffer buffer according to the format string format. This function returns an iterator which will read equally-sized chunks from the buffer until all its contents have been consumed. The buffer's size in bytes must be a multiple of the size required by the format, as reflected by `calcsize()`.
Each iteration yields a tuple as specified by the format string.
根据格式字符串格式迭代地从缓冲区缓冲区中解包。该函数返回一个迭代器，该迭代器将从缓冲区读取大小相等的块，直到其所有内容都被消耗为止。缓冲区的大小（以字节为单位）必须是格式要求的大小的倍数，如`calcsize()`所反映。每次迭代都会产生一个由格式字符串指定的元组。

### `struct.calcsize(format)`
Return the size of the struct(and hence of the bytes object produced by pack(format, ... ) ) corresponding to the format string format.
返回对应于格式字符串格式的结构的大小（以及由此`pack(format，...)`生成的bytes对象的大小）。

## Format Strings
格式字符串是用于在打包和拆包数据时指定预期布局的机制。它们是由格式字符构建的，格式字符指定了要打包/解包的数据类型。此外，还有一些特殊字符可用于控制字节顺序，大小和对齐方式。
### Byte Order, Size, and Alignment
默认情况下，C类型以计算机的本机格式和字节顺序表示，并在必要时通过跳过填充字节来正确对齐（根据C编译器使用的规则）。或者，根据下表，格式字符串的第一个字符可用于指示打包数据的字节顺序，大小和对齐方式：

| Character | Byte order             | Size     | Alignment |
| :-------- | :--------------------- | :------- | :-------- |
| `@`       | native                 | native   | native    |
| `=`       | native                 | standard | none      |
| `<`       | little-endian          | standard | none      |
| `>`       | big-endian             | standard | none      |
| `!`       | network (= big-endian) | standard | none      |

如果第一个字符不是其中一个，则假定为“ `@`”。

注意“ @”和“ =”之间的区别：两者都使用本机字节顺序，但是后者的大小和对齐方式是标准化的。

'`！`'表示网络字节顺序，按照IETF RFC 1700的定义，该字节始终为大端顺序。无法指示非本机字节顺序（强制字节交换）；使用适当的“ <”或“>”选项。

**注意：**

仅在连续的结构成员之间自动添加填充。在编码结构的开头或结尾不添加填充。使用非原生大小和对齐方式时，例如，不添加填充。“ <”，“>”，“ =”和“！”。要将结构的末尾与特定类型的对齐要求对齐，请使用该重复类型为零的该类型的代码结束格式。

## Format Characters

| Format | C Type               | Python type       | Standard size | Notes    |
| :----- | :------------------- | :---------------- | :------------ | :------- |
| `x`    | pad byte             | no value          |               |          |
| `c`    | `char`               | bytes of length 1 | 1             |          |
| `b`    | `signed char`        | integer           | 1             | (1), (2) |
| `B`    | `unsigned char`      | integer           | 1             | (2)      |
| `?`    | `_Bool`              | bool              | 1             | (1)      |
| `h`    | `short`              | integer           | 2             | (2)      |
| `H`    | `unsigned short`     | integer           | 2             | (2)      |
| `i`    | `int`                | integer           | 4             | (2)      |
| `I`    | `unsigned int`       | integer           | 4             | (2)      |
| `l`    | `long`               | integer           | 4             | (2)      |
| `L`    | `unsigned long`      | integer           | 4             | (2)      |
| `q`    | `long long`          | integer           | 8             | (2)      |
| `Q`    | `unsigned long long` | integer           | 8             | (2)      |
| `n`    | `ssize_t`            | integer           |               | (3)      |
| `N`    | `size_t`             | integer           |               | (3)      |
| `e`    | (6)                  | float             | 2             | (4)      |
| `f`    | `float`              | float             | 4             | (4)      |
| `d`    | `double`             | float             | 8             | (4)      |
| `s`    | `char[]`             | bytes             |               |          |
| `p`    | `char[]`             | bytes             |               |          |
| `P`    | `void *`             | integer           |               | (5)      |