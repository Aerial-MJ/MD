# C语言知识

## C语言运算符

C语言中有多种类型的运算符，它们用于执行各种操作，包括算术运算、逻辑运算、位运算等。以下是C语言中常用的运算符：

1. **算术运算符**：
   - `+`：加法
   - `-`：减法
   - `*`：乘法
   - `/`：除法
   - `%`：取模（取余）

2. **关系运算符**：
   - `==`：等于
   - `!=`：不等于
   - `>`：大于
   - `<`：小于
   - `>=`：大于等于
   - `<=`：小于等于

3. **逻辑运算符**：
   - `&&`：逻辑与
   - `||`：逻辑或
   - `!`：逻辑非

4. **位运算符**：
   - `&`：按位与
   - `|`：按位或
   - `^`：按位异或
   - `~`：按位取反
   - `<<`：左移位
   - `>>`：右移位

5. **赋值运算符**：
   - `=`：简单赋值
   - `+=`：加法赋值
   - `-=`：减法赋值
   - `*=`：乘法赋值
   - `/=`：除法赋值
   - `%=`：取模赋值
   - `&=`：按位与赋值
   - `|=`：按位或赋值
   - `^=`：按位异或赋值
   - `<<=`：左移位赋值
   - `>>=`：右移位赋值

6. **其他运算符**：
   - `sizeof()`：返回变量或数据类型的大小（字节数）
   - `&`：取地址运算符，返回变量的地址
   - `*`：指针运算符，指向指针变量所指向的地址的值
   - `?:`：条件运算符，也称为三元运算符

以上是C语言中常用的运算符，它们能够执行各种不同类型的操作，是C语言中非常重要的一部分。

## 输入输出

当我们提到**输入**时，这意味着要向程序填充一些数据。输入可以是以文件的形式或从命令行中进行。C 语言提供了一系列内置的函数来读取给定的输入，并根据需要填充到程序中。

当我们提到**输出**时，这意味着要在屏幕上、打印机上或任意文件中显示一些数据。C 语言提供了一系列内置的函数来输出数据到计算机屏幕上和保存数据到文本文件或二进制文件中。

### getchar() & putchar() 函数

**int getchar(void)** 函数从屏幕读取下一个可用的字符，并把它返回为一个整数。这个函数在同一个时间内只会读取一个单一的字符。您可以在循环内使用这个方法，以便从屏幕上读取多个字符。

**int putchar(int c)** 函数把字符输出到屏幕上，并返回相同的字符。这个函数在同一个时间内只会输出一个单一的字符。您可以在循环内使用这个方法，以便在屏幕上输出多个字符。

请看下面的实例：

```c
#include <stdio.h>
 
int main( )
{
   int c;
 
   printf( "Enter a value :");
   c = getchar( );
 
   printf( "\nYou entered: ");
   putchar( c );
   printf( "\n");
   return 0;
}
```

### gets() & puts() 函数

**char \*gets(char \*s)** 函数从 **stdin** 读取一行到 **s** 所指向的缓冲区，直到一个终止符或 EOF。

**int puts(const char \*s)** 函数把字符串 s 和一个尾随的换行符写入到 **stdout**。

```c
#include <stdio.h>
 
int main( )
{
   char str[100];
 
   printf( "Enter a value :");
   gets( str );
 
   printf( "\nYou entered: ");
   puts( str );
   return 0;
}
```

### scanf() 和 printf() 函数

**int scanf(const char \*format, ...)** 函数从标准输入流 **stdin** 读取输入，并根据提供的 **format** 来浏览输入。

**int printf(const char \*format, ...)** 函数把输出写入到标准输出流 **stdout** ，并根据提供的格式产生输出。

**format** 可以是一个简单的常量字符串，但是您可以分别指定 %s、%d、%c、%f 等来输出或读取字符串、整数、字符或浮点数。还有许多其他可用的格式选项，可以根据需要使用。如需了解完整的细节，可以查看这些函数的参考手册。现在让我们通过下面这个简单的实例来加深理解：

```c
#include <stdio.h>
int main( ) {
 
   char str[100];
   int i;
 
   printf( "Enter a value :");
   scanf("%s %d", str, &i);
 
   printf( "\nYou entered: %s %d ", str, i);
   printf("\n");
   return 0;

```

## 格式化输出

如果你想在格式化输出中为 `%d`、`%u`、`%f`、`%s` 这些格式说明符指定一定的位数，你可以在格式说明符中使用修饰符来控制输出的宽度和精度。下面是一些常用的修饰符：

1. **宽度修饰符：** 宽度修饰符可以控制输出的最小宽度。你可以在 `%d`、`%u`、`%f`、`%s` 等格式说明符之前加上数字，表示输出的最小宽度。例如，`%5d` 表示输出的最小宽度为5个字符。

2. **精度修饰符：** 精度修饰符通常用于浮点数 `%f` 格式说明符中，用于控制小数点后的位数。你可以在 `%f` 之前加上 `.N`，其中 `N` 是希望保留的小数位数。例如，`%.2f` 表示保留两位小数。

下面是一些示例：

```c
int num = 123;
printf("%05d\n", num);  // 输出宽度为5，不足的位置补零，结果为"00123"

unsigned int unsigned_num = 456;
printf("%010u\n", unsigned_num);  // 输出宽度为10，不足的位置补零，结果为"0000000456"

float float_num = 3.1415926;
printf("%.2f\n", float_num);  // 输出保留两位小数，结果为"3.14"

char str[] = "Hello";
printf("%010s\n", str);  // 输出宽度为10，右对齐，不足的位置补空格，结果为"     Hello"
```

在这些示例中，修饰符控制了输出的格式，使输出更加符合需求。

## 文件操作

一个文件，无论它是文本文件还是二进制文件，都是代表了一系列的字节。C 语言不仅提供了访问顶层的函数，也提供了底层（OS）调用来处理存储设备上的文件。

### 打开文件

您可以使用 **fopen( )** 函数来创建一个新的文件或者打开一个已有的文件，这个调用会初始化类型 **FILE** 的一个对象，类型 **FILE** 包含了所有用来控制流的必要的信息。下面是这个函数调用的原型：

```c
FILE *fopen( const char *filename, const char *mode );
```

在这里，**filename** 是字符串，用来命名文件，访问模式 **mode** 的值可以是下列值中的一个：
| mode    | 具体作用                       |
| ---- | :----------------------------------------------------------- |
| r    | 打开一个已有的文本文件，允许读取文件。                       |
| w    | 打开一个文本文件，允许写入文件。如果文件不存在，则会创建一个新文件。在这里，您的程序会从文件的开头写入内容。如果文件存在，则该会被截断为零长度，重新写入。 |
| a    | 打开一个文本文件，以追加模式写入文件。如果文件不存在，则会创建一个新文件。在这里，您的程序会在已有的文件内容中追加内容。 |
| r+   | 打开一个文本文件，允许读写文件。                             |
| w+   | 打开一个文本文件，允许读写文件。如果文件已存在，则文件会被截断为零长度，如果文件不存在，则会创建一个新文件。 |
| a+   | 打开一个文本文件，允许读写文件。如果文件不存在，则会创建一个新文件。读取会从文件的开头开始，写入则只能是追加模式。 |

如果处理的是二进制文件，则需使用下面的访问模式来取代上面的访问模式：

```
"rb", "wb", "ab", "rb+", "r+b", "wb+", "w+b", "ab+", "a+b"
```

### 关闭文件

为了关闭文件，请使用 fclose( ) 函数。函数的原型如下：

```c
 int fclose( FILE *fp );
```

如果成功关闭文件，**fclose( )** 函数返回零，如果关闭文件时发生错误，函数返回 **EOF**。这个函数实际上，会清空缓冲区中的数据，关闭文件，并释放用于该文件的所有内存。EOF 是一个定义在头文件 **stdio.h** 中的常量。

C 标准库提供了各种函数来按字符或者以固定长度字符串的形式读写文件。

### 写入文件

下面是把字符写入到流中的最简单的函数：

```
int fputc( int c, FILE *fp );
```

函数 **fputc()** 把参数 c 的字符值写入到 fp 所指向的输出流中。如果写入成功，它会返回写入的字符，如果发生错误，则会返回 **EOF**。您可以使用下面的函数来把一个以 null 结尾的字符串写入到流中：

```
int fputs( const char *s, FILE *fp );
```

函数 **fputs()** 把字符串 **s** 写入到 fp 所指向的输出流中。如果写入成功，它会返回一个非负值，如果发生错误，则会返回 **EOF**。您也可以使用 **int fprintf(FILE \*fp,const char \*format, ...)** 函数把一个字符串写入到文件中。尝试下面的实例：

```c
#include <stdio.h>
 
int main()
{
   FILE *fp = NULL;
 
   fp = fopen("/tmp/test.txt", "w+");
   fprintf(fp, "This is testing for fprintf...\n");
   fputs("This is testing for fputs...\n", fp);
   fclose(fp);
}
```

当上面的代码被编译和执行时，它会在 /tmp 目录中创建一个新的文件 **test.txt**，并使用两个不同的函数写入两行。接下来让我们来读取这个文件。

### 读取文件

下面是从文件读取单个字符的最简单的函数：

```
int fgetc( FILE * fp );
```

**fgetc()** 函数从 fp 所指向的输入文件中读取一个字符。返回值是读取的字符，如果发生错误则返回 **EOF**。下面的函数允许您从流中读取一个字符串：

```
char *fgets( char *buf, int n, FILE *fp );
```

函数 **fgets()** 从 fp 所指向的输入流中读取 n - 1 个字符。它会把读取的字符串复制到缓冲区 **buf**，并在最后追加一个 **null** 字符来终止字符串。

如果这个函数在读取最后一个字符之前就遇到一个换行符 '\n' 或文件的末尾 EOF，则只会返回读取到的字符，包括换行符。您也可以使用 **int fscanf(FILE \*fp, const char \*format, ...)** 函数来从文件中读取字符串，但是在遇到第一个空格和换行符时，它会停止读取。

```c
#include <stdio.h>
 
int main()
{
   FILE *fp = NULL;
   char buff[255];
 
   fp = fopen("/tmp/test.txt", "r");
   fscanf(fp, "%s", buff);
   printf("1: %s\n", buff );
 
   fgets(buff, 255, (FILE*)fp);
   printf("2: %s\n", buff );
   
   fgets(buff, 255, (FILE*)fp);
   printf("3: %s\n", buff );
   fclose(fp);
 
}
```

### 文件的定位

`ftell()`, `fseek()`, 和 `rewind()` 是C语言标准库 `<stdio.h>` 中提供的文件定位和定位指针管理函数。

1. **`ftell()`**：
   - `ftell()` 函数用于获取文件指针当前位置相对于文件起始位置的偏移量（以字节为单位）。
   - 它的函数原型是：`long ftell(FILE *stream);`
   - `stream` 是指向文件的指针。
   - `ftell()` 返回值是当前文件指针的位置，如果发生错误则返回 -1。

2. **`fseek()`**：
   - `fseek()` 函数用于设置文件指针的位置。
   - 它的函数原型是：`int fseek(FILE *stream, long offset, int origin);`
   - `stream` 是指向文件的指针。
   - `offset` 是要设置的偏移量，可以为正（向后移动）或负（向前移动）。
   - `origin` 指定起始位置，可以是 `SEEK_SET`（文件起始位置）、`SEEK_CUR`（当前位置）、或 `SEEK_END`（文件末尾）。
   - `fseek()` 函数在成功时返回0，失败时返回非0值。

3. **`rewind()`**：
   - `rewind()` 函数用于将文件指针重新定位到文件的开头。
   - 它的函数原型是：`void rewind(FILE *stream);`
   - `stream` 是指向文件的指针。
   - `rewind()` 函数没有返回值。

这些函数在处理文件时非常有用，可以用于查找文件中的特定位置，跳转到文件的不同部分，以及重新定位文件指针到文件的开头等操作。

下面是一个简单的例子，演示了如何使用 `ftell()`、`fseek()` 和 `rewind()` 函数来操作文件指针：

```c
#include <stdio.h>

int main() {
    FILE *fp;
    char ch;

    // 打开文件
    fp = fopen("example.txt", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return -1;
    }

    // 使用 ftell() 获取文件指针当前位置
    long position = ftell(fp);
    printf("Current position: %ld\n", position);

    // 移动文件指针到文件末尾
    fseek(fp, 0, SEEK_END);

    // 使用 ftell() 再次获取文件指针当前位置
    position = ftell(fp);
    printf("End position: %ld\n", position);

    // 将文件指针重新定位到文件开头
    rewind(fp);

    // 使用 ftell() 再次获取文件指针当前位置
    position = ftell(fp);
    printf("Position after rewind: %ld\n", position);

    // 关闭文件
    fclose(fp);

    return 0;
}
```

在这个例子中，我们首先打开一个文件（假设文件名为 `example.txt`），然后使用 `ftell()` 获取文件指针当前位置，并打印出来。接着，我们使用 `fseek()` 将文件指针移动到文件末尾，并再次使用 `ftell()` 获取文件指针当前位置，然后打印出来。最后，我们使用 `rewind()` 将文件指针重新定位到文件开头，并再次使用 `ftell()` 获取文件指针当前位置，并打印出来。

这个例子演示了如何使用这三个函数来管理文件指针的位置，以及如何获取和改变文件指针的位置。

## 内存四区

在C语言中，内存被划分为以下四个主要区域：

1. **栈（Stack）**：
   - 栈是一种线性的数据结构，它用于存储函数的局部变量、函数参数和函数调用的返回地址。
   - 栈的分配和释放由编译器自动完成，通常是在函数调用和返回时进行管理。
   - 栈是一种后进先出（LIFO）的数据结构。

2. **堆（Heap）**：
   - 堆是用于动态内存分配的区域，程序员可以在运行时从堆中分配内存，也可以在不需要时释放已分配的内存。
   - 堆上的内存分配和释放是由程序员控制的，而不是由编译器。
   - 堆上的内存分配通常使用函数如`malloc()`、`calloc()`、`realloc()`来完成，释放则使用`free()`函数。

3. **全局/静态存储区（Global/Static Storage Area）**：
   - 全局存储区包含了全局变量、静态变量以及常量。
   - 全局变量和静态变量在程序的整个执行期间都存在，它们在内存中的位置是固定的。
   - 全局存储区也包含 了字符串常量，这些字符串常量通常存储在只读内存区域。

4. **代码区（Text/Code Area）**：
   - 文字常量区包含了程序的代码段，以及常量字符串。
   - 文字常量区通常是只读的，存储着程序的指令和常量字符串，这些内容在程序执行期间不会被修改。

这些内存区域在程序运行时起着不同的作用，了解它们有助于编写更有效、更安全的程序，并且有助于理解动态内存管理和变量的作用域。

C++程序在执行时，将内存大方向划分为4个区域
代码区:存放函数体的二进制代码，由操作系统进行管理的
全局区:存放全局变是和静态变量以及常量
栈区:由编译器自动分配释放，存放函数的参数值.局部变量等
堆区:由程序员分配和释放，若程序员不释放，程序结束时由操作系统回收

**程序运行前，在程序编译后，生成了exe可执行程序，未执行该程序前分为俩个区域：**

**代码区:**

存放 CPU 执行的机器指令
代码区是共享的，共享的目的是对于频繁被执行的程序，只需要在内存中有一份代码即可
代码区是只读的，使其只读的原因是防止程序意外地修改了它的指令

**全局区：**

全局变量和静态变量存放在此
全局区还包含了常量区,字符串常量和其他常量也存放在此
该区域的数据在程序结束后由操作系统释放

```c
int a1 = 10;//全局变量
int main(void)
{
	
	//全局区 
	//全局变量，静态变量，常量
	//全局变量:不在函数体中的变量
	//静态变量 普通变量前加static 
	static int a;
	printf("%d\n",&a);
	//常量：
	//字符串常量   "helloworld"
	printf("%d\n",&"helloworld");

	//const修饰的变量
	//const 修饰的全局变量 ->全局区 
	//const 修饰的局部变量 ->栈区 
    
    return 0;
} 
```

**程序运行后**

**栈区 ：**

由编译器自动分配释放,存放函数的参数值,局部变量等
注意事项: 不要返回局部变量的地址，栈区开辟的数据由编译器自动释放

**堆区：**

由程序员分配释放,若程序员不释放,程序结束时由操作系统回收
在C中主要利用`malloc`在堆区开辟内存

## const 和 static

### static

**1. static修饰变量**

**全局变量**
全局变量定义在函数体外部，在全局数据区分配存储空间，且编译器会自动对其初始化，当我们没有赋值是编译器自动优化为0

对于普通全局变量对整个工程可见，其他文件可以使用extern外部声明后直接使用。也就是说其他文件不能再定义一个与其相同名字的变量了（否则编译器会认为它们是同一个变量）。

```c
//file1.c
#include <stdio.h>
extern int a;
int main()
{
	printf("a = %d\n",a); //a = 10
	return 0;
}

//file2.c
#include <stdio.h>
int a = 10;
```

对于 `static` 修饰的全局变量，仅对当前文件可见，其他文件不可访问，其他文件可以定义与其同名的变量，两者互不影响

```c
//file1.c
#include <stdio.h>
static int a;
int main()
{
	printf("a = %d\n",a); //a = 0
	return 0;
}

//file2.c
#include <stdio.h>
int a = 10; //这是完全可以的
int fun();
```

**局部变量**
普通局部变量是再熟悉不过的变量了，在任何一个函数内部定义的变量（不加static修饰符）都属于这个范畴,编译器也不会为其初始化

普通的局部变量存储在栈空间类，使用完毕后会立即被释放

静态局部变量使用`static`修饰符定义，即使在声明时未赋初值，编译器也会把它初始化为0。且静态局部变量存储于进程的全局数据区，即使函数返回，它的值也会保持不变。

```c
#include <stdio.h>

int fun1()
{
    int a = 1;
    printf("a = %d\n",a);
    a++;
    printf("a++ = %d\n",a);
}

int fun2()
{
    static int a = 1;
    printf("static a = %d\n",a);
    a++;
    printf("a++ = %d\n",a);
}

int main()
{
    fun1();
    printf("========\n");
    fun1();
    printf("********\n");
    fun2();
    printf("========\n");
    fun2();
   
    return 0;
}

/*
a = 1
a++ = 2
========
a = 1
a++ = 2
********
static a = 1
a++ = 2
========
static a = 2
a++ = 3
*/
```

**2. static修饰函数**

函数的使用方式与全局变量类似，在函数的返回类型前加上static，就是静态函数。其特性如下：

- 静态函数只能在声明它的文件中可见，其他文件不能引用该函数
- 不同的文件可以使用相同名字的静态函数，互不影响

非静态函数可以在另一个文件中直接引用，甚至不必使用extern声明

```c
//file1.c 
#include <stdio.h>

static void fun(void)
{
    printf("hello from fun\n");
}

int main(void)
{
    fun(); //只能在本文件中调用
    fun1(); //不同文件的普通函数可以直接调用
    //fun2(); //这里就不能调用另一个文件static修饰的fun2();
    return 0;
}

//file2.c
#include <stdio.h>

void fun1(void)
{
    printf("hello from static fun1\n");
}
static void fun2(void)
{
    printf("hello from static fun1\n");
}

```

### const

#### 1.const 修饰普通变量

当一个变量用 const 修饰后就不允许改变它的值了
比如

```c
const int a = 10;
//a = 5; 错误
const int arr[5]={1,2,3,4,5};
//arr[0] = 0; 错误
```

#### 2.const修饰指针的三种效果

```c
int a;
int *p = &a;
```

1.const int*p=&a；
const 和 int的位置是可以互换的，修饰的是 *p ，那么 *p 就不可变。\*p 表示的是指针变量 p 所指向的内存单元里面的内容，此时这个内容不可变

```c
void fun()
{
	int a =10;
	const int *p = &a;
	//*p = 20; 就不能使用 
}
```

2.int*const p=&a；
const距离 p 比较近，此时 const 修饰的是 p，所以 p 中存放的内存单元的地址不可变，而内存单元中的内容可变。即 p 的指向不可变，p 所指向的内存单元的内容可变。

```c
void fun()
{
	int a =10;
	int b = 5;
	int* const p = &a;
	*p = 20;
	//p = &b; 这就不能使用了  
}
```

3.const int*const p=&a；
此时 *p 和 p 都被修饰了，那么 p 中存放的内存单元的地址和内存单元中的内容都不可变。

```c
void fun()
{
	int a =10;
	int b = 5;
	const int * const p = &a;
	//*p = 20; 这也不行
	//p = &b; 这就不能使用了  
}
```

## 内存泄露

使用**malloc()、calloc()、realloc()**动态分配的内存，如果没有指针指向他，就无法进行任何操作，这段内存会一直被程序占用，知道程序运行结束由操作系统回收。

```c
#include <stdio.h>
#include <stdlib.h>
int main(){
    char *p = (char*)malloc(100 * sizeof(char));
    p = (char*)malloc(50 * sizeof(char));
    free(p);
    p = NULL;
    return 0;
}
```

该程序中，第一次分配100字节的内存，并将p指向他；第二次分配50字节的内存，依然使用p指向他。 这就导致了一个问题，第一次分配的100字节的内存没有指针指向他了，而且我们也不知道这块内存的地址，所以就再也无法找回了，也没法释放了。这块内存就成了垃圾内存，虽然毫无用处，但依然占用资源，唯一的办法就是等程序运行结束后由操作系统回收。

这就是内存泄露`（Memory Leak）`，可以理解为程序和内存失去了联系，再也无法对他进行任何的操作。

内存泄露形象的比喻是“操作系统可提供给所有程序使用的内存空间正在被某个程序榨干”，最终结果是程序运行时间越长，占用内存空间越来越多，最终用尽全部内存空间，整个系统崩溃。 

再来看一种内存泄露情况：

```c
int *pOld = (int*) malloc( sizeof(int) );
int *pNew = (int*) malloc( sizeof(int) );
```

这两句代码分别创建了一块内存，并且将内存的地址传给了指针pOld和pNew。此时指针pOld和pNew分别指向这两块内存。 

如果接下来进行这样的操作：`pOld = pNew;`

`pOld`指针就指向了`pNew`指向的内存地址，这时候再进行释放内存的操作：` free(pOld) `

此时释放的`pOld`所指向的内存空间就是原来`pNew`指向的，于是这块空间被释放掉了。但是`pOld`原来指向的那块内存空间还没有被释放，不过因为没有指针指向这块内存，所以这块内存就造成了丢失。 

另外，你不应该进行类似下面这样的操作： 

```c
malloc( 100 * sizeof(int) ); 
```

这样的操作没有意义，因为没有指针指向分配的内存，无法使用，而且无法通过`free()`释放掉，造成了内存泄露。

## 枚举类型

C语言中的枚举类型（Enum）是一种用户自定义的数据类型，它允许程序员定义一个包含一组命名常量的新类型。枚举类型的常量被称为枚举常量。

枚举类型的定义形式如下：

```c
enum enum_name {
    enumeration_constants
};
```

其中，`enum_name` 是枚举类型的名称，`enumeration_constants` 是枚举常量的列表，用逗号分隔。

以下是一个简单的枚举类型的例子：

```c
#include <stdio.h>

enum Weekday {
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
    SATURDAY,
    SUNDAY
};

int main() {
    enum Weekday today;
    today = WEDNESDAY;
    
    printf("Today is %d\n", today); // 输出 Today is 2

    return 0;
}
```

在上面的例子中，我们定义了一个枚举类型 `Weekday`，它包含了一周中的每一天。默认情况下，第一个枚举常量的值是0，后续的枚举常量的值依次递增。在 `main` 函数中，我们声明了一个 `Weekday` 类型的变量 `today`，并将它赋值为 `WEDNESDAY`。

枚举类型的具体操作包括：

1. 定义枚举类型：使用 `enum` 关键字定义枚举类型，指定枚举类型的名称和枚举常量。

2. 声明枚举变量：声明一个枚举类型的变量，以便在程序中使用。

3. 访问枚举常量：可以使用枚举常量的名称来访问它们。

4. 枚举变量的赋值：将枚举常量赋给枚举变量，或者将枚举变量赋给另一个枚举变量。

5. 比较枚举变量：可以对枚举变量进行相等性比较和大小比较。

6. 使用枚举常量作为 `switch` 语句的 case 标签：枚举常量在 `switch` 语句中很常用，因为它们提供了一种可读性强的方式来处理多种情况。

枚举类型在编程中通常用于提高代码的可读性和可维护性，因为它允许开发者使用有意义的符号来表示不同的状态或选项，而不是使用数字或字符串。

在C语言中，默认情况下，枚举类型的第一个枚举常量的值是0，后续的枚举常量的值依次递增。因此，如果你没有显式地给枚举常量指定值，第一个枚举常量的值就是0，第二个枚举常量的值是1，以此类推。

例如，对于以下的枚举类型：

```c
enum ExampleEnum {
    FIRST,
    SECOND,
    THIRD
};
```

`FIRST` 的值是0，`SECOND` 的值是1，`THIRD` 的值是2。

你也可以显式地为枚举常量指定值，例如：

```c
enum ExampleEnum {
    FIRST = 1,
    SECOND = 5,
    THIRD = 10
};
```

在这个例子中，`FIRST` 的值是1，`SECOND` 的值是5，`THIRD` 的值是10。

总之，默认情况下，如果你不为枚举常量指定值，它们将从0开始递增。



## 结构体和联合

在C语言中，结构体（struct）和联合（union）是用于组织和存储相关数据的复合数据类型。它们允许你在一个数据类型中存储多个不同类型的成员。

### 结构体（struct）

结构体是一种用户自定义的数据类型，它允许将不同类型的数据组合在一起以形成一个单一的数据单元。结构体的成员可以包括基本数据类型、数组、指针、甚至其他结构体。

结构体的定义形式如下：

```c
struct struct_name {
    member1_type member1_name;
    member2_type member2_name;
    // ...
};
```

以下是一个简单的结构体示例：

```c
#include <stdio.h>

struct Person {
    char name[50];
    int age;
    float salary;
};

int main() {
    struct Person person1;

    strcpy(person1.name, "John");
    person1.age = 30;
    person1.salary = 50000.0;

    printf("Name: %s\n", person1.name);
    printf("Age: %d\n", person1.age);
    printf("Salary: %.2f\n", person1.salary);

    return 0;
}
```

### 联合（union）

联合是一种特殊的数据类型，它允许在同一内存位置存储不同的数据类型。与结构体不同，联合的所有成员共享相同的内存空间。因此，联合的大小等于其最大成员的大小。

联合的定义形式如下：

```c
union union_name {
    member1_type member1_name;
    member2_type member2_name;
    // ...
};
```

以下是一个简单的联合示例：

```c
#include <stdio.h>

union Data {
    int i;
    float f;
    char str[20];
};

int main() {
    union Data data;

    data.i = 10;
    printf("data.i : %d\n", data.i);

    data.f = 220.5;
    printf("data.f : %f\n", data.f);

    strcpy(data.str, "C Programming");
    printf("data.str : %s\n", data.str);

    printf("Size of data : %lu\n", sizeof(data));

    return 0;
}
```

注意，在这个例子中，修改了一个成员会影响到其他成员。这是因为所有成员共享同一块内存空间。因此，在使用联合时，要小心确保不会意外地修改了其他成员。

### 结构体（struct）能定义函数吗

在C语言中，结构体本身不能定义函数，但是你可以在结构体中定义指向函数的指针。这样做的目的是将函数与数据关联起来，使得函数可以操作结构体的数据成员。

以下是一个简单的示例，展示了如何在结构体中定义函数指针：

```c
#include <stdio.h>

// 定义结构体
struct Rectangle {
    int length;
    int width;
    // 定义指向函数的指针
    void (*area)(struct Rectangle*);
};

// 计算矩形面积的函数
void calculate_area(struct Rectangle* r) {
    printf("Area of rectangle: %d\n", r->length * r->width);
}

int main() {
    // 创建结构体对象
    struct Rectangle rect;
    rect.length = 5;
    rect.width = 3;
    // 将函数指针指向计算面积的函数
    rect.area = calculate_area;
    // 调用通过函数指针间接调用计算面积的函数
    rect.area(&rect);

    return 0;
}
```

在这个示例中，结构体 `Rectangle` 中包含了一个函数指针 `area`，指向了一个接受 `Rectangle` 结构体指针作为参数并计算面积的函数 `calculate_area`。通过设置函数指针，我们可以通过结构体间接调用函数。

需要注意的是，在结构体中定义函数指针时，函数的签名必须与指针所指向的函数匹配。在实际应用中，这种方法可以用来实现一些面向对象的特性，例如C语言中的函数指针表（Function Pointer Table）或虚函数（Virtual Functions）的概念。

## 字符串操作

字符串操作是计算机C语言中非常重要的一部分，以下是它们的详细解释和示例代码：

**1、字符串定义和赋值**

在C语言中，字符串是一个字符数组，以`'\0'`作为结束符。可以使用双引号或大括号来定义和赋值字符串变量。

示例代码：

```c
char str1[] = "Hello, world!"; // 使用双引号
char str2[] = {'H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\0'}; // 使用大括号
```

**2、字符串输入和输出**

可以使用`scanf`函数和`printf`函数来输入和输出字符串。

示例代码：

```c
char str[50];
printf("Enter a string: ");
scanf("%s", str);
printf("You entered: %s\n", str);
```

在上面的示例中，我们使用`scanf`函数输入一个字符串，并使用`printf`函数输出该字符串。

**3、字符串比较**

可以使用`strcmp`函数来比较两个字符串是否相等。如果两个字符串相等，则返回0；否则返回一个非零值。

示例代码：

```c
char str1[] = "Hello";
char str2[] = "World";
int result = strcmp(str1, str2);
if (result == 0) {
    printf("The two strings are equal.\n");
} else {
    printf("The two strings are not equal.\n");
}
```

在上面的示例中，我们比较了两个字符串`str1`和`str2`，并根据比较结果输出不同的消息。

**4、字符串复制和拼接**

可以使用`strcpy`函数来将一个字符串复制到另一个字符串中，使用`strcat`函数将一个字符串拼接到另一个字符串中。

示例代码：

```c
char str1[20] = "Hello";
char str2[20] = "World";
strcpy(str2, str1);
printf("str2: %s\n", str2); // 输出 "Hello"
strcat(str2, ", World!");
printf("str2: %s\n", str2); // 输出 "Hello, World!"
```

在上面的示例中，我们将字符串`str1`复制到`str2`中，并使用`strcat`函数将字符串`", World!"`拼接到`str2`中。

**5、字符串长度和查找**

可以使用`strlen`函数来获取一个字符串的长度，使用`strchr`函数查找一个字符串中的某个字符，并使用`strstr`函数查找一个字符串中的另一个字符串。

示例代码：

```c
char str[] = "Hello, world!";
int length = strlen(str);
printf("Length of str: %d\n", length); // 输出 13
char *p = strchr(str, 'w');
if (p != NULL) {
    printf("Found 'w' at position %d.\n", p - str);
} else {
    printf("Did not find 'w'.\n");
}
char *q = strstr(str, "world");
if (q != NULL) {
    printf("Found 'world' at position %d.\n", q - str);
} else {
printf("Did not find 'world'.\n");
} 
```

在上面的示例中，我们获取了字符串`str`的长度，并使用`strchr`函数查找`'w'`字符和`strstr`函数查找`"world"`子串。

## 二进制，八进制，十进制，十六进制

在C语言中，你可以使用不同的前缀来表示二进制、八进制、十进制和十六进制数。下面是一些示例：

1. **二进制数**：

   使用前缀 `0b` 或 `0B` 来表示二进制数。

   ```c
   int binary = 0b1010; // 十进制的 10
   ```

2. **八进制数**：

   使用前缀 `0` 来表示八进制数。

   ```c
   int octal = 012; // 十进制的 10
   ```

3. **十进制数**：

   不需要前缀，直接使用数字表示。

   ```c
   int decimal = 10;
   ```

4. **十六进制数**：

   使用前缀 `0x` 或 `0X` 来表示十六进制数。

   ```c
   int hexadecimal = 0xA; // 十进制的 10
   ```

下面是一个完整的示例程序，演示了如何声明和打印这些不同进制的数：

```c
#include <stdio.h>

int main() {
    int binary = 0b1010;    // 二进制数
    int octal = 012;         // 八进制数
    int decimal = 10;        // 十进制数
    int hexadecimal = 0xA;   // 十六进制数

    printf("Binary: %d\n", binary);
    printf("Octal: %d\n", octal);
    printf("Decimal: %d\n", decimal);
    printf("Hexadecimal: %d\n", hexadecimal);

    return 0;
}
```

在这个示例中，我们声明了四个不同进制的整数，然后使用 `printf` 函数将它们打印出来。

## memset 和 sizeof

`memset` 和 `sizeof` 是 C 语言中的两个不同的功能。

1. **memset**:

   `memset` 是一个函数，用于将一块内存区域的内容设置为特定的值。其原型如下：

   ```c
   void *memset(void *ptr, int value, size_t num);
   ```

   - `ptr`：指向要设置值的内存块的指针。
   - `value`：要设置的值，以 `int` 类型表示，通常是一个字节的值。
   - `num`：要设置的字节数。

   例如，下面的代码将数组 `arr` 的前 10 个int都设置为 0：

   ```c
   int arr[100];
   memset(arr, 0, 10 * sizeof(int));
   ```

2. **sizeof**:

   `sizeof` 是一个操作符，用于计算数据类型或变量的大小（以字节为单位）。它返回其操作数的大小。

   例如，`sizeof(int)` 返回 `int` 类型的大小（通常是 4 字节），`sizeof(arr)` 返回数组 `arr` 的总字节数。

   在上面的 `memset` 的示例中，`sizeof(int)` 用于计算要设置的字节数，确保我们只设置了数组的前 10 个整数的大小。

总结：`memset` 用于设置内存块的值，而 `sizeof` 用于计算数据类型或变量的大小。在某些情况下，它们可能会一起使用，比如确定要设置的内存块的大小。

## 函数指针

函数指针是指向函数的指针变量。在C语言中，函数指针允许你在运行时动态地选择调用哪个函数，这在某些情况下非常有用，比如实现回调函数、动态调用函数等。

函数指针的声明方式如下：

```c
return_type (*pointer_name)(parameter_list);
```

- `return_type` 是函数返回类型。
- `pointer_name` 是函数指针的名称。
- `parameter_list` 是函数参数列表。

下面是一个简单的示例，演示如何声明和使用函数指针：

```c
#include <stdio.h>

// 声明一个函数
int add(int a, int b) {
    return a + b;
}

int main() {
    // 声明一个指向函数的指针
    int (*ptr)(int, int);

    // 将函数的地址赋值给指针
    ptr = add;

    // 通过函数指针调用函数
    int result = ptr(10, 20);
    printf("Result: %d\n", result);

    return 0;
}
```

在上面的例子中，我们声明了一个函数 `add()`，然后声明了一个指向函数 `add()` 的指针 `ptr`。我们将 `add()` 函数的地址赋值给了指针 `ptr`。然后，通过 `ptr` 调用了 `add()` 函数，得到了相加的结果并打印输出。

函数指针在许多高级的C编程场景中非常有用，比如实现回调函数、事件处理等。

## 宏定义和typedef

### 宏定义

C语言中的宏定义是一种预处理指令，允许程序员在编译之前定义一些标识符，这些标识符在源代码中会被预处理器替换成特定的文本。宏定义通常用于创建代码片段的别名，或者在编译时进行简单的文本替换。

宏定义使用 `#define` 指令来创建，其基本语法如下：

```c
#define 宏名称 替换文本
```

其中：

- `宏名称` 是你希望定义的标识符或名称。
- `替换文本` 是在程序中出现宏名称时要替换的文本。

例如，下面是一个简单的宏定义示例：

```c
#define PI 3.14159
```

在这个示例中，`PI` 是宏名称，`3.14159` 是替换文本。在程序的其他地方，每当预处理器遇到 `PI`，它都会将其替换为 `3.14159`。

宏定义也可以带有参数，类似于函数。例如：

```c
#define SQUARE(x) ((x) * (x))
```

在这个示例中，`SQUARE` 是带有一个参数的宏。当你在程序中使用 `SQUARE(5)` 时，预处理器会将其替换为 `(5) * (5)`，结果为 `25`。

需要注意的是，宏定义是简单的文本替换，没有类型检查和作用域。因此，在使用宏定义时，要小心确保替换的文本在逻辑上是正确的，并且要避免出现意外的副作用。

**\#ifndef、#define、#ifdef、#endif **

下面是一个简单的例子，演示了如何使用预处理指令 `#ifndef`、`#define`、`#ifdef` 和 `#endif` 来防止头文件被多次包含：

假设我们有一个头文件 `example.h`，内容如下：

```c
#ifndef EXAMPLE_H
#define EXAMPLE_H

// 在这里放置头文件的内容

#endif
```

现在解释一下这段代码：

- `#ifndef EXAMPLE_H` 检查是否定义了名为 `EXAMPLE_H` 的宏。如果 `EXAMPLE_H` 未定义，则进入下一步；如果定义了，就跳过 `#ifndef` 到 `#endif` 之间的代码。
- `#define EXAMPLE_H` 定义 `EXAMPLE_H` 宏，这样下次再遇到 `#ifndef EXAMPLE_H` 就会跳过，因为 `EXAMPLE_H` 已经被定义了。
- `#endif` 结束条件编译块。

在实际编写代码时，我们可以在需要包含 `example.h` 的源文件中使用 `#include "example.h"`。这样，预处理器会在编译时检查 `example.h` 是否被包含，如果没有被包含，则会将其包含，否则会跳过。

这种技术避免了多次包含同一个头文件可能导致的重定义错误，并且确保头文件的内容只被包含一次。

### typedef

`typedef` 是 C 语言中的一个关键字，用于创建类型别名。它允许程序员为已有的数据类型定义一个新的名字，从而使代码更具可读性、易于理解和维护。

`typedef` 的基本语法如下：

```c
typedef 原类型名 新类型名;
```

例如，假设我们想要为 `int` 类型定义一个新的名字 `Integer`，可以这样做：

```c
typedef int Integer;
```

现在，`Integer` 就可以作为 `int` 类型的别名使用。我们可以像使用 `int` 一样使用 `Integer`：

```c
Integer num1, num2;
```

`typedef` 最常用于创建复杂的数据类型，例如结构体、联合体和函数指针类型的别名，以使代码更具可读性和可维护性。例如：

```c
typedef struct {
    int x;
    int y;
} Point;

typedef void (*FunctionPointer)(int, int);
```

在这些示例中，`Point` 是一个结构体类型的别名，`FunctionPointer` 是一个函数指针类型的别名。

使用 `typedef` 可以使代码更易于理解，并且可以减少出错的可能性，特别是当使用复杂的数据类型时。

**当我们使用 `typedef` 来定义一个函数指针类型时，通常是为了简化代码，提高可读性。这种类型的定义在很多情况下都很有用，比如在回调函数中、在函数参数中传递函数指针等。**

下面是一个具体的例子，展示了如何使用 `typedef` 来定义一个函数指针类型，以及如何使用这个类型来声明函数指针变量和调用函数：

```c
#include <stdio.h>

// 定义一个函数指针类型，该函数没有返回值，接受两个整型参数
typedef void (*FunctionPointer)(int, int);

// 定义一个函数，接受一个函数指针作为参数，并调用该函数指针所指向的函数
void executeCallback(FunctionPointer callback, int a, int b) {
    printf("Executing callback function...\n");
    callback(a, b);
}

// 定义一个简单的回调函数，打印两个整数的和
void callbackFunction(int x, int y) {
    printf("Callback function result: %d\n", x + y);
}

int main() {
    // 声明一个函数指针变量，指向 callbackFunction 函数
    FunctionPointer ptr = &callbackFunction;

    // 调用 executeCallback 函数，将 ptr 作为参数传递进去
    executeCallback(ptr, 10, 20);

    return 0;
}
```

在这个例子中，我们首先使用 `typedef` 定义了一个函数指针类型 `FunctionPointer`，它可以指向没有返回值、接受两个整型参数的函数。

然后，我们定义了一个函数 `executeCallback`，它接受一个 `FunctionPointer` 类型的参数 `callback`，并调用了这个函数指针所指向的函数。

接着，我们定义了一个简单的回调函数 `callbackFunction`，它接受两个整型参数，并打印它们的和。

在 `main` 函数中，我们声明了一个 `FunctionPointer` 类型的变量 `ptr`，并将其指向 `callbackFunction` 函数。然后，我们调用 `executeCallback` 函数，并将 `ptr` 作为参数传递进去，以触发回调函数的执行。

这样，通过使用 `typedef` 定义函数指针类型，我们可以使代码更加清晰易读，同时提高了代码的灵活性和可维护性。

**define和typedef的关系**

`#define Integer int` 是合法的预处理指令，它将 `Integer` 定义为 `int` 的别名。这意味着在代码中使用 `Integer` 将会被预处理器替换为 `int`。

例如：

```c
#include <stdio.h>

#define Integer int

int main() {
    Integer num = 10;
    printf("Value of num: %d\n", num);
    return 0;
}
```

在这个示例中，`Integer` 被预处理器替换为 `int`，因此代码会被展开为：

```c
#include <stdio.h>

int main() {
    int num = 10;
    printf("Value of num: %d\n", num);
    return 0;
}
```

这样，`Integer` 就被替换为了 `int`，并且代码可以正常编译运行。

## C 强制类型转换

强制类型转换是把变量从一种类型转换为另一种数据类型。例如，如果您想存储一个 long 类型的值到一个简单的整型中，您需要把 long 类型强制转换为 int 类型。您可以使用**强制类型转换运算符**来把值显式地从一种类型转换为另一种类型，如下所示：

```c
(type_name) expression
```

请看下面的实例，使用强制类型转换运算符把一个整数变量除以另一个整数变量，得到一个浮点数：

```c
#include <stdio.h>  
int main() {   
    int sum = 17, count = 5;   
    double mean;
    mean = (double) sum / count;
    printf("Value of mean : %f\n", mean );  
}
```

当上面的代码被编译和执行时，它会产生下列结果：

```
Value of mean : 3.400000
```

这里要注意的是强制类型转换运算符的优先级大于除法，因此 **sum** 的值首先被转换为 **double** 型，然后除以 count，得到一个类型为 double 的值。

类型转换可以是隐式的，由编译器自动执行，也可以是显式的，通过使用**强制类型转换运算符**来指定。在编程时，有需要类型转换的时候都用上强制类型转换运算符，是一种良好的编程习惯。

### 整数提升

整数提升是指把小于 **int** 或 **unsigned int** 的整数类型转换为 **int** 或 **unsigned int** 的过程。请看下面的实例，在 int 中添加一个字符：

```c
#include <stdio.h>  
int main() {
    int  i = 17;
    char c = 'c'; /* ascii 值是 99 */
    int sum;
    sum = i + c;
    printf("Value of sum : %d\n", sum );  
}
```

当上面的代码被编译和执行时，它会产生下列结果：

```
Value of sum : 116
```

在这里，sum 的值为 116，因为编译器进行了整数提升，在执行实际加法运算时，把 'c' 的值转换为对应的 ascii 值。

### 常用的算术转换

**常用的算术转换**是隐式地把值强制转换为相同的类型。编译器首先执行**整数提升**，如果操作数类型不同，则它们会被转换为下列层次中出现的最高层次的类型：

```c
#include <stdio.h>
 
int main()
{
   int  i = 17;
   char c = 'c'; /* ascii 值是 99 */
   float sum;
 
   sum = i + c;
   printf("Value of sum : %f\n", sum );
 
}
```

当上面的代码被编译和执行时，它会产生下列结果：

```text
Value of sum : 116.000000
```

在这里，c 首先被转换为整数，但是由于最后的值是 float 型的，所以会应用常用的算术转换，编译器会把 i 和 c 转换为浮点型，并把它们相加得到一个浮点数。

## typeof的使用

`typeof`通常用于宏定义中。在C语言中，`typeof`通常用于宏定义中，它允许你在编译时获取表达式的类型，然后可以将该类型用于声明变量、函数参数类型等。这对于编写通用的代码和宏非常有用，因为它可以让你编写更加灵活和通用的宏，而不需要显式指定类型。

下面是一个简单的示例，演示了如何在宏中使用`typeof`：

```c
#include <stdio.h>

// 定义一个宏，交换两个变量的值
#define SWAP(x, y) do { 
    typeof(x) temp = x; 
    x = y; 
    y = temp; 
} while(0)

int main() {
    int a = 5, b = 10;
    
    printf("Before swapping: a = %d, b = %d\n", a, b);
    SWAP(a, b);
    printf("After swapping: a = %d, b = %d\n", a, b);
    
    return 0;
}
```

在这个例子中，`typeof(x)`允许宏在编译时确定变量`x`的类型，并在`temp`变量的声明中使用它。这样，无论`x`是什么类型的变量，`SWAP`宏都可以正确地工作。

可以看到，`typeof()`中可以是任何有类型的东西，变量就是其本身的类型，函数是它返回值的类型。`typeof`一般用于声明变量，如：

```c
typeof(a) var;
```

不过，这也不是绝对的，从语法上来说，所有可以出现基本类型关键词的地方都可以使用`typeof`，比如`sizeof(typeof(a))`这样的用法，虽然这里的`typeof`是多余的，不过它是符合语法的。

再来看一些高级用法：

```c
int fun(int a);
typeof(fun) * fptr;    // int (*fptr)(int);

typeof(int *)a, b;     // int * a, * b;
typeof(int) * a, b;    // int * a, b;
```

可以看到，`typeof`还可以用来定义函数指针等，且`typeof(int *)a, b`是定义了两个指针变量。

最后指出一些需要注意的问题。`typeof()`是在编译时处理的，故**其中的表达式在运行时是不会被执行的**，比如`typeof(fun())`，`fun()`函数是不会被执行的，`typeof`只是在编译时分析得到了`fun()`的返回值而已。`typeof`还有一些局限性，其中的变量是不能包含存储类说明符的，如`static`、`extern`这类都是不行的。

## 可变参数

C语言中的可变参数是指函数能够接受数量可变的参数。在C语言中，可变参数的处理主要依赖于 `<stdarg.h>` 头文件中的一组宏和函数。

下面是一个简单的示例，展示了如何使用可变参数函数 `sum()` 来计算任意数量的整数的和：

```c
#include <stdio.h>
#include <stdarg.h>

// 定义一个可变参数函数
int sum(int num, ...) {
    int result = 0;
    va_list args; // 定义一个va_list类型的变量

    // 使用宏va_start初始化可变参数列表
    va_start(args, num);

    // 遍历参数列表，并累加参数值
    for (int i = 0; i < num; i++) {
        result += va_arg(args, int); // 使用宏va_arg获取下一个参数的值
    }

    // 使用宏va_end结束可变参数列表的使用
    va_end(args);

    return result;
}

int main() {
    // 调用sum函数，传入不定数量的参数
    int total = sum(3, 10, 20, 30);
    printf("Total: %d\n", total);

    return 0;
}
```

在这个例子中，`sum()` 函数可以接受不定数量的参数，首先接收一个参数 `num` 表示后续参数的数量。然后使用 `<stdarg.h>` 中的宏来处理可变参数列表。

- `va_list` 是一个类型，用于声明可变参数列表的变量。
- `va_start` 宏用于初始化可变参数列表，传入 `va_list` 变量和最后一个固定参数的地址。
- `va_arg` 宏用于从可变参数列表中获取下一个参数的值，它需要一个 `va_list` 变量和参数的类型。
- `va_end` 宏用于结束可变参数列表的使用，传入 `va_list` 变量。

通过这些宏，你可以在函数中处理任意数量的参数。请注意，在使用可变参数时，要特别小心，确保参数的数量和类型与函数的预期相匹配，以避免出现不确定的行为。

## 位域

位域（Bit-fields）是一种C语言特性，允许你在一个数据结构中定义成员变量占用特定数量的位数。通常，位域用于节省内存空间，并在处理硬件相关的数据结构时特别有用，比如处理寄存器。

在C语言中，位域的定义形式如下：

```c
struct {
   type member_name : width;
};
```

- `type`：成员的数据类型。
- `member_name`：成员的名称。
- `width`：成员占用的位数。

下面是一个示例：

```c
#include <stdio.h>

struct {
   unsigned int a:4; // a占用4位
   unsigned int b:6; // b占用6位
   unsigned int c:26; // c占用26位
} Bits;

int main() {
   Bits.a = 2;
   Bits.b = 3;
   Bits.c = 64;

   printf("a: %d\n", Bits.a);
   printf("b: %d\n", Bits.b);
   printf("c: %d\n", Bits.c);

   printf("Size of structure: %lu bytes\n", sizeof(Bits));

   return 0;
}
```

在这个例子中，我们定义了一个结构体，其中包含了三个位域成员变量 `a`、`b` 和 `c`，分别占用了4位、6位和26位。在 `main()` 函数中，我们给每个位域成员赋值，并打印它们的值以及整个结构体的大小。

注意：使用位域时，需要小心位域的宽度和类型的兼容性，因为一些编译器对位域的实现可能有所不同。此外，位域的行为也会受到字节序的影响。

