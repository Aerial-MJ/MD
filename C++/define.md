# #define

#define是c++的**宏定义**

`#ifdef #endif #ifndef`属于C/C++**预处理指令**，常见的预处理指令还包括``#include #define #undef #elif #error``等。

一般情况下，源程序中所有的行都参加编译。C/C++中有个概念叫做**条件编译。**条件编译”要求做到对**指定部分**内容编译。当满足某条件时对一组语句进行编译，而当条件不满足时则编译另一组语句。而这些预处理指令，可以帮助我们达到这个效果。

## #ifdef

```c++
#include<iostream>
using namespace std;
#define NYJ
 
int main()
{
#ifdef NYJ
	cout << "ifdef NYJ" << endl;
#else
	cout << "else" << endl;
#endif
}
```

## #ifndef

```c++
#include<iostream>
using namespace std;
 
int main()
{
#ifndef NYJ
	cout << "ifndef NYJ" << endl;
#else
	cout << "else" << endl;
#endif
}
```

## 防止双重定义

### 1.通过 #ifndef / #define 解决头文件重复包含

```c++
#ifndef __XXX_H__
#define __XXX_H__

int a=1;

#endif
```

```text
如果(没有定义宏__XXX_H__)
{
    那么直接定义宏__XXX_H__
    定义变量a 并且赋值为 1
}
结束程序
```

假如第一次包含时，由于没有定义宏` __XXX_H__`，所以做了两件事，定义宏 `__XXX_H__`，然后定义 int a = 1;
假如第二次包含时，由于已经定义宏` __XXX_H__`，所以啥都不做；
假如第N次包含时，由于已经定义宏 `__XXX_H__`，所以啥都不做；
整个过程，无论头文件被包含多少次，变量 a 只被定义一次，不会有重复包含重复定义的问题存在！

**需要注意的是**，`#if` 后面跟的是“整型常量表达式”，而 `#ifdef`和 `#ifndef` 后面跟的只能是一个宏名，不能是其他的。

### 2.通过 #pragma once 解决头文件重复包含

`#pragma once` 是上述方式的简写，好处是再也不会有两个头文件因为使用了同样的 `__XXX_H__` 而被忽略了。

```cpp
#pragma once

... ... // 声明、定义语句
```

`#pragma once`一般由编译器提供保证：同一个文件不会被包含多次。注意这里所说的"同一个文件"是指物理上的一个文件，而不是指内容相同的两个文件。
你无法对一个头文件中的一段代码作`pragma once`声明，而只能针对文件。
