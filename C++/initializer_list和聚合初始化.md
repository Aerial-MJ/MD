## c++ 11 initializer_list

**C++11**引入了`initializer_list`，这是一个用于**初始化对象的轻量级、固定长度的数组**。`initializer_list`可以被用于以下场景：

1. 初始化对象的构造函数中。
2. 参数传递。
3. 函数返回值。

在定义一个`initializer_list`时，需要使用大括号将其括起来。例如，下面的代码定义了一个`initializer_list`，其中包含了三个整数：
```c++
std::initializer_list<int> mylist = {10，20，30};
```
在使用`initializer_list`时，可以使用auto关键字来自动推断其类型。例如，下面的代码使用auto来推断`mylist`的类型：

```c++
auto mylist = {10,20,30};
```

由于`initializer_list`使用了模板技术，因此它可以用于任意类型的对象，包括自定义类型。
总的来说，`initializer_list`为C++程序员提供了一种方便、简洁的方式来初始化对象，特别是在自定义类型中使用时，可以减少代码量并提高代码的可读性。

```c++
#include <iostream>
#include <vector>
#include <initializer_list>
 
template <class T>
struct S {
    std::vector<T> v;
    S(std::initializer_list<T> l) : v(l) {
         std::cout << "constructed with a " << l.size() << "-element list\n";
    }
    void append(std::initializer_list<T> l) {
        v.insert(v.end(), l.begin(), l.end());
    }
    std::pair<const T*, std::size_t> c_arr() const {
        return {&v[0], v.size()};  // 在 return 语句中复制列表初始化
                                   // 这不使用 std::initializer_list
    }
};
 
template <typename T>
void templated_fn(T) {}
 
int main()
{
    S<int> s = {1, 2, 3, 4, 5}; // 复制初始化
    s.append({6, 7, 8});      // 函数调用中的列表初始化
 
    std::cout << "The vector size is now " << s.c_arr().second << " ints:\n";
 
    for (auto n : s.v)
        std::cout << n << ' ';
    std::cout << '\n';
 
    std::cout << "Range-for over brace-init-list: \n";
 
    for (int x : {-1, -2, -3}) // auto 的规则令此带范围 for 工作
        std::cout << x << ' ';
    std::cout << '\n';
 
    auto al = {10, 11, 12};   // auto 的特殊规则
 
    std::cout << "The list bound to auto has size() = " << al.size() << '\n';
 
//    templated_fn({1, 2, 3}); // 编译错误！“ {1, 2, 3} ”不是表达式，
                             // 它无类型，故 T 无法推导
    templated_fn<std::initializer_list<int>>({1, 2, 3}); // OK
    templated_fn<std::vector<int>>({1, 2, 3});           // 也 OK
}
```

输出：

```text
constructed with a 5-element list
The vector size is now 8 ints:
1 2 3 4 5 6 7 8
Range-for over brace-init-list: 
-1 -2 -3 
The list bound to auto has size() = 3
```

## 聚合初始化

定义如下结构体：

```c++
struct A{
    double a;
    int b;
    bool c;
}
```

**A a={1.0,1,false} ;为什么合法**

这是一种使用聚合初始化方式来创建结构体 A 的对象 a 的方法，其中 {1.0, 1, false} 按顺序分别对应结构体 A 中的成员 a、b、c。在这种情况下，编译器会自动为结构体 A 的成员赋值，并且不需要写构造函数。

**该声明方式不是initializer_list**

A a={1.0,1,false} 并不是使用 initializer_list 进行初始化，而是使用聚合初始化的方式。

### vector\<int> vec = {1, 2, 3}是initializer_list

在 C++11 中引入了 initializer_list，它是一种用于**初始化容器类型（如 std::vector、std::map 等）**的特殊语法。例如，可以使用以下语法来创建一个包含三个整数的 std::vector 对象：

```c++
std::vector<int> vec = {1, 2, 3};
```

在这种情况下，编译器会将 {1, 2, 3} 视为一个 initializer_list\<int> 对象，并调用 std::vector\<int> 类的构造函数来创建 vec 对象。

### 证明是initializer_list

基于 C++ 语言标准（C++ Standard），特别是关于聚合初始化和初始化列表的定义和规则。

根据 C++ 标准，聚合类型的初始化可以使用花括号进行。以下是对聚合类型初始化的定义（C++17 标准草案，第11.6.1节）：

> An aggregate is an array or a class (Clause 12) with no user-declared constructors (12.1), no private or protected non-static data members (Clause 11), no base classes (Clause 13), and no virtual functions (13.3).
>
> When an aggregate is initialized the initializer can contain an initializer-clause consisting of a brace-enclosed, comma-separated list of initializer-clauses for the members of the aggregate, written in increasing subscript or member order.

这意味着聚合类型的初始化可以使用花括号，并且可以使用成员的初始化列表。

接下来，我们看一下 `std::vector` 类型是否被定义为聚合类型。根据 C++ 标准（C++17 标准草案，第23.3.6节），`std::vector` 是一个模板类，其定义如下：

```cpp
template<class T, class Allocator = allocator<T>>
class vector {
    // ...
};
```

根据这个定义，`std::vector` 是一个类模板，它接受一个类型参数 `T` 和一个可选的分配器类型参数 `Allocator`。

C++ 标准中没有明确指定 `std::vector` 是否是一个聚合类型，但根据标准定义，它是一个类模板，而不是一个数组或只包含数据成员的类。因此，根据标准的定义，`std::vector` 不是一个聚合类型。

由于 `std::vector` 不是一个聚合类型，我们不能使用聚合初始化语法来初始化它。相反，`std::vector` 类有多个构造函数，其中之一接受 `std::initializer_list` 作为参数，以实现初始化列表的语义。

因此，我们可以确定给定的代码 `std::vector<int> vec = {1, 2, 3};` 使用的是 `std::initializer_list` 初始化列表，而不是聚合初始化。

请注意，C++ 标准的不同版本可能会对这些定义和规则有所不同。上述证明基于 C++17 标准草案的定义。如果你在使用不同的标准版本，可能需要参考相应的标准文件来获取准确的定义和规则。

**需要注意的是，只有符合特定条件的类才可以使用聚合初始化，否则需要使用构造函数进行初始化。**

## 聚合初始化条件

聚合初始化是一种在 C++ 中用于初始化聚合类型的语法。聚合类型是指没有用户定义的构造函数、没有基类和虚函数、所有非静态成员都是公有的并且没有初始值的类或结构体类型。常见的聚合类型包括数组、结构体、联合等。

在 C++11 及以后的版本中，可以使用花括号 {} 来进行聚合初始化。例如，定义一个结构体类型：

```c++
struct MyStruct {
    int x;
    float y;
    char z;
};

```

**可以使用聚合初始化的语法来创建 MyStruct 的对象：**

```c++
MyStruct ms = {1, 2.0f, 'a'};
```

在这个例子中，我们使用了花括号来创建了一个 MyStruct 的对象 ms，并用三个逗号分隔的值对其成员进行了初始化。C++ 编译器会根据这些值的类型和顺序自动对结构体的成员进行赋值。

需要注意的是，聚合初始化只适用于聚合类型，对于其他类型（如具有构造函数的类类型），需要使用相应的构造函数来进行初始化。

## class 类 可以使用聚合初始化

对于 C++11 及以后的标准，如果一个 class 满足聚合类型的定义，那么它也可以使用聚合初始化。一个 class 满足聚合类型的定义，必须满足以下条件：

1. 所有非静态成员都是 public 的。
2. 没有用户自定义的构造函数。
3. 没有私有、保护或虚基类。
4. 没有虚函数。

例如，以下是一个满足聚合类型定义的 class：

```c++
class Point {
public:
    int x;
    int y;
};
```

可以使用花括号进行聚合初始化：

```c++
Point p = {1, 2};
```

注意，如果 class 有用户自定义的构造函数，则无法使用聚合初始化语法进行初始化。此时，需要使用相应的构造函数来进行初始化。

在 C++ 中，花括号 `{}` 有多种语义。当用于聚合类型的初始化时，它表示聚合初始化列表，用于初始化聚合类型的成员；当用于创建数组对象时，它表示数组初始化列表，用于初始化数组元素；当用于创建 `std::initializer_list` 对象时，它表示初始化列表，用于初始化 `std::initializer_list` 对象的元素。

对于 `pair<int, double> p={1, 1}` 这个语句来说，它使用了聚合初始化，而不是 `std::initializer_list`。这是因为 `pair` 类型没有定义接受 `std::initializer_list` 作为参数的构造函数，所以不能使用 `std::initializer_list` 进行初始化。

```c++
#include<iostream>
#include<cstring>
#include<algorithm>

class A{
public:
    int a;
    int b;
};

int main(){

    A a{1,2};
    A b({1,2});
    A c={1,2};
    A d(A ({1,2}));
    A e(A {1,2});
    A f=A ({1,2});
    A g=A{1,2};
    
    return 0;
}
```

## 只有聚合类型才能聚合初始化

在C++中，只有满足特定条件的类型被定义为聚合类型，才能使用聚合初始化语法。以下是聚合类型的一些特征：

1. 没有用户声明的构造函数：聚合类型不能有任何显式声明的构造函数，包括默认构造函数、拷贝构造函数和移动构造函数。

2. 没有私有或受保护的非静态数据成员：聚合类型不能有私有或受保护的非静态数据成员。

3. 没有基类：聚合类型不能有基类。

4. 没有虚函数：聚合类型不能有虚函数。

对于满足上述条件的类型，可以使用聚合初始化语法进行初始化，这意味着可以使用花括号进行初始化，同时可以提供成员的初始化列表。

需要注意的是，C++11之后的标准引入了统一的初始化语法，使得聚合初始化的使用更加广泛。即使一个类型不是聚合类型，也可以使用花括号初始化语法来进行初始化，只是语义上有所区别。

对于`std::vector`这样的标准库容器类型，它不满足聚合类型的定义，因此不能使用聚合初始化语法进行初始化。相反，它具有适当的构造函数来接受不同的初始化参数，包括接受`std::initializer_list`的构造函数，以支持初始化列表的语义。
