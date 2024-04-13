# Pair

## map.insert

无需指定插入位置，直接将键值对添加到 map 容器中。insert() 方法的语法格式有以下 2 种：

```c++
//1、引用传递一个键值对
pair<iterator,bool> insert (const value_type& val);
//2、以右值引用的方式传递键值对
template <class P>
pair<iterator,bool> insert (P&& val);
```

**右值引用**

```c++
//类型 && 引用名 = 右值表达式;

class A{};
A & rl = A();  //错误，无名临时变量 A() 是右值，因此不能初始化左值引用 r1
A && r2 = A();  //正确，因 r2 是右值引用

```

引用本身是**先取地址再解引用的过程**

## 顺序赋值

构造函数的顺序赋值

在C++中，struct结构体类型也可以使用构造函数进行初始化。当使用构造函数初始化struct结构体类型时，可以使用成员初始化列表来指定初始化顺序。

在成员初始化列表中，成员的初始化顺序与在struct结构体中定义的顺序相同。例如，如果在struct结构体中首先定义了成员变量a，其次是成员变量b，那么在成员初始化列表中也应该先初始化成员变量a，然后再初始化成员变量b。

以下是一个示例代码，演示了如何在struct结构体类型中使用构造函数进行初始化：

```c++
#include <iostream>
using namespace std;

struct Person {
    string name;
    int age;

    // 构造函数
    Person(string n, int a): name(n), age(a) {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

int main() {
    Person p("Tom", 20);
    return 0;
}
```

在上面的代码中，我们定义了一个名为Person的struct结构体类型，它包含了成员变量name和age。然后我们定义了一个构造函数，它接受两个参数，分别为name和age，并使用成员初始化列表来初始化结构体中的成员变量。

在main函数中，我们创建了一个名为p的Person类型对象，传递了两个参数："Tom"和20。在构造函数中，输出了初始化后的name和age的值。

需要注意的是，成员初始化列表中的初始化顺序与在struct结构体中定义的顺序相同，因此应该仔细考虑结构体中成员变量的定义顺序，以便在构造函数中正确地进行初始化。

