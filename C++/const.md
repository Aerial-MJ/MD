# const 关键字

const除了修饰变量之外，还可以修饰函数，主要有以下几种形式

```cpp
const int& fun(int& a); //修饰返回值
int& fun(const int& a); //修饰形参
int& fun(int& a) const{} //const成员函数
```

## const修饰返回值

这种多是修饰返回值是引用类型的情况下，为了避免返回值被修改的情况。

返回值是引用的函数， 可以肯定的是这个引用必然不是临时对象的引用， 因此一定是成员变量或者是函数参数， 所以在返回的时候为了避免其成为左值被修改，就需要加上`const`关键字来修饰。

举个例子：

```cpp
#include<iostream>

using namespace std;

class A
{
private:
    int data;
public:
    A(int num):data(num){}
    ~A(){};
    int& get_data()
    {
        return data;
    }
};

int main()
{
    A a(1);
    a.get_data()=3;
    cout<<a.get_data()<<endl; //data=3
    return 0;
}
```

那么这个时候为了避免成员函数被修改就需要加上`const`关键字，这个时候如果试图改变返回值是不允许的：

```text
error: cannot assign to return value because function 'get_data' returns a const value
```

需要指出的是，如果函数的返回类型是内置类型，比如 int char 等，修改返回值本身就是不合法的！所以` const `返回值是处理返回类型为用户定义类型的情况。

## const修饰形参

多数情况下，我们都会选择 pass by reference，这样可以节省内存，并且可以起到改变实参的目的。不过有的时候我们并不希望改变实参的值，就要加上`const`关键字。

## const修饰成员函数

这种情况多数情形下很容易被忽视，其实这个是非常重要的一个内容。

设想这样一种场景：

```cpp
const String str("hello world");
str.print();
```

假设在string类中有一个成员函数叫做print， 如果这个函数在定义的时候没有加上const 关键字，上述的代码是无法通过编译的，下面举个具体的例子：

```cpp
#include<iostream>

using namespace std;

class A
{
private:
    int data;
public:
    A(int num):data(num){}
    ~A(){};
    int& get_data()
    {
        return data;
    }
};

int main()
{
    const A a(1);
    a.get_data();
    return 0;
}
```

毫不意外的出错了：

**error: 'this' argument to member function 'get_data' has type 'const A', but function is not marked const。**

我们敏锐的发现了一个“this"指针

其实任何成员函数的参数都是含有this 指针的，好比python中的 self ，只不过c++中规定全部不写这个参数， 其实这个参数就是对象本身， 即谁调用成员函数， 这个 this 就是谁！

我们的例子中 a 是一个`const` 对象， 它调用了 `get_data` 方法，所以函数签名应该是：`get_data(a){}`

而 a是一个 `const` 对象， 我们默认的 this 指针并不是 `const` 类型，所以参数类型不匹配， 编译无法通过！

这下我们终于清楚了， `const` 修饰成员函数， 根本上是修饰了 this 指针。

如果成员函数同时具有` const` 和` non-const` 两个版本的话， `const` 对象只能调用`const`成员函数， `non-const` 对象只能调用 `non-const `成员函数。