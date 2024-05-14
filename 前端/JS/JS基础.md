# JS基础

## path.join()

### 使用 path.join() 的步骤

1. **引入 `path` 模块**：
   Node.js内置了`path`模块，因此你只需要引入它即可开始使用。

2. **使用 `path.join()` 方法连接路径**：
   通过传递路径的各个部分作为参数，`path.join()` 将它们合并成一个规范化的路径。

### 示例代码

以下是一个使用 `path.join()` 的示例，演示如何在Node.js中合并路径部分：

```javascript
// 引入path模块
const path = require('path');

// 使用path.join()合并路径
const directory = 'Users';
const subdirectory = 'Documents';
const filename = 'example.txt';

const fullPath = path.join(directory, subdirectory, filename);

console.log(fullPath);
// 在Windows系统上输出: 'Users\Documents\example.txt'
// 在Unix系统上输出: 'Users/Documents/example.txt'
```

### 注意事项

- `path.join()` 会智能地处理多余的斜杠和相对路径符号（如`..`和`.`）。
- 路径中的参数可以是绝对或相对路径，`path.join()` 将从左到右处理它们，以生成规范化的路径。
- 如果任何一个路径片段是绝对路径，那么它之前的所有路径都将被忽略，并从该绝对路径片段开始重新构建路径。

### 在浏览器中处理路径

在浏览器中，通常不需要像在服务器端（例如Node.js）那样处理文件系统路径。然而，如果你需要在浏览器中处理URL或路径字符串，你可以自己编写函数来连接字符串，或者使用URL对象来处理：

```javascript
function joinUrl(...parts) {
    return parts.reduce((acc, part) => {
        return acc.replace(/\/+$/, '') + '/' + part.replace(/^\/+/, '');
    });
}

const fullUrl = joinUrl('http://example.com/', '/users/', '/john/');
console.log(fullUrl);  // 输出: 'http://example.com/users/john/'
```

这段代码提供了一个简单的函数来模拟类似`path.join()`的行为，用于连接URL的各个部分。

在大多数前端JavaScript用例中，处理的是URL而不是文件路径，因此确保使用正确的工具和方法来处理你的特定情况非常重要。

## 正则表达式

在JavaScript中，创建正则表达式有两种主要方式：使用字面量语法和使用构造函数语法。这两种方法各有特点，适用于不同的场景。

### 1. 字面量语法

字面量语法是通过直接在代码中书写正则表达式的方式创建正则表达式。这是最简单和最常用的方式，特别是当正则表达式在编译时就已知并且不会改变时。

**语法：**
```javascript
var regex = /pattern/flags;
```

**示例：**
```javascript
var re = /ab+c/i;
```
这个正则表达式用于匹配任意包含一个 'a' 后跟一个或多个 'b'，然后是一个 'c' 的字符串，忽略大小写。

### 2. 构造函数语法

构造函数语法使用 `RegExp` 对象的构造函数来创建正则表达式。这种方法在你需要动态生成正则表达式的字符串时非常有用，例如，当你需要在运行时根据用户的输入或其他数据来源构建表达式时。

**语法：**
```javascript
var regex = new RegExp("pattern", "flags");
```

**示例：**
```javascript
var pattern = "ab+c";
var flags = "i";
var re = new RegExp(pattern, flags);
```
这个例子与之前的字面量例子相同，但是使用构造函数创建正则表达式，这使得可以动态地设置模式和标志。

### 两种方式的对比

- **编译时间**：字面量语法的正则表达式在脚本加载时编译，而构造函数语法的正则表达式在运行时编译。因此，字面量语法的性能通常略好，特别是当正则表达式在代码中重复使用时。
- **灵活性**：构造函数语法在需要动态生成正则表达式的场景下非常有用，因为你可以传递任何字符串作为正则表达式的模式和标志。
- **易读性**：对于固定的、简单的正则表达式，使用字面量语法通常更清晰易读。

### 使用示例

以下是如何在实际代码中应用这两种方式的简单例子：

```javascript
// 使用字面量语法匹配邮箱
var emailPattern = /\S+@\S+\.\S+/;
var testEmail = "example@example.com";
console.log(emailPattern.test(testEmail));  // 输出: true

// 使用构造函数语法根据用户输入匹配
var userInput = "hello";
var userInputRegex = new RegExp("^" + userInput + "$", "i");
console.log(userInputRegex.test("Hello"));  // 输出: true
```

在选择使用哪种方式时，可以考虑正则表达式是否需要动态生成，以及是否需要在代码的可读性和性能之间做权衡。对于大多数固定的正则表达式，推荐使用字面量语法。对于需要根据变量构建的正则表达式，使用构造函数语法更合适。

当你在JavaScript代码中看到类似 `var regex = /pattern/flags;` 的语句时，这是在使用正则表达式的字面量语法来创建一个正则表达式对象。这种语法非常简洁，直接定义了正则表达式的模式和可选的标志（flags）。下面，我将分解这个结构并解释每个部分的作用。

### 1. 正则表达式的组成

在 `/pattern/flags` 中：

- `/` 和 `/`：这两个斜杠之间的内容被视为正则表达式的模式（或规则）。这是正则表达式的主体部分，定义了你想要匹配的字符串模式。
- `pattern`：这是你希望匹配的文本模式。例如，`ab+c` 表示匹配一个字符串，其中有一个 'a'，后面跟着一个或多个 'b'，然后是一个 'c'。
- `flags`：这是一系列可选的字符，用来改变正则表达式的搜索行为。常见的标志包括：
  - `g`：全局搜索，意味着查找字符串中的所有匹配，而不是在第一个匹配后停止。
  - `i`：忽略大小写。
  - `m`：多行模式，影响 `^` 和 `$` 的行为。

### 2. 示例解释

让我们通过几个具体的例子来看看如何使用这种语法：

```javascript
// 匹配所有包含至少一个 'a' 的字符串，不区分大小写
var regex = /a+/i;

// 使用全局标志搜索字符串中的所有 'word'
var globalSearch = /word/g;

// 匹配一行开头的任何字符，多行模式
var multiLine = /^./m;
```

### 3. 在实际代码中的应用

使用创建的正则表达式对象进行匹配、搜索或替换操作：

```javascript
var text = "Hello World";
var pattern = /hello/i; // 不区分大小写

// 测试文本是否符合模式
console.log(pattern.test(text)); // 输出：true

// 使用正则表达式在文本中搜索匹配项
console.log(text.match(pattern)); // 输出：['Hello']
```

### 总结

`var regex = /pattern/flags;` 是一种创建正则表达式的方式，它通过直观的字面量语法使代码简洁而易于理解。使用这种方式可以方便地指定查找模式和行为标志，非常适合静态的正则表达式，这些表达式在代码编写时就已经确定，并且不需要基于运行时的数据动态生成。

## splice

`splice()` 是 JavaScript 中数组的一个方法，用于修改数组，它可以实现删除、插入和替换数组元素的功能。`splice()` 方法可以接受多个参数，下面是它的语法：

```javascript
array.splice(start[, deleteCount[, item1[, item2[, ...]]]])
```

- `start`: 必需，指定修改的起始位置（索引）。如果是负数，则从数组的末尾开始计数。
- `deleteCount`: 可选，指定要删除的元素个数。如果省略，则删除从 `start` 位置到数组末尾的所有元素。
- `item1`, `item2`, ...: 可选，要添加到数组的新元素。从 `start` 位置开始插入。

### 删除元素

如果只指定 `start` 参数，`splice()` 将从该位置开始删除数组中的所有元素（包括 `start` 位置的元素）。

```javascript
let fruits = ["Banana", "Orange", "Apple", "Mango"];
fruits.splice(2); // 从索引 2 开始删除所有元素
console.log(fruits); // ["Banana", "Orange"]
```

如果指定了 `start` 和 `deleteCount` 参数，则从 `start` 位置开始删除指定数量的元素。

```javascript
let fruits = ["Banana", "Orange", "Apple", "Mango"];
fruits.splice(1, 2); // 从索引 1 开始删除 2 个元素
console.log(fruits); // ["Banana", "Mango"]
```

### 插入元素

除了删除元素外，`splice()` 还可以在指定位置插入新元素。你可以通过添加额外的参数来实现。

```javascript
let fruits = ["Banana", "Orange", "Apple", "Mango"];
fruits.splice(2, 0, "Lemon", "Kiwi"); // 从索引 2 开始插入 "Lemon" 和 "Kiwi"
console.log(fruits); // ["Banana", "Orange", "Lemon", "Kiwi", "Apple", "Mango"]
```

### 替换元素

除了删除和插入元素外，`splice()` 还可以替换数组中的元素。

```javascript
let fruits = ["Banana", "Orange", "Apple", "Mango"];
fruits.splice(2, 1, "Lemon", "Kiwi"); // 从索引 2 开始替换 1 个元素，替换成 "Lemon" 和 "Kiwi"
console.log(fruits); // ["Banana", "Orange", "Lemon", "Kiwi", "Mango"]
```

总之，`splice()` 是一个非常有用的数组方法，它可以让你灵活地对数组进行修改、删除、插入和替换操作。
