# TS

TypeScript简称TS，JavaScript的超集，就是在JavaScript的基础上做一层封装，封装出TS的特性，最终可以编译为JavaScript。

TS最初是为了让习惯编写强类型语言的后端程序员能快速编写web应用，因为 JavaScript 没有强数据类型，所以 TypeScript 提供了静态数据类型，这是 TypeScript 的核心。

JavaScript 属于动态编程语言，而TypeScript属于静态编程语言。

**在每个变量后面加上:类型**   **(变量:类型)**

## TS的类型

ts 的常用基础类型分为两种： js 已有类型

1. 原始类型：1`number/string/boolean/null/undefined/symbol`
2. 对象类型：`object`（包括，数组、对象、函数等对象）

### 联合类型

需求：如何定义一个变量可以是null也可以是 number 类型? 这个时候，前面所学的已经不能满足我们的需求了，就需要用到一个新的类型 - 组合类型。 语法：

```typescript
let 变量: 类型1 | 类型2 | 类型3 .... = 初始值
let arr1 :number | string = 1 // 可以写两个类型
```

### 类型别名

在我们定义类型的时候，有时候自己定义的类型名往往很长，这个时候就需要在定义个别名，方便书写。 语法：

```typescript
type 别名 = 类型

type s = string // 定义

const str1:s = 'abc'
const str2:string = 'abc'
```

### 数组类型

```typescript
// 写法1：
let 变量: 类型[] = [值1，...]:
let numbers: number[] = [1, 3, 5] 
//  numbers必须是数组，每个元素都必须是数字
// 写法2：
let 变量: Array<类型> = [值1，...]
let strings: Array<string> = ['a', 'b', 'c'] 
//  strings必须是数组，每个元素都必须是字符串
```

### 函数

函数涉及的类型实际上指的是：`函数参数`和`返回值`的类型

#### 定义单个函数

语法：

```typescript
// 普通函数
function 函数名(形参1： 类型=默认值， 形参2：类型=默认值,...): 返回值类型 { }
// 声明式实际写法:
function add(num1: number, num2: number): number {
  return num1 + num2
}

// 箭头函数
const 函数名（形参1： 类型=默认值， 形参2：类型=默认值, ...):返回值类型 => { }
const add2 = (a: number =100, b: number = 100): number =>{
   return a + b
 }
 // 注意： 箭头函数的返回值类型要写在参数小括号的后面
add（1,'1') // 报错
```

#### 统一定义函数格式

当函数的类型一致时，写多个就会显得代码冗余，所以需要统一定义函数的格式 如下所示：

```typescript
const add2 = (a: number =100, b: number = 100): number => {
    return a + b
  }

function add1 (a:number = 100 , b: number = 200): number {
    return a + b
  }
// 这里的 add1 和 add2 的参数类型和返回值一致，
// 那么就可以统一定义一个函数类型
type Fn = (n1:number,n2:number) => number 
const add3 : Fn = (a,b)=>{return a+b }
// 这样书写起来就简单多啦
```

#### 函数返回值类型void

在 ts 中，如果一个函数没有返回值，应该使用 `void` 类型

```typescript
function greet(name: string): void {  console.log('Hello', name)  //}
```

可以用到`void` 有以下几种情况

1. 函数没写return
2. 只写了 return， 没有具体的返回值
3. return 的是 undefined

```typescript
// 如果什么都不写，此时，add 函数的返回值类型为： void
const add = () => {}

// 如果return之后什么都不写，此时，add 函数的返回值类型为： void
const add = () => { return }

const add = (): void => {
  // 此处，返回的 undefined 是 JS 中的一个值
  return undefined
}
// 这种写法是明确指定函数返回值类型为 void，与上面不指定返回值类型相同
const add = (): void => {}

```

那么就有人好奇，既然return undefined，那么为什么不可以直接在返回值那里写 :undefined 呢？ 如果函数没有指定返回值，调用结束之后，值是undefined的，但是不能直接声明返回值是undefined

```typescript
function add(a:number, b:number): undefined { // 这里会报错
  console.log(a,b)
}
```

#### 函数-可选参数

使用函数实现某个功能时，参数可以传也可以不传。

例如：数组的 slice 方法，可以 `slice()` 也可以 `slice(1)` 还可以 `slice(1, 3)` 那么就可以定义可选参数 语法：

```typescript
//可选参数：在可选参数名的后面添加 ?（问号）
function slice (a?: number, b?: number) {
    // ? 跟在参数名字的后面，表示可选的参数
    // 注意：可选参数只能在 必选参数的后面
    // 如果可选参数在必选参数的前面，会报错
    console.log(111);
    
  }
  slice()
  slice(1)
  slice(1,2)
}
```

##### 可选和默认值的区别

相同点： 调用函数时，可以少传参数

区别：设置了默认值之后，就是可选的了，不写就会使用默认值； 可选的参数一定有值。

注意：它们不能一起使用。优先使用默认值

### 对象类型-单独使用

格式： 方法有两种写法： 普通函数 和 箭头函数

```typescript
const 对象名: {
  属性名1：类型1，
  属性名2?：类型2，
  方法名1(形参1: 类型1，形参2: 类型2): 返回值类型,
  方法名2:(形参1: 类型1，形参2: 类型2) => 返回值类型
} = { 属性名1: 值1，属性名2：值2  }
```

#### 对象类型-类型别名

```typescript
// 创建类型别名
type Person = {
  name: string，
  age: number
  sayHi(): void
}

// 使用类型别名作为对象的类型：
let person: Person = {
  name: '小花',
  age: 18
  sayHi() {}
}
```

### 接口

当一个对象类型被多次使用时，有如下两种方式来来**描述对象**的类型，以达到复用的目的：

1. 类型别名，type
2. 接口，interface

语法：

```typescript
interface 接口名  {属性1: 类型1, 属性2: 类型2}
// 这里用 interface 关键字来声明接口
interface IGoodItem  {
	// 接口名称(比如，此处的 IPerson)，可以是任意合法的变量名称，推荐以 `I` 开头
   name: string, price: number, func: ()=>string
}

// 声明接口后，直接使用接口名称作为变量的类型
const good1: IGoodItem = {
   name: '手表',
   price: 200,
   func: function() {
       return '看时间'
   }
}
const good2: IGoodItem = {
    name: '手机',
    price: 2000,
    func: function() {
        return '打电话'
    }
}
```

 `interface`（接口）和` type`（类型别名）的对比：

- 相同点：都可以给对象指定类型
- 不同点:
  - 接口，只能为对象指定类型。**它可以继承。**
  - 类型别名，不仅可以为对象指定类型，实际上可以为**任意类型**指定别名

先有的 `interface`，后有的 `type`,推荐使用 `type`

```typescript
// 接口的写法-------------
interface IPerson {
	name: string,
	age: number
}

const user1：IPerson = {
	name: 'a',
	age: 20
}

// type的写法-------------
type Person  = {
	name: string,
	age: number
}
const user2：Person = {
	name: 'b',
	age: 20
}

```

#### 接口继承

如果两个接口之间有相同的属性或方法，可以将**公共的属性或方法抽离出来，通过继承来实现复用** 语法：

```typescript
interface 接口2 extends 接口1 {
 属性1： 类型1， // 接口2中特有的类型 
 }

interface a { x: number; y: number }
// 继承 a
// 使用 extends(继承)关键字实现了接口
interface b extends a {
  z: number
}
// 继承后，b 就有了 a 的所有属性和方法(此时，b 同时有 x、y、z 三个属性)
```

### 元组

**元组**是一种特殊的**数组**。有两点特殊之处

1. 它约定了的元素个数
2. 它约定了特定索引对应的数据类型

举个例子： 就拿 react 里面的 useState来举例：

```typescript
function useState(n: number): [number, (number)=>void] {
        const setN = (n1) => {
            n = n1
        }
        return [n, setN]
    	//返回元组，元素个数是2，第一个元素是一个number，第二个元素是函数
    }

const [num ,setNum] = useState(10)
```

### 字面量类型

例如下面的代码

```typescript
let str1 = 'hello TS'
const str2 = 'hello TS'
```

str1 的类型为 `string` 类型，str2 的类型为 `Hello TS`类型这是为啥呢？

1. str1 是一个变量(let)，它的值可以是任意字符串，所以类型为:string
2. str2 是一个常量(const)，它的**值不能变化**只能是 'hello TS'，所以，它的类型为:'hello TS'

注意：此处的 'Hello TS'，就是一个**字面量类型**，也就是说某个特定的字符串也可以作为 TS 中的类型

这时候就有人好奇了，那字面量类型有啥作用呢？ 字面量类型一般是配合联合类型一起使用的， 用来表示一组明确的可选值列表。 例如下面的例子：

```typescript
type Gender = 'girl' | 'boy'
// 声明一个类型，他的值 是 'girl' 或者是 'boy'
let g1: Gender = 'girl' // 正确
let g2: Gender = 'boy' // 正确
let g3: Gender = 'man' // 错误
```

### 枚举

枚举（enum）的功能类似于**字面量类型+联合类型组合**的功能，来描述一个值，该值只能是 一组命名常量 中的一个。 在没有 type 的时候，大家都是用枚举比较多的，现在比较少了。 语法：

```typescript
enum 枚举名 { 可取值1, 可取值2,.. }

// 使用格式：
//枚举名.可取值
```

注意：

1. 一般枚举名称以大写字母开头
2. 枚举中的多个值之间通过 `,`（逗号）分隔
3. 定义好枚举后，直接使用枚举名称作为类型注解

枚举也分数值枚举 和 字符串枚举。 数值枚举： 默认情况下，枚举的值是**数值**。默认为：从 0 开始自增的数值 当然，也可以给枚举中的成员初始化值

```typescript
enum Direction { Up = 10, Down, Left, Right }
// Down -> 11、Left -> 12、Right -> 13

enum Direction { Up = 2, Down = 3, Left = 8, Right = 16 }
```

字符串枚举：

```typescript
enum Direction {
  Up = 'UP',
  Down = 'DOWN',
  Left = 'LEFT',
  Right = 'RIGHT'
}
```

注意：字符串枚举没有自增长行为，因此，**字符串枚举的每个成员必须有初始值**

### any 类型

any: 任意的。当类型设置为 any 时，就取消了类型的限制。 例如：

```typescript
let obj: any = { x: 0 }

obj.bar = 100
obj()
// obj 可以是任意类型
const n: number = obj
```

#### 使用any的场景

- 函数就是不挑类型。 例如，`console.log()` ； 定义一个函数，输入任意类型的数据，返回该数据类型
- **临时使用** any 来“避免”书写很长、很复杂的类型

还有一种隐式 any，有下面两种情况会触发

1. 声明变量不提供类型也不提供默认值
2. 定义函数时，参数不给类型