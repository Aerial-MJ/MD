# JS中的this

## 普通函数

它是**函数运行**时，在**函数体内部自动生成的一个对象**，只能在函数体内部使用。

> ```javascript
> function test() {
> 　this.x = 1;
> }
> ```

上面代码中，函数`test`运行时，内部会自动有一个`this`对象可以使用。

总的来说，`this`就是**函数运行时所在的环境对象**。下面分四种情况，详细讨论`this`的用法。

**情况一：纯粹的函数调用**

这是函数的最通常用法，属于全局性调用，因此`this`就代表全局对象。请看下面这段代码，它的运行结果是1。

> ```javascript
> var x = 1;
> function test() {
>    console.log(this.x);
> }
> test();  // 1
> ```

**情况二：作为对象方法的调用**

函数还可以作为某个对象的方法调用，这时`this`就指这个上级对象。

> ```javascript
> function test() {
>   console.log(this.x);
> }
> 
> var obj = {};
> obj.x = 1;
> obj.m = test;
> 
> obj.m(); // 1
> ```

**情况三 作为构造函数调用**

所谓构造函数，就是通过这个函数，可以生成一个新对象。这时，`this`就指这个新对象。

> ```javascript
> function test() {
> 　this.x = 1;
> }
> 
> var obj = new test();
> obj.x // 1
> ```

运行结果为1。为了表明这时this不是全局对象，我们对代码做一些改变：

> ```javascript
> var x = 2;
> function test() {
>   this.x = 1;
> }
> 
> var obj = new test();
> x  // 2
> ```

运行结果为2，表明全局变量`x`的值根本没变。

**情况四 apply 调用**

`apply()`是函数的一个方法，作用是改变函数的调用对象。它的第一个参数就表示改变后的调用这个函数的对象。因此，这时`this`指的就是这第一个参数。

> ```javascript
> var x = 0;
> function test() {
> 　console.log(this.x);
> }
> 
> var obj = {};
> obj.x = 1;
> obj.m = test;
> obj.m.apply() // 0
> ```

`apply()`的参数为空时，默认调用全局对象。因此，这时的运行结果为`0`，证明`this`指的是全局对象。

如果把最后一行代码修改为

> ```javascript
> obj.m.apply(obj); //1
> ```

运行结果就变成了`1`，证明了这时`this`代表的是对象`obj`。

**bind，call，apply的作用都是用来改变this指向的**

**不是简单的继承关系**

```javascript
const func=function test() {
    console.log(this)
    func1()
}

function func1(){
    console.log(this)
}


new func()

```

```text
test {}
<ref *1> Object [global]
//运行时所在的环境变量时global
```

## 箭头函数

箭头函数的this是声明式的this

箭头函数不同于匿名函数，箭头函数本身是没有this的，在箭头函数中的this自动绑定的是所在父函数运行时所在的环境变量

而父函数所在的环境变量是由**函数运行时所在的环境对象**

## React中的Class

```js
class App extends React.Component {
 constructor(){
     this.onClickHandle=this.onClickHandle.bind(this)
     //可以理解为简单的赋值，this.onClickHandle绑定了this之后，函数中的this就是class实例
 }
    
 func1(){
     
 }
    
 render(){
     return(
     	<button onClick={function (){this.func1()}} />
		//函数function中的this运行环境是undefined的，所以只能使用箭头函数
		//就算绑定也没有效果，因为func1外层函数的this是undefined的
		//类似于在vue中写get("url").then(()=>{}).catch(()=>{}) 此处如果使用function(){//此处的this很抽象}
     )
 }
    
    
}
```

```js
app=new App()

//此时实例中所有函数的this都指向的是实例
```

## new class

```js
const func=function test() {
    console.log(this)
    const func2=function(){
        console.log(this)
        const func3=function(){
            console.log(this)
        }
        func3()
    }
    func2()
}

function func1(){
    console.log(this)
}


var obj=new func()

```

```text
//func的this会认为是在 obj运行时所在的环境对象
//其余func2和func3是在global环境对象中
```

