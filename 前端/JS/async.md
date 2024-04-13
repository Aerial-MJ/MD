# JS异步操作

## Promise
Promise 对象代表了未来将要发生的事件，用来传递异步操作的消息。
对象的状态不受外界影响。Promise `对象代表一个异步操作`，有三种状态：

1. pending: 初始状态，不是成功或失败状态。
2. fulfilled: 意味着操作成功完成。
3. rejected: 意味着操作失败。

### Promise 创建
要想创建一个 promise 对象、可以使用 new 来调用 Promise 的构造器来进行实例化。

下面是创建 promise 的步骤：

```js
var promise = new Promise(function(resolve, reject) {
    // 异步处理
    // 处理结束后、调用resolve 或 reject
});
```

Promise 构造函数包含一个参数和一个带有 resolve（解析）和 reject（拒绝）两个参数的回调。在回调中执行一些操作（例如异步），如果一切都正常，则调用 resolve，否则调用 reject。

```js
var myFirstPromise = new Promise(function(resolve, reject){
    //当异步代码执行成功时，我们才会调用resolve(...), 当异步代码失败时就会调用reject(...)
    //在本例中，我们使用setTimeout(...)来模拟异步代码，实际编码时可能是XHR请求或是HTML5的一些API方法.
    setTimeout(function(){
        resolve("成功!"); //代码正常执行！
    }, 250);
});

myFirstPromise.then(function(successMessage){
    //successMessage的值是上面调用resolve(...)方法传入的值.
    //successMessage参数不一定非要是字符串类型，这里只是举个例子
    document.write("Yay! " + successMessage);
});
```

对于已经实例化过的 promise 对象可以调用 `promise.then()` 方法，传递 resolve 和 reject 方法作为回调。

`promise.then()` 是 promise 最为常用的方法。

`promise.then(onFulfilled, onRejected)`
promise简化了对error的处理，上面的代码我们也可以这样写：

`promise.then(onFulfilled).catch(onRejected)`

## Async

`async`和`await`是`promise`的语法糖

async是一个加在函数前的修饰符，被async定义的函数会默认返回一个Promise对象的resolve的值。因此对`async函数`可以直接then，返回值就是then方法传入的函数。

```js
// async基础语法
async function fun0(){
    console.log(1);
    return 1;
}
fun0().then(val=>{
    console.log(val) // 1,1
})

async function fun1(){
    console.log('Promise');
    return new Promise(function(resolve,reject){
        resolve('Promise')
    })
}
fun1().then(val => {
    console.log(val); // Promise Promise
})
```

## Await
await 也是一个修饰符，**只能放在async定义的函数内**。可以**理解为等待**。

await 修饰的如果是Promise对象：可以获取Promise中返回的内容（resolve或reject的参数），且取到值后语句才会往下执行；**同步**

如果不是Promise对象：把这个非promise的东西当做await表达式的结果。

```js
async function fun(){
    let a = await 1;
    let b = await new Promise((resolve,reject)=>{
        setTimeout(function(){
            resolve('setTimeout')
        },3000)
    })
    let c = await function(){
        return 'function'
    }()
    console.log(a,b,c)
}
fun(); // 3秒后输出： 1 "setTimeout" "function"
```

```js
function log(time){
    setTimeout(function(){
        console.log(time);     
    },time)
    console.log(5555)
	return 1;
}
async function fun(){
    let a = await log(1000);
    let b = await new Promise((resolve,reject)=>{
        setTimeout(function(){
            resolve('setTimeout')
        },3000)
    })
    let c = log(5000);
    console.log(a);
    console.log(1)
}
fun(); 
// 1s输出 1000
// 再过2秒后输出 undefined//此时的Promise对象执行完毕
// 需要让Promise对象执行完毕才可以继续进行异步操作
// 再过5秒后输出 5000
```

## JS异步

```js
(function(){

    var promise = new Promise(function(resolve, reject) {
        setTimeout(function(){
            console.log("async4")
            resolve("成功")
        },2000);
        console.log("async3")
    })

    promise.then(()=>{
        console.log("sync2")
    })
    console.log("sync3")

})()
```

```text
async3
sync3
sync2
async4
```

```js

async function fun(){
    let a = await 1;
    let b = await new Promise((resolve,reject)=>{
        setTimeout(function(){
            resolve('setTimeout')
        },3000)
        
    })
    let c = await function(){
        return 'function'
    }()
    console.log(a)
    console.log(b)
    console.log(c)
}
fun(); 

console.log("hello")
// 立即输出hello
// 3秒后输出： 1 "setTimeout" "function"
```

## async不是也返回的promise，为什么await不等待呢

```js
async function hello(){
    return ()=>{
        setTimeout(function(){
            console.log("hhhh")
        },3000)
    }
}

async function fun(){

    let m=hello()

    console.log(m)

    let mx=new Promise((resolve,reject)=>{
        setTimeout(function(){
            resolve('setTimeout')
        },3000)})
    
    console.log(mx)

}
fun(); 

```

```shell
> Promise { [Function (anonymous)] }

> Promise { <pending> }

#再次等待三秒，程序才会停止

```

为什么await会将第二个Promise同步，而不会将第一个Promis同步

为什么函数会等三秒才会停止，注意这里并没有同步，因为程序先输出，再等待的三秒，这里的原因仅仅是要等`setTimeout`线程结束才能让整个程序停止。

```js

async function hello(){
    return new Promise((resolve,reject)=>{
        setTimeout(function(){
            resolve('setTimeout')
        },3000)})
}

async function fun(){

    let m=hello()

    await m;
    let mx=new Promise((resolve,reject)=>{
        setTimeout(function(){
            resolve('setTimeout')
        },3000)})

    await mx
	console.log(m)
    console.log(mx)
    
}
fun(); 
//以上程序六秒结束

async function hello(){
    return new Promise((resolve,reject)=>{
        setTimeout(function(){
            resolve('setTimeout')
        },3000)})
}

async function fun(){

    let m=hello()

    
    let mx=new Promise((resolve,reject)=>{
        setTimeout(function(){
            resolve('setTimeout')
        },3000)})
	await m;
    await mx
	console.log(m)
    console.log(mx)
    
}
fun(); 

//三秒结束
```

都打印出了

```text
Promise { 'setTimeout' }
Promise { 'setTimeout' }
```

**所以为了避免歧义，可以直接将await直接写到函数的旁边即可**

```js
//promise默认会直接执行

new Promise((resolve,reject)=>{
    setTimeout(function(){
        resolve('setTimeout')
    },3000)})

```

