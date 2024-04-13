# react
## react中的JSX对象

推荐使用小括号包裹JSX，避免JS自动插入分号的陷阱

我们return的都是**JSX(Javascript XML)对象**

```js
//类组件，有this（代表的是组件实例）
 render(){
        return(
            <button onClick={this.handleClick}>点我点我</button>
        )
    }
    
//函数组件，无this
      return(
        <button onClick={handleClick}>点我</button>
    )
```

### 在JSX使用JavaScript表达式

可以在单大括号{}中使用任意合格的JavaScript合格的表达式

JSX自身也是js表达式 `{js表达式}` 

`（JSX表达式）` 所以（JSX表达式）可以放到{}中

`{(JSX表达式)}`

**定义JSX表达式，在JSX表达式中，使用的是驼峰命名法**

```js
const title1 = (
    <h1 className={"title"}>hello JSX
        <div>
            {
            3>5?'大于':'小于等于'
        }</div>
 
        <div>
            {
            sayHi()
        }</div>
    </h1>
)
```

**条件渲染，返回三元表达式（js表达式）**

```js
const loadData=()=>{
    return isLoading ? (<div>loading......</div>) : (<div>数据已经加载完毕</div>)
}
```
## React父子传值

### 父传子
**父组件**

```react
class Welcome extends React.component{
    render(){
        return(
            <h1>{this.props.name}</h1>
        )
    }
}
```
**子组件**
```react
<Welcome name="111" />
```

### 子传父

响应函数定义在父组件之中，Vue与React区别在于，vue是向父组件传值，然后再在父组件中执行。

而react是将函数拷贝到子组件中，再在子组件中直接执行父组件的方法

**父组件**

```react
<Child onChange={this.handleFatherChange} />
```
**子组件**

```react
class Child extends React.component{
    
    handleClick(e){
        this.props.onChange(e.target.value)
    }
    
    render(){
        return(
            <input onclick={this.handleClick}></input>
        )
    }
    
}
```

### 父组件调用子组件的函数

**可以使用useRef hooks 进行对组件的引用**