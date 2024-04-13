# React的css

## 行内样式 

行内样式是一种最基本的写法，也是用HTML时写的内联样式那样：

```jsx
class App extends React.Component {
  // ...
  render() {
    return (
      <div style={{ background: '#eee', width: '200px', height: '200px'}}>
        <p style= {{color:'red', fontSize:'40px'}}>行内样式</p>
      </div>
    )
  }
}

```

需要注意的是，这里的css样式名采用驼峰命名法：如`font-size →fontSize`，并且需要将`CSS属性`放在双大括号之间。

## 声明样式

声明样式其实是行内样式的一种改进写法，在render函数外部创建style对象，然后传递给组件，让css与标签分离。

```jsx
class App extends React.Component {

 const style1={    
      background:'#eee',
      width:'200px',
      height:'200px'
    }

  const style2={    
      color:'red',
      fontSize:'40px'
    }

  render() {
    return (

      <div style={style1}>
        <p style= {style2}>行内样式</p>
      </div>

    )
  }
}
```

注意这里实用的还是驼峰命名法，其次因为已经在外部声明了style对象，所以在JSX中使用的时候只需要一个大括号{}。

## 引入样式

引入样式就是将CSS文件写下外部，再引入使用。

**css 文件**

```css
.person{
    width: 60%;
    margin:16px auto;
    border: 1px solid #eee;
    box-shadow: 0 2px 3px #ccc;
    padding:16px;
    text-align: center;
}
```

**js 文件**

```jsx
import React from 'react';
import './Person.css';
class App extends React.Component {
  //....  
  render() {

    return (
      <div className='person'>
        <p>person:Hello world</p>
      </div> 
    )
  }
}

export default App;
```

因为**CSS的规则都是全局的**，任何一个组件的样式规则，都对整个页面有效，这可能会导致大量的冲突。也就是说如果有两个css文件，它们的中的一些样式名是一样的，那么就会被覆盖，简单的解决办法就是将样式的命名变得复杂且不重复

