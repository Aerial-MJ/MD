# React Hook

区别于Vue的`ref`和`reactive`，react的hooks进行响应式处理相对较为复杂。

## useState

useState 是 React Hooks 中最常用的一个，它用于在函数组件中添加状态。useState 函数接受一个初始状态值，并返回一个数组，数组的第一个元素是当前状态的值，第二个元素是用于更新状态值的函数。例如：

```react
import React, { useState } from 'react';

function Example() {
  // 定义一个名为 count 的状态变量，初始值为 0
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}
```

在上面的代码中，我们使用 useState 定义了一个名为 count 的状态变量，并将其初始值设为 0。然后在组件中使用 count 的值，并且使用 setCount 函数来更新 count 的值。

### 为什么多次调用setCount函数，而count的值只改变一次

在 React 中，当调用 `setState` 或者 `useState` 的更新函数时，React 并不会立即更新状态变量的值。相反，React 会将新的状态值加入一个更新队列，然后在下一次渲染时批量更新所有状态的值。

因此，当你在多次调用 `setCount` 函数时，React 会将每次调用的值加入更新队列，但是只会在下一次渲染时将更新后的值应用到状态变量上。这就是为什么你看到 `count` 的值只改变一次的原因。

需要注意的是，React 在合并更新队列时，会对多次更新同一个状态变量的值进行合并，以减少不必要的渲染次数。因此，如果你在一个函数组件中多次调用 `setCount` 函数，但是每次调用的值都相同（不相同也只渲染最后一次），那么实际上只会进行一次渲染。

例如，下面的代码中，我们多次调用 `setCount` 函数，但是每次调用都传入相同的值（不相同也只渲染最后一次），因此实际上只会进行一次渲染：

```react
import React, {useEffect, useRef, useState} from "react";

export default function App() {
    const [count, setCount] = useState(0);
    const [flag, setFlag] = useState(false);

    useEffect(() => {
        console.log("hello")
    }, [count])

    function handleClick() {
        setCount(1)
        console.log(count)
        setCount(2)
        console.log(count)
        setCount(3)
        console.log(count)
    }

    console.log('render')
    return (
        <div>
            <button onClick={handleClick}>Next</button>
            <h1 style={{color: flag ? "blue" : "black"}}>{count}</h1>
        </div>
    );
}

```

总之，React 的状态更新机制是基于队列的，它会在下一次渲染时批量更新所有状态变量的值，以提高性能。

当你在 React 事件处理函数中调用 `setState` 或者 `useState` 的更新函数时，React 会对这些更新进行批量处理。也就是说，React 会将这些更新放入一个更新队列中，然后在事件处理函数执行结束之后，才会对队列中的更新进行合并和批量处理。这样做是为了优化性能，减少不必要的 DOM 操作和渲染次数。

除了事件处理函数之外，在 React 生命周期方法和钩子函数中，React 也会对状态更新进行批量处理。具体来说，在 `componentDidUpdate`、`getDerivedStateFromProps`、`getSnapshotBeforeUpdate` 这些生命周期方法中，React 会对所有状态更新进行合并和批量处理，以减少不必要的渲染次数。

需要注意的是，在 React 的原生事件中（例如原生的 DOM 事件、`setTimeout`、`setInterval` 等），React 会将更新立即应用到状态变量上，而不会对更新进行合并和批量处理。这是因为这些事件是由浏览器直接触发的，React 没有办法对它们进行控制。如果你在 React 事件处理函数中使用了原生的 DOM 事件或者定时器，那么这些事件也会立即应用到状态变量上，而不会进行批量处理。

总之，React 会在事件处理函数执行结束之后，对所有状态更新进行批量处理，以优化性能。在 React 的生命周期方法和钩子函数中，React 也会对所有状态更新进行批量处理。但是在原生事件中，React 会立即应用状态更新，而不会进行批量处理。

### 怎么实现同步操作

在 React 中，当你调用 `setState` 或者 `useState` 的更新函数时，React 会将更新加入更新队列，并在下一次渲染时批量处理所有更新。这种更新方式被称为异步更新。

但是，如果你需要在更新后立即获取最新的状态值，那么可以使用 `useEffect` 来监听状态变量的变化，并在变化后执行一些操作。**由于 `useEffect` 是在组件渲染之后执行的，因此它可以保证在状态更新后立即执行。**

例如，下面的代码中，我们使用 `useState` 定义了一个状态变量 `count`，然后在点击按钮时多次调用 `setCount` 函数，最后使用 `useEffect` 在 `count` 变化后打印最新的 `count` 值：

```react
function Example() {
  const [count, setCount] = useState(0);

  function handleClick() {
    setCount(count + 1);
    setCount(count + 1);
    setCount(count + 1);
  }

  useEffect(() => {
    console.log("Count:", count);
  }, [count]);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={handleClick}>Increment</button>
    </div>
  );
}
```

在上面的代码中，当我们点击按钮时，`count` 的值会增加 1，但是由于 `setCount` 的更新是异步的，因此在点击按钮后，`useEffect` 中打印的 `count` 值仍然是更新前的值。如果你需要立即获取更新后的 `count` 值，可以使用函数形式的 `setCount`：

```react
function handleClick() {
  setCount((prevCount) => prevCount + 1);
  setCount((prevCount) => prevCount + 1);
  setCount((prevCount) => prevCount + 1);
}
```

在上面的代码中，我们使用函数形式的 `setCount`，并在回调函数中使用上一个状态的值来更新状态变量。由于函数形式的 `setCount` 是同步更新的，因此在点击按钮后，`useEffect` 中打印的 `count` 值会是更新后的值。

**函数形式的setCount是同步更新的，虽然会损耗性能，但是却保证了同步更新**

## useEffect

1. 什么都不传，组件每次 render 之后 useEffect 都会调用，相当于 componentDidMount 和 componentDidUpdate
2. 传入一个空数组 [], 只会调用一次，相当于 componentDidMount 和 componentWillUnmount 
3. 传入一个数组，其中包括变量，只有这些变量变动时，useEffect 才会执行

**`useEffect` 是在组件渲染之后执行的，因此它可以保证在状态更新后立即执行**

## useRef

1. 当你更新 `useRef` 的 `current` 属性时，React 不会重新渲染组件
2. 获取 DOM 节点的引用

在 React 中，`useRef` 是一个 Hook，用于在函数组件中存储和访问可变值。使用 `useRef` 可以创建一个 ref 对象，该对象包含一个 `current` 属性，该属性可以存储任意值。与 `useState` 不同，当你更新 `useRef` 的 `current` 属性时，React 不会重新渲染组件。因此，`useRef` 可以用来存储一些不需要触发组件重新渲染的数据，例如定时器的 ID、DOM 节点的引用等。

除了存储数据外，`useRef` 还可以用来获取 DOM 节点的引用。例如，下面的代码中，我们使用 `useRef` 创建一个 ref 对象，然后将它赋值给 `<input>` 元素的 `ref` 属性。这样，我们就可以在组件中通过 `inputRef.current` 来访问 `<input>` 元素的属性和方法：

```react
function Example() {
  const inputRef = useRef(null);

  function handleClick() {
    inputRef.current.focus();
  }

  return (
    <div>
      <input type="text" ref={inputRef} />
      <button onClick={handleClick}>Focus</button>
    </div>
  );
}
```

在上面的代码中，当我们点击按钮时，`handleClick` 函数会使用 `inputRef.current.focus()` 来将焦点设置到 `<input>` 元素上。由于 `inputRef` 是通过 `useRef` 创建的，因此它的 `current` 属性始终包含最新的 `<input>` 元素的引用。

除了使用 `useRef` 获取 DOM 节点的引用外，`useRef` 还可以用来存储组件的某些状态或变量，以供其他组件或 Hook 使用。例如，下面的代码中，我们使用 `useRef` 存储一个定时器的 ID，并在组件卸载时清除定时器：

```react
function Example() {
  const timerRef = useRef(null);

  useEffect(() => {
    timerRef.current = setInterval(() => {
      console.log("Tick");
    }, 1000);

    return () => {
      clearInterval(timerRef.current);
    };
  }, []);

  return <div>Example</div>;
}
```

在上面的代码中，我们使用 `useRef` 创建了一个 `timerRef` 对象，并在组件挂载时创建了一个定时器，并将定时器的 ID 存储在 `timerRef.current` 中。在组件卸载时，我们通过 `clearInterval(timerRef.current)` 来清除定时器。

总之，`useRef` 是一个非常实用的 Hook，可以用来存储和访问可变值，以及获取 DOM 节点的引用。

## useContext

**主要是为了实现父子组件传递**

### 父组件传递给子组件

在 React 中，`useContext` 是一个 Hook，用于在函数组件中访问 React Context。React Context 可以帮助我们在组件树中传递数据，而无需一级一级地手动将 props 传递下去。使用 `useContext` 可以方便地访问 Context 中存储的数据。

使用 `useContext` 需要两个步骤：

1. 创建一个 Context 对象，并将数据存储在 Context 中。例如：

   ```react
   const MyContext = React.createContext(defaultValue);
   ```

   其中，`defaultValue` 是当组件没有匹配到任何 Provider 时，Context 中所存储的默认值。

2. 在组件中使用 `useContext` 来访问 Context 中的数据。例如：

   ```react
   const myData = useContext(MyContext);
   ```

   `useContext` 接收一个 Context 对象作为参数，并返回 Context 中存储的数据。

下面是一个简单的例子，展示了如何使用 `useContext` 在组件树中传递数据：

```react
// 创建一个 Context 对象
const ThemeContext = React.createContext("light");

// 使用 Context.Provider 在组件树中提供数据
function App() {
  return (
    <ThemeContext.Provider value="dark">
      <Header />
      <Main />
      <Footer />
    </ThemeContext.Provider>
  );
}

// 在组件中使用 useContext 访问 Context 中的数据
function Header() {
  const theme = useContext(ThemeContext);

  return (
    <header style={{ background: theme === "dark" ? "#333" : "#eee" }}>
      <h1>Header</h1>
    </header>
  );
}

function Main() {
  const theme = useContext(ThemeContext);

  return (
    <main style={{ color: theme === "dark" ? "#fff" : "#000" }}>
      <h2>Main</h2>
      <p>This is the main content.</p>
    </main>
  );
}

function Footer() {
  const theme = useContext(ThemeContext);

  return (
    <footer style={{ background: theme === "dark" ? "#333" : "#eee" }}>
      <h3>Footer</h3>
    </footer>
  );
}
```

在上面的代码中，我们首先使用 `createContext` 创建了一个名为 `ThemeContext` 的 Context 对象。然后，在 `App` 组件中，我们使用 `ThemeContext.Provider` 提供了一个名为 `dark` 的值，使得 `Header`、`Main` 和 `Footer` 组件中可以通过 `useContext` 访问到这个值。最后，我们在三个组件中分别使用 `useContext(ThemeContext)` 来获取当前的主题，从而控制组件的样式。

总之，`useContext` 是一个方便的 Hook，可以帮助我们在函数组件中访问 Context 中存储的数据。

### 实现子组件传递给父组件

在 React 中，子组件传递数据给父组件一般需要通过 props 来实现。但是，如果你的应用程序采用了 Context 来存储一些全局状态或者配置信息，那么使用 `useContext` 就可以在子组件中获取 Context 中的数据，并将这些数据传递给父组件。

具体来说，可以通过以下步骤实现子组件传递数据给父组件：

1. 在父组件中创建 Context 对象，并在 Context 中存储一个回调函数，该回调函数可以接受子组件传递过来的数据，并将这些数据传递给父组件的状态中。例如：

   ```js
   const MyContext = React.createContext({
     sendData: () => {}
   });
   
   function Parent() {
     const [data, setData] = useState([]);
   
     function handleReceiveData(newData) {
       setData(prevData => [...prevData, newData]);
     }
   
     return (
       <MyContext.Provider value={{ sendData: handleReceiveData }}>
         <Child />
         <Child />
         <Child />
       </MyContext.Provider>
     );
   }
   ```

   在上面的代码中，我们创建了一个名为 `MyContext` 的 Context 对象，并在 Context 中存储了一个名为 `sendData` 的回调函数。在 `Parent` 组件中，我们使用 `useState` 创建了一个名为 `data` 的状态，该状态用于存储从子组件传递过来的数据。我们还创建了一个名为 `handleReceiveData` 的回调函数，该函数接受子组件传递过来的数据，并将这些数据添加到 `data` 状态中。

2. 在子组件中使用 `useContext` 获取 Context 中的回调函数，并在需要的时候调用该函数，将子组件的数据传递给父组件。例如：

   ```js
   function Child() {
     const { sendData } = useContext(MyContext);
   
     function handleClick() {
       const newData = { id: Date.now(), name: "New Data" };
       sendData(newData);
     }
   
     return <button onClick={handleClick}>Send Data</button>;
   }
   ```

   在上面的代码中，我们使用 `useContext` 获取了 `MyContext` 中的 `sendData` 回调函数，并在按钮的点击事件处理程序中调用了该函数，并将一个名为 `newData` 的新数据传递给父组件。

这样，当子组件点击按钮时，它就可以将数据传递给父组件了。在 `Parent` 组件中，我们定义了 `handleReceiveData` 回调函数，该函数会将子组件传递过来的数据添加到 `data` 状态中。由于状态的更新会触发组件重新渲染，因此父组件会重新渲染，并将更新后的数据传递给所有的子组件。
