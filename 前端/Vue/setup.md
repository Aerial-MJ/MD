## setup

**`setup`函数的执行时间取决于组件实例的创建过程。它在组件实例创建之前执行，因此可以在`setup`函数中访问组件实例之前执行的任何代码。这使得在`setup`函数中可以执行一些初始化的操作，例如设置响应式数据、订阅事件、执行异步操作等。**

在 Vue 3 中，如果你在 `<script>` 标签中的组件选项中加入了 `setup()` 函数，它将会覆盖组件的其他选项（如 `data`、`methods`、`computed` 等）。`setup()` 函数成为了组件的入口点，它负责设置组件的响应式数据、方法和生命周期钩子。

通过在 `<script>` 中加入 `setup()` 函数，你可以在函数中编写组件的逻辑，并且可以利用 Vue 3 提供的响应式 API 和生命周期钩子函数。

下面是一个示例，展示了在 `<script>` 中加入 `setup()` 函数的效果：

```vue
<template>
  <div>
    <p>Count: {{ state.count }}</p>
    <button @click="increment">Increment</button>
  </div>
</template>

<script>
import { reactive } from 'vue';

export default {
  setup() {
    const state = reactive({
      count: 0
    });

    const increment = () => {
      state.count++;
    };

    return {
      state,
      increment
    };
  }
};
</script>
```

在上面的示例中，`setup()` 函数被定义在 `<script>` 标签中，并且返回一个对象，该对象包含了 `state` 和 `increment` 这两个属性，它们成为了组件实例的一部分。`state` 对象是响应式的，因此在模板中可以实现双向绑定。`increment` 方法用于增加 `state.count` 的值。

通过在 `<script>` 中使用 `setup()` 函数，你可以更直接地定义组件的行为，并且可以利用 Vue 3 提供的新的响应式 API，如 `reactive`、`ref`、`computed` 等，以及生命周期钩子函数。这种函数式的组件编写方式提供了更大的灵活性和可维护性。

## \<script setup>

在 Vue 3 中，你可以使用 `<script setup>` 语法来简化组件的编写，它将 `setup()` 函数与组件选项合并到一个 `<script>` 标签中，使得组件的逻辑更加紧凑和易于阅读。

下面是一个使用 `<script setup>` 的示例：

```vue
<template>
  <div>
    <p>Count: {{ state.count }}</p>
    <button @click="increment">Increment</button>
  </div>
</template>

<script setup>
import { reactive } from 'vue';

const state = reactive({
  count: 0
});

const increment = () => {
  state.count++;
};
</script>
```

在上面的示例中，`<script setup>` 标签中直接定义了变量 `state` 和函数 `increment`，无需显式地返回一个对象。这些变量和函数将自动成为组件实例的一部分，可以在模板中直接使用。

使用 `<script setup>` 的好处是可以减少冗余代码，不需要显式地返回一个对象，并且能够更紧凑地定义组件的逻辑。此外，`<script setup>` 还可以自动推导响应式变量和函数，无需显式使用 `reactive` 或 `ref`。

需要注意的是，使用 `<script setup>` 语法时，一些组件选项如 **`data`、`methods`、`computed` **等将会被忽略，因为它们被隐式地放在了 `setup()` 函数中。

总结起来，`<script setup>` 语法是 Vue 3 中用于简化组件编写的一种方式，将 `setup()` 函数与组件选项合并到一个 `<script>` 标签中，使得代码更加紧凑和易读。它是一种推荐的组件编写方式，特别适合简单组件和函数式组件。

## props 和 context

`setup()`函数接收两个参数：`props`和`context`。

1. `props`参数是一个包含传递给组件的属性的对象。它可以是普通的 JavaScript 对象，也可以是由 `defineProps()` 函数生成的类型定义。你可以直接在 `setup()` 函数中使用 `props` 对象来获取父组件传递的属性值，并且可以对其进行解构、访问和修改。

2. `context`参数是一个包含组件上下文信息的对象。它提供了一些实用的方法和属性，如 `attrs`、`slots` 和 `emit` 等。

   - `attrs` 是一个包含了未被声明为 props 的父组件传递的属性的对象。你可以在 `setup()` 函数中使用 `attrs` 对象来获取这些属性值。
   
   - `slots` 是一个包含了插槽内容的对象。你可以在 `setup()` 函数中使用 `slots` 对象来访问和渲染插槽内容。
   
   - `emit` 是一个用于触发父组件事件的函数。你可以在 `setup()` 函数中使用 `emit` 函数来向父组件发送自定义事件。

下面是一个示例，演示了如何在`setup()`函数中使用`props`和`context`：

```javascript
import { defineProps } from 'vue';

export default {
  props: {
    message: String
  },
  setup(props, context) {
    // 访问props属性
    console.log(props.message);

    // 访问未声明的父组件属性
    console.log(context.attrs.unDeclaredProp);

    // 渲染插槽内容
    const slotContent = context.slots.default && context.slots.default();

    // 发送自定义事件
    const handleClick = () => {
      context.emit('custom-event', 'Hello from child component');
    };

    return {
      slotContent,
      handleClick
    };
  }
};
```

在上面的示例中，通过解构`props`对象，可以直接访问父组件传递的`message`属性。通过`context.attrs`可以访问父组件传递的未声明的属性。`context.slots.default()`用于渲染默认插槽内容。`context.emit`用于发送自定义事件。

`props`参数是一个包含传递给组件的属性的对象，这些属性可以在`setup()`函数中直接使用。可以通过解构来访问单个属性，或者使用`toRefs()`函数将所有属性转化为响应式对象。

`context`参数是一个包含组件上下文信息的对象，它提供了一些实用的方法和属性，如`attrs`、`slots`和`emit`等。通过解构或直接访问`context`对象的属性，可以在`setup()`函数中使用这些功能。

