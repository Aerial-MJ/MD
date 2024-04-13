# Vue2与Vue3

vue2中的响应式核心是es5的Object.defineProperty，缺点有：

1. 深度监听需要递归到底，性能层面考虑不太好
2. 无法监听对象新增属性和删除属性，需要vue特殊处理
3. 无法监听原生数组，需要在数组原型上做拦截
4. 所以vue3采用了es6之后的proxy去重构了响应式原理，proxy能够很好的解决Object.defineProperty的缺点。

ref表层使用的是Object.defineProperty，深层使用的Proxy实现响应式

reactive使用的是Proxy实现响应式

## 选项式api

选项式 API 也是 Vue 3 的一部分。Vue 3 在设计时考虑到了现有 Vue 2 项目的平滑迁移和向后兼容性，因此保留了选项式 API 的支持。

选项式 API 是 Vue 2 中常用的 API 风格，它通过在 Vue 组件选项中定义数据、计算属性、方法等来组织和管理组件的逻辑。Vue 3 仍然支持这种方式，所以如果你已经熟悉 Vue 2 的选项式 API，你可以继续在 Vue 3 中使用它来编写组件。

然而，Vue 3 还引入了组合式 API，它提供了更灵活和可组合的方式来编写组件逻辑。组合式 API 通过 `setup` 函数来组织组件的逻辑，并使用响应式函数（如 `ref`、`reactive`）来创建响应式数据。组合式 API 可以更好地支持 TypeScript 类型推断，并且更适合于复杂的组件逻辑和代码复用。

无论是选项式 API 还是组合式 API，它们都属于 Vue 3，并且可以在 Vue 3 项目中使用。你可以根据自己的偏好和项目需求选择使用哪种 API 风格来编写组件。
