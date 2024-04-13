# Vue

## 传值

`ref `跟` reactive` 都是响应系统的核心方法，作为整个系统的入口

可以将 ref 看成 reactive 的一个变形版本，这是由于 reactive 内部采用 Proxy 来实现，而 Proxy 只接受对象作为入参，这才有了 ref 来解决值类型的数据响应，如果传入 ref 的是一个对象，内部也会调用 reactive 方法进行深层响应转换
### 父组件向子组件传值
使用`defineProps`
**子组件**

```vue
<script>
import { defineComponent } from "vue"
export default defineComponent({
  name: "children",
  props: {
    title: {
      type: String,
    },
  },
  setup(props) {
    return {
    }
  })
</script>

```
**父组件**

```vue
<template>
  <child :title="fatherTitle"/>
</template>

<script lang="ts">
import {defineComponent, ref} from "vue";

export default defineComponent({
  
  setup(){
    let fatherTitle=ref("hello")
    return {
      fatherTitle
    }
  }
})
</script>

```

### 子组件向父组件传值

使用emit方式`defineEmits`

**子组件**

```vue
<script>
import { defineComponent } from "vue"
export default defineComponent({
  name: "children",
  props: {
    title: {
      type: String,
    },
  },
  setup(props, context) {
    //props接受的是上面props定义的属性，context接收的是除了props属性的其他自定义属性
    const sayToFather = () => {
      const ask = "我是子组件，我对父组件说话"
      context.emit("listen", ask)
        //向父组件传值
    }
    return {
      sayToFather,
    }
  })
</script>

```

响应函数定义在父组件之中，与react区别在于，vue是向父组件传值，然后再在父组件中执行。

而react是将函数拷贝到子组件中，再在子组件中直接执行父组件的方法

**父组件**

```vue
<template>
	<children :title="msg" @listen="listenToChildren"></children>
</template>

<script>
export default defineComponent({
    
    setup() {
    	let msg = "我是父组件"
    	let listenToChildren = (val) => {
      		children_msg.value = val // 使用ref包裹的数据，需要通过.value的形式访问他的值
    	}
    	return {
      		msg,
      		listenToChildren,
    	}
    }
})
</script>
```


## vue中的ref()和reactive()
`ref` :表层对象还是使用的是`defineProperty`的getter或者setter实现响应式，深层对象使用的是Proxy实现响应式。

`ref`获取DOM
```react
<template>
	<div ref="eleDom">ref-dom-test</div>
</template>

<script>
import { ref, onMounted } from 'vue'
export default {
	setup() {
		const eleDom = ref(null)
		onMounted(() => {
			console.log(eleDom.value.innerHTML) // ref-dom-test
		})
		return {
			eleDom 
		}
	},
}
```

`reactive` :reactive内部使用的是Proxy实现的。这是由于 reactive 内部采用 Proxy 来实现，而 Proxy 只接受对象作为入参，这才有了 ref 来解决值类型的数据响应。

`toRef`:针对一个响应式对象的一个prop，创建一个ref，具有响应式，两者保持引用关系

```js
setup() {
		const state = reactive({
			name: 'JL',
			age: 18
		})
		const ageRef = toRef(state, 'age')
		setTimeout(() => {
			state.age = 20
		}, 1000)
		
		setTimeout(() => {
			ageRef.value = 21
		}, 2000)
		
		return {
			state,
			ageRef
		}
	}
```

`toRefs`:将一个响应式对象转为普通对象，对象的每一个属性都是响应式的，且都是对应的ref,两者保持引用关系

```js
<template>
	<div>{{ name }}---{{ age }}</div>
</template>

<script>
import { reactive, toRefs } from 'vue'
export default {
	setup() {
		const state = reactive({
			name: 'JL',
			age: 18
		})

		const stateRefs = toRefs(state)

		setTimeout(() => {
			state.age = 20
		}, 1000)

		setTimeout(() => {
			stateRefs.age.value = 21
		}, 2000)

		return {...stateRefs}
        //可以写成 return stateRefs 自动解构
	},
}
</script>
```


同时，**当需要将响应类型的对象解包时**，就需要使用`toRefs()`函数。

`toRefs()`函数**将响应式对象解包成为一个普通对象，但其中的属性都变为响应式属性。**

**reactive对象本身是响应式对象，但是其中的属性是普通属性。**

```js
//响应式对象
let state = reactive({
    name: 'zly',
    age: 47
})

let state2 = toRefs(state)

console.log(state)//是一个响应式对象，但是其中的值是普通属性
console.log(state2)//是一个普通对象，但是其中的属性为响应式数据


return{
	...state2, //需要解构
	state
}

```

## Vue 3中数组响应式用ref还是reactive

在`vue3`中，定义响应式数据一般有两种方式：`ref` 和 `reactive`

一般来说，我们使用 ref 来定义基本数据类型，使用 `reactive` 来定义复杂数据类型

**数组响应式可以使用ref也可以使用reactive**

**在 Vue 3 中，`ref` 函数用于创建一个响应式的数据引用。**

对于基本类型数据（如字符串、数字、布尔值等），`ref` 会返回一个包装过的对象，该对象具有 `value` 属性用于获取和修改数据。这里使用的是 `defineProperty` 来实现响应式。

对于对象类型的数据，`ref` 会对对象进行浅层包装，并使用 `Proxy` 来实现响应式。这意味着在访问和修改深层对象的属性时，会通过 `Proxy` 拦截器来实现响应式更新。**(ref.value->proxy)**
