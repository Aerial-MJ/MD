## 创建rn项目

如果你之前全局安装过旧的react-native-cli命令行工具，请使用 npm uninstall -g react-native-cli 卸载掉它以避免一些冲突

使用 React Native 内建的命令行工具来创建一个名为“AwesomeProject"的新项目。这个命令行工具不需要安装，可以直接用 node 自带的npx命令来使用：

**输入创建项目命令**

```shell
npx react-native init AwesomeProject
```

**或者使用react-native-cli 脚手架**

**cmd打开dos命令窗口，开始创建RN项目，如下命令：**

```shell
react-native init AwesomeProject
```

## 创建react项目

```shell
npm install -g create-react-app       //全局安装
create-react-app 项目名称              //新建并对react项目进行命名（注:项目名称不能有大写
cd 项目名称                            //通过命令进入文件夹内部，准备运行项目
npm start                             //运行项目
```

**利用npx命令**

```shell
npx create-react-app react-app
```

## 创建Vue项目

```shell
npm install -g @vue/cli //这个是从国外下载的比较慢`
cnpm install -g @vue/cli //这个是从镜像源下载

vue create vue-project
```

## WebPack

Vue-cli内置了WebPack，当运行npm build的时候，会使用webpack工具进行打包

Webpack 是一个前端资源加载/打包工具。它将根据模块的依赖关系进行静态分析，然后将这些模块按照指定的规则生成对应的静态资源。

### 安装 Webpack

在安装 Webpack 前，你本地环境需要支持 node.js

由于 npm 安装速度慢，本教程使用了淘宝的镜像及其命令 cnpm

使用 cnpm 安装 webpack：

`cnpm install webpack -g`

### 创建项目

接下来我们创建一个目录 app：

`mkdir app`

在 app 目录下添加 runoob1.js 文件，代码如下：

**app/runoob1.js 文件**

`document.write("It works.");`

在 app 目录下添加 index.html 文件，代码如下：

**app/index.html 文件**
```html		
<html>
    <head>
        <meta charset="utf-8">
    </head>
    <body>
        <script type="text/javascript" src="bundle.js" charset="utf-8"></script>
    </body>
</html>
```
接下来我们使用 webpack 命令来打包：

`webpack runoob1.js bundle.js`

## Babel

Babel官网对Babel的定义就是：

> Babel 是一个 JavaScript 编译器。

用通俗的话解释就是它主要用于将高版本的JavaScript代码转为向后兼容的JS代码，从而能让我们的代码运行在更低版本的浏览器或者其他的环境中。

比如我们在代码中使用了ES6箭头函数：

```js
var fn = (num) => num + 2;
```

但我们如果用IE11浏览器（鬼知道用户会用什么浏览器来看）运行的话会出现报错；但是经过Babel编译之后的代码就可以运行在IE11以及更低版本的浏览器中了：

```js
var fn = function fn(num) {
  return num + 2;
}
```

Babel就是做了这样的编译转换工作，来让我们不用考虑浏览器的兼容性问题，只要专心于代码的编写工作。

## node消除缓存

`npm cache clean --force`