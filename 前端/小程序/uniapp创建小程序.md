## uniapp和Hbuilder

`uni-app` 是一个使用 Vue.js 开发所有前端应用的框架，开发者编写一套代码，可发布到iOS、Android、Web（响应式）、以及各种小程序（微信/支付宝/百度/头条/飞书/QQ/快手/钉钉/淘宝）、快应用等多个平台。

`hbuilder`是Dcloud公司可以编写Vue.js的**IDE**，可以直接编译成适应各种平台的语言。

现在`uni-app`推出了npm脚本，可以不依赖与Hbuider IDE开发。

当编译成微信小程序的时候，可以直接使用微信小程序打开，并且支持热部署

## 创建uni-app

**全局安装 vue-cli**

```shell
npm install -g @vue/cli
```
**创建uni-app**

```shell
vue create -p dcloudio/uni-preset-vue my-project

# 创建以 javascript 开发的工程(vue3)
npx degit dcloudio/uni-preset-vue#vite my-vue3-project 

# 创建以 typescript 开发的工程(vue3)
npx degit dcloudio/uni-preset-vue#vite-ts my-vue3-project
```
##  运行、发布uni-app

```shell
npm run dev:%PLATFORM%
npm run build:%PLATFORM%
```

`%PLATFORM%`
`mp-weixin`	=>微信小程序

