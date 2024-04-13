# 创建Node项目

## require引入

创建的node项目（`npm init`）默认是commonJS规范的，也就是说使用的是require进行模块化引入的。

想在 Node.js 环境中使用 CommonJS 规范，你无需对 `package.json` 文件进行修改。只需按照以下步骤初始化你的 Node.js 项目：

1. 创建项目目录并进入该目录，例如：

   ```shell
   mkdir myproject
   cd myproject
   ```

2. 在该目录中初始化 Node.js 项目，可以使用以下命令：

   ```shell
   npm init
   ```

   执行此命令后，会提示你输入一些项目相关的信息，例如项目名称、描述、版本、作者等等。最后，它将在该目录中创建一个 `package.json` 文件，其中包含你输入的所有信息。

3. 在你的项目代码中使用 CommonJS 规范，例如：

   ```js
   express = require('express');
   const app = express();
   ```

   在这个例子中，我们使用 `require` 函数加载 `express` 模块，并将它赋值给一个变量 `express`。这里的 `require` 函数就是 CommonJS 规范中定义的模块加载函数。

总之，对于 Node.js 项目来说，CommonJS 规范是默认的模块系统，你只需要按照上述步骤初始化你的项目，并使用 `require` 函数加载模块即可。

下面以server.js和index.js为例子

```js
const axios= require("axios")

let runPythonScript=(text1)=>{
  return new Promise((resolve,reject)=>{
    axios.get('http://localhost:8080')
      .then((response) => {
          //console.log(response.data)
          resolve(response)
      })
      .catch((error) => {
          return "发生错误，请稍后再试"
      });
  })
}


runPythonScript('').then((res)=>{
  console.log(res.data)
  console.log(res.data)
})
```

```js
const { axios } = require("axios");
const express = require("express");

const app=express();

//创建一个异步函数
const func= async ()=>{
    //直接在promise前面加上await
    let ans=await new Promise((resolve,reject)=>{
        //异步函数  
         setTimeout(()=>{
            resolve("1");
        },2000)
    })
//只有拿到resolve才能继续向后执行，
//当拿到return ans的时候才能执行then的方法
    return ans
}


app.get('/',(request,response)=>{
    func().then((res)=>{
        response.send(res)
    }).catch((err)=>{
        console.log(err);
    })
   
})

//监控8080端口
app.listen(8080, () => {
    console.log('Server listening on port 8080');
});
```

## import引入

在 package.json 文件中指定 "type":"module" 后，这个项目就可以按照 ES Module 规范

## 混合使用

### 一、指定使用 CommonJS 模块

1.在 node 项目里，如果不在 package.json 文件中指定 type，那这个项目默认就是 CommonJS 规范的
2.一个 .js 结尾的文件，默认是 CommonJS 规范，也可以强制的指定文件后缀为 .cjs（一般在 ES Module项目中如果希望一个文件是 CommonJS 规范的，可以这样指定后缀名）

所以可以在package.json文件指定了type是module的情况下，将是commonJS模块的js文件强制指定文件后缀为 .cjs，即可使用


### 二、指定使用 ES Module 模块

1.在 package.json 文件中指定 "type":"module" 后，这个项目就可以按照 ES Module 规范来写了
2.指定文件的后缀名为 .mjs，那么这个文件会被强制指定使用 ES Module 规范
