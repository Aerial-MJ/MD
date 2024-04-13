## package.json文件

**我不使用--save，这个库就不会声明到dependencies字段中吗**

是的，如果你在 `npm install` 命令中没有使用 `--save` 或 `--save-dev` 参数，则安装的库将不会自动添加到 `package.json` 文件的 `dependencies` 或 `devDependencies` 字段中。

**错误的，现在的node项目就算你不写--save，会默认将字段添加到package.json的dependencies中**

虽然这并不会影响你的项目依赖项的正常运行，但是如果你希望在以后重新构建或分享项目时保留所有依赖项信息，最好使用 `--save` 或 `--save-dev` 参数。这样，所有安装的库都将被添加到 `package.json` 文件中，方便其他人或部署到其他计算机时使用。

###  --save或 --save-dev参数区别

`npm install` 命令用于安装 Node.js 项目所需的依赖项。它可以通过两种方式安装依赖项：全局安装（参数加上`-g`）和本地安装。其中，本地安装是最常用的方式，因为它会将依赖项安装到项目目录中，并将它们列入 `package.json` 文件中。

当你使用 `npm install` 命令安装依赖项时，你可以使用 `--save` 或 `--save-dev` 参数将依赖项添加到 `package.json` 文件中。

- `--save` 参数会将依赖项添加到 `dependencies` 字段中。这意味着该依赖项是项目的必需项，需要在生产环境中使用。例如：

  ```
  npm install express --save
  ```

  上面的命令会将 `express` 库安装到项目中，并将其添加到 `dependencies` 字段中。

- `--save-dev` 参数会将依赖项添加到 `devDependencies` 字段中。这意味着该依赖项只是开发过程中的工具或辅助库，不需要在生产环境中使用。例如：

  ```
  npm install nodemon --save-dev
  ```

  上面的命令会将 `nodemon` 库安装到项目中，并将其添加到 `devDependencies` 字段中。

总的来说，使用 `--save` 或 `--save-dev` 参数的主要好处是，它们可以让您在以后轻松地重建您的项目依赖项，而无需手动重新安装它们。当您需要将项目传递给其他人或部署到另一个计算机时，这非常有用。只需在新的计算机上运行 `npm install` 命令，npm 将自动下载和安装 `package.json` 中指定的所有依赖项。
