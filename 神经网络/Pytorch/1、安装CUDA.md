# 安装CUDA

 ## cuda安装

 windows10 版本安装 [CUDA](https://so.csdn.net/so/search?q=CUDA&spm=1001.2101.3001.7020) ，首先需要下载两个安装包

- CUDA toolkit（toolkit就是指工具包）
- cuDNN

注：cuDNN 是用于配置深度学习使用

cuDNN 其实就是 CUDA 的一个补丁而已，专为深度学习运算进行优化的。

- CUDA：为“GPU通用计算”构建的运算平台。

- cudnn：为深度学习计算设计的软件库。

- CUDA Toolkit (nvidia)： CUDA完整的工具安装包，其中提供了 Nvidia 驱动程序、开发 CUDA 程序相关的开发工具包等可供安装的选项。包括 CUDA 程序的编译器、IDE、调试器等，CUDA 程序所对应的各式库文件以及它们的头文件。（各个软件语言都可以使用）

- CUDA Toolkit (Pytorch)： CUDA不完整的工具安装包，其主要包含在使用 CUDA 相关的功能时所依赖的动态链接库。不会安装驱动程序。

- 注：CUDA Toolkit 完整和不完整的区别：在安装了CUDA Toolkit (Pytorch)后，只要系统上存在与当前的 cudatoolkit 所兼容的 Nvidia 驱动，则已经编译好的 CUDA 相关的程序就可以直接运行，不需要重新进行编译过程。如需要为 Pytorch 框架添加 CUDA 相关的拓展时（Custom C++ and CUDA Extensions），需要对编写的 CUDA 相关的程序进行编译等操作，那么就需要安装完整的 Nvidia 官方提供的 CUDA Toolkit。

- 安装的一些问题

  [【一文解决】已安装CUDA与Pytorch但torch.cuda.is_available()为False_torch.cuda.is available返回false-CSDN博客](https://blog.csdn.net/qq_41112170/article/details/131191827)

## Cuda版本的安装及下载

可以看到自己电脑支持的CUDA版本（支持的最高版本Driver Version）

```
nvidia-smi 中的CUDA 版本与 nvcc不一致
其实是因为CUDA 有两种API，分别是运行时 API 和 驱动API，即所谓的 Runtime API 与 Driver API。
nvidia-smi 的结果除了有 GPU 驱动版本型号，还有 CUDA Driver API的型号。
而nvcc的结果是对应 CUDA Runtime API。

CUDA有两种API，一个是驱动API（Driver Version），依赖NVIDIA驱动，由nvidia-smi查看；另一个是运行API（Runtime Version）是软件运行所需要的，一般驱动API版本>=运行API版本即可。
驱动API的依赖文件由GPU driver installer安装，nvidia-smi属于这一类API；
运行API的依赖文件由CUDA Toolkit installer安装。
```

安装完CUDA Toolkit之后，可以直接


## 直接安装pytorch对应的cuda

在cmd里输入 [nvidia-smi](https://so.csdn.net/so/search?q=nvidia-smi&spm=1001.2101.3001.7020) 查看对应的 cuda 版本

流程如下

1. 首先下载`miniconda` ，我下的是`python3.8`的
2. 创建自己的自定义环境
3. 检查自己的`cuda`版本，我的是`cuda:12.0`
4. 然后再`pytorch`上找到对应`cuda`版本的进行下载，`pip install`或者`conda install` 都可以

![image-20240808203916505](C:\Users\19409\Desktop\MD\Image\image-20240808203916505.png)

