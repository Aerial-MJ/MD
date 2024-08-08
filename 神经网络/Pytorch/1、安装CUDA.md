# 安装CUDA

 ## cuda安装

 windows10 版本安装 [CUDA](https://so.csdn.net/so/search?q=CUDA&spm=1001.2101.3001.7020) ，首先需要下载两个安装包

- CUDA toolkit（toolkit就是指工具包）
- cuDNN

注：cuDNN 是用于配置深度学习使用

cuDNN 其实就是 CUDA 的一个补丁而已，专为深度学习运算进行优化的。

- CUDA：为“GPU通用计算”构建的运算平台。
- cudnn：为深度学习计算设计的软件库。
- CUDA Toolkit (nvidia)： CUDA完整的工具安装包，其中提供了 Nvidia 驱动程序、开发 CUDA 程序相关的开发工具包等可供安装的选项。包括 CUDA 程序的编译器、IDE、调试器等，CUDA 程序所对应的各式库文件以及它们的头文件。
- CUDA Toolkit (Pytorch)： CUDA不完整的工具安装包，其主要包含在使用 CUDA 相关的功能时所依赖的动态链接库。不会安装驱动程序。
  （NVCC 是CUDA的编译器，只是 CUDA Toolkit 中的一部分）
- 注：CUDA Toolkit 完整和不完整的区别：在安装了CUDA Toolkit (Pytorch)后，只要系统上存在与当前的 cudatoolkit 所兼容的 Nvidia 驱动，则已经编译好的 CUDA 相关的程序就可以直接运行，不需要重新进行编译过程。如需要为 Pytorch 框架添加 CUDA 相关的拓展时（Custom C++ and CUDA Extensions），需要对编写的 CUDA 相关的程序进行编译等操作，则需安装完整的 Nvidia 官方提供的 CUDA Toolkit。

## Cuda版本的安装及下载

可以看到自己电脑支持的cuda（支持的最高版本）
