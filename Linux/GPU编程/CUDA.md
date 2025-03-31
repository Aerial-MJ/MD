## CUDA

### CUDA 是什么

**CUDA** 是一个**并行计算平台和编程模型**，本质上是一种**软件平台**，它允许开发者使用 C、C++、Python 等编程语言，在 NVIDIA 的 GPU 上编写并行计算程序。

> **简而言之：CUDA 是 NVIDIA 提供的一整套工具和 API，帮助开发者在 GPU 上进行计算。**

#### CUDA 的本质

从不同的角度来看，CUDA 可以理解为以下几种东西：

| 角度                                         | 描述                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| **平台（Platform）**                         | CUDA 是一个专门用于 GPU 编程的平台，类似于 CPU 上的 POSIX、Windows API，负责 GPU 资源的调度和管理。 |
| **编程模型（Programming Model）**            | CUDA 提供了一种基于线程块（Thread Block）、网格（Grid）等概念的并行编程模型。 |
| **API（Application Programming Interface）** | CUDA 提供了丰富的 API（如 `cudaMalloc()`、`cudaMemcpy()`、`cudaFree()`），简化 GPU 编程。 |
| **工具链（Toolkit）**                        | CUDA 提供了编译器（`nvcc`）、调试器（`cuda-gdb`）、性能分析工具（`nvprof`、`Nsight`）等完整的开发环境。 |
| **库（Library）**                            | CUDA 包含许多高性能库，如 `cuBLAS`（线性代数）、`cuDNN`（深度学习）、`Thrust`（并行算法）等。 |

#### CUDA 的架构结构

CUDA 的整体架构包括以下几个关键部分：

1. 硬件（Hardware）
   - GPU 的物理硬件，如 CUDA 核心、SM（流式多处理器）等。
2. 驱动（Driver）
   - NVIDIA 提供的 GPU 驱动，负责 GPU 的底层管理。
3. 运行时（Runtime）
   - CUDA 提供的 API（如 `cudaMalloc()`、`cudaMemcpy()`）供用户编写程序并调用 GPU。
4. 库（Libraries）
   - NVIDIA 提供的高性能库，简化矩阵运算、深度学习等任务。
5. 编译器（Compiler）
   - `nvcc`（NVIDIA CUDA Compiler）编译 CUDA C/C++ 代码，生成可在 GPU 上运行的 PTX（Parallel Thread Execution）指令

CUDA支持多种语言，如C，C++，Python，Fortran等，本文中的示例主要是C/C++语言。CUDA C在标准C的基础上增加了一些标准函数，编译指令等以实现对GPU的控制。CUDA API采用分层设计，从底向上有[CUDA Driver API](https://zhida.zhihu.com/search?content_id=249848054&content_type=Article&match_order=1&q=CUDA+Driver+API&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDI0NjExMjYsInEiOiJDVURBIERyaXZlciBBUEkiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDk4NDgwNTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.zjS-imc24ZTiGjWO9a9h0BVsG2SL3t8wtIm84FTtTbk&zhida_source=entity)，[CUDA Runtime API](https://zhida.zhihu.com/search?content_id=249848054&content_type=Article&match_order=1&q=CUDA+Runtime+API&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDI0NjExMjYsInEiOiJDVURBIFJ1bnRpbWUgQVBJIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQ5ODQ4MDU0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.s3lQR74i1KbKdD5-GQaMW5NncvWKDgxLbMPckZMxjsI&zhida_source=entity)，CUDA Libraries（在基础API之上封装的一些库，例如cublas，cutlass等等）。用户开发CUDA程序可以基于Driver API或Runtime API（一般都是使用Runtime API），但是不能同时使用两者。（本系列文章中的示例代码均基于CUDA Runtime API）

![image-20250318170054274](../../Image/image-20250318170054274.png)

CUDA程序通过**nvcc**编译器进行编译，CUDA程序中包含CPU部分代码和GPU部分代码，编译时会分别编译，CPU部分代码叫**host code**，GPU部分代码一般叫**device code**。

![image-20250318170132220](../../Image/image-20250318170132220.png)

#### CUDA 与 PyTorch、TensorFlow 的关系

- **CUDA 是底层平台**，负责 GPU 计算的核心机制。
- **PyTorch / TensorFlow** 等框架在底层调用 CUDA 库（如 `cuDNN`、`cuBLAS`），以实现高效的 GPU 训练和推理。

 PyTorch 示例（使用 CUDA 加速）：

```python
import torch

# 创建两个 GPU 张量
a = torch.tensor([1.0, 2.0, 3.0]).cuda()
b = torch.tensor([4.0, 5.0, 6.0]).cuda()

# 在 GPU 上执行运算
c = a + b
print(c)  # tensor([5., 7., 9.], device='cuda:0')
```

#### CUDA、cuDNN、NVIDIA 驱动的区别

| 名称            | 功能                                                         |
| --------------- | :----------------------------------------------------------- |
| **CUDA**        | 提供 GPU 并行计算的编程平台和工具，支持 GPU 上的通用计算。   |
| **cuDNN**       | 专为深度学习优化的 GPU 加速库，是 PyTorch、TensorFlow 等框架的核心依赖。 |
| **NVIDIA 驱动** | GPU 的底层驱动程序，CUDA 必须依赖正确版本的驱动才能运行。    |

### Driver API/ Runtime API

✅ **大多数情况下，安装 NVIDIA 驱动时，Driver API 会自动安装。**
✅ **使用 PyTorch、TensorFlow 等深度学习框架时，通常无需额外关注 Driver API。若使用 PyTorch、TensorFlow 等框架，通常不需要直接调用 Driver API，安装 NVIDIA 驱动和 CUDA Runtime API 即可。**
✅ **编写 CUDA C/C++ 程序时，Driver API 更为重要。**

#### 什么是 Driver API

- **Driver API** 是 CUDA 的核心部分，负责 GPU 的底层管理。
- 它和操作系统内核紧密集成，控制 GPU 的硬件资源。
- NVIDIA 驱动程序中的 Driver API 提供了较底层的控制接口，适合更底层的 GPU 控制和调度。

> 🚨 **注意：Driver API ≠ Runtime API**
>
> - **Driver API**（底层）➡️ 提供对 GPU 设备的直接控制，需要显式初始化和更复杂的调用。
> - **Runtime API**（更常用）➡️ 更高级，简化了 GPU 编程的细节，推荐用于 PyTorch、TensorFlow 等框架。

#### CUDA Driver API vs Runtime API 的区别

CUDA 提供了两种编程接口：

- **Driver API**（驱动 API）
- **Runtime API**（运行时 API）

这两者的区别主要体现在**编程模型**、**调用方式**、**灵活性**和**复杂度**上。以下是详细对比：

##### 1. 核心区别对比

| 特点             | **Driver API**                                               | **Runtime API**                                 |
| ---------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| **定义**         | 提供底层 GPU 控制，允许开发者直接管理 GPU 资源。             | 提供更简化的 API，隐藏了底层 GPU 控制细节。     |
| **初始化**       | 需要显式初始化（`cuInit()`）。                               | 自动初始化，调用任意 CUDA API 时自动完成。      |
| **API 调用风格** | 函数名以 `cu` 开头（如 `cuMemcpy()`）。                      | 函数名以 `cuda` 开头（如 `cudaMemcpy()`）。     |
| **灵活性**       | 更灵活，允许对 GPU 资源进行更细粒度的控制。                  | 更简洁，推荐用于 PyTorch、TensorFlow 等框架。   |
| **复杂度**       | 编程更复杂，需显式管理上下文、内存、流 (Stream) 等。         | 编程更简单，封装了复杂的 GPU 管理逻辑。         |
| **兼容性**       | 适用于不依赖 CUDA Runtime 的项目，可与其他 C++ 项目无缝集成。 | 需要 CUDA Runtime 环境支持，依赖 `libcuda.so`。 |

##### 2. 示例代码对比

以下是同一功能（在 GPU 上执行向量加法）分别用 **Driver API** 和 **Runtime API** 实现的对比：

###### （1）Runtime API 示例

更简洁，推荐在 PyTorch、TensorFlow 等框架中使用。

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 5;
    int h_a[N] = {1, 2, 3, 4, 5};
    int h_b[N] = {10, 20, 30, 40, 50};
    int h_c[N];

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, N>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        std::cout << h_c[i] << " ";
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

✅ **特点：**

- 简洁易用，CUDA 初始化和上下文管理由系统自动完成。
- 适用于 PyTorch、TensorFlow 等框架。

###### （2）Driver API 示例

Driver API 提供了更底层的控制，需要显式初始化和管理资源。

```cpp
#include <cuda.h>
#include <iostream>

__global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 5;
    int h_a[N] = {1, 2, 3, 4, 5};
    int h_b[N] = {10, 20, 30, 40, 50};
    int h_c[N];

    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    CUdeviceptr d_a, d_b, d_c;

    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    cuMemAlloc(&d_a, N * sizeof(int));
    cuMemAlloc(&d_b, N * sizeof(int));
    cuMemAlloc(&d_c, N * sizeof(int));

    cuMemcpyHtoD(d_a, h_a, N * sizeof(int));
    cuMemcpyHtoD(d_b, h_b, N * sizeof(int));

    cuModuleLoad(&cuModule, "vectorAdd.ptx");
    cuModuleGetFunction(&cuFunction, cuModule, "add");

    void *args[] = { &d_a, &d_b, &d_c };
    cuLaunchKernel(cuFunction, 1, 1, 1, N, 1, 1, 0, 0, args, 0);

    cuMemcpyDtoH(h_c, d_c, N * sizeof(int));

    for (int i = 0; i < N; i++)
        std::cout << h_c[i] << " ";
    
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuCtxDestroy(cuContext);

    return 0;
}
```

✅ **特点：**

- 需要显式初始化 GPU 设备 (`cuInit()`)、创建上下文 (`cuCtxCreate()`) 等。
- 适合更底层、对 GPU 资源精细控制的场景。

##### 3. 适用场景对比

| 使用场景                                        | 推荐 API          |
| ----------------------------------------------- | ----------------- |
| **使用 PyTorch、TensorFlow 等框架**             | ✅ **Runtime API** |
| **编写深度学习模型训练/推理代码**               | ✅ **Runtime API** |
| **编写 GPU 内核代码（CUDA C/C++）**             | ✅ **Runtime API** |
| **需要对 GPU 设备进行精细控制 (如多 GPU 调度)** | ✅ **Driver API**  |
| **开发 CUDA 底层库 (如 cuDNN、cuBLAS)**         | ✅ **Driver API**  |
| **开发支持多种 GPU 设备的通用工具**             | ✅ **Driver API**  |

##### 4. 总结：如何选择？

- ✅ **Runtime API**：更易用，推荐使用。几乎所有深度学习框架（如 PyTorch、TensorFlow）都使用 Runtime API。
- ✅ **Driver API**：更底层、更灵活，适用于系统级开发、编写 GPU 管理工具、或实现 CUDA 底层库时。

在绝大多数 GPU 编程场景下，推荐使用 **Runtime API**。只有在以下特殊情况下，才需使用 **Driver API**：

- 需要精确控制 GPU 设备、流（stream）或上下文（context）。
- 要求与非 CUDA 编程环境（如 OpenGL、DirectX）交互。
