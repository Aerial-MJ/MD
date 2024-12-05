## Make步骤

完整的 `CMakeLists.txt`

```cmake
# 设置最低版本
cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(MyProject)

# 设置编译选项
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-Wall -g")

# 定义源文件目录
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(BUILD_DIR "${CMAKE_BINARY_DIR}")

# 包含头文件目录
include_directories(${INCLUDE_DIR})

# 搜索源文件
file(GLOB SOURCES "${SRC_DIR}/*.c")

# 定义目标
add_executable(program ${SOURCES})

# 设置输出目录
set_target_properties(program PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
)
```

------

### **项目结构**

假设你的项目结构是这样的：

```
project/
├── CMakeLists.txt
├── src/
│   ├── main.c
│   ├── module1.c
│   └── module2.c
├── include/
│   ├── module1.h
│   └── module2.h
└── build/ (由 CMake 生成)
```

------

### **构建步骤**

1. **进入项目目录**：

   ```bash
   cd project
   ```

2. **创建 `build/` 目录并进入**：

   ```bash
   mkdir build && cd build
   ```

3. **运行 CMake**：

   ```bash
   cmake ..
   ```

4. **编译项目**：

   ```bash
   make
   ```

5. **运行生成的可执行文件**：

   ```bash
   ./program
   ```

6. **清理构建文件**： 如果需要清理构建文件，可以直接删除 `build/` 目录：

   ```bash
   rm -rf build
   ```

------

### **编译和运行结果**

**运行 `cmake ..` 输出**：

```bash
-- The C compiler identification is GNU 9.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /path/to/project/build
```

**运行 `make` 输出**：

```bash
[ 33%] Building C object CMakeFiles/program.dir/src/main.c.o
[ 66%] Building C object CMakeFiles/program.dir/src/module1.c.o
[100%] Building C object CMakeFiles/program.dir/src/module2.c.o
[100%] Linking C executable program
```

**运行程序输出**：

```bash
Hello from main!
Function from module1
Function from module2
```

------

### **CMakeLists.txt 的解析**

1. **项目基本设置**：

   ```cmake
   cmake_minimum_required(VERSION 3.10)
   project(MyProject)
   ```

   - 设置 CMake 最低版本。
   - 定义项目名称为 `MyProject`。

2. **定义源文件和头文件目录**：

   ```cmake
   set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
   set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
   ```

   - `SRC_DIR` 是存放 `.c` 文件的路径。
   - `INCLUDE_DIR` 是存放 `.h` 文件的路径。

3. **自动发现源文件**：

   ```cmake
   file(GLOB SOURCES "${SRC_DIR}/*.c")
   ```

   - 使用 `file(GLOB ...)` 动态找到 `src/` 目录下的所有 `.c` 文件。

4. **指定头文件路径**：

   ```cmake
   include_directories(${INCLUDE_DIR})
   ```

   - 包括 `include/` 目录作为头文件查找路径。

5. **创建目标可执行文件**：

   ```cmake
   add_executable(program ${SOURCES})
   ```

6. **设置输出目录**：

   ```cmake
   set_target_properties(program PROPERTIES
       RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
   )
   ```

   - 将生成的可执行文件放到 `build/` 目录中。

------

### **CMake 的优点**

- **跨平台**：CMake 支持在不同的平台（如 Windows、Linux、macOS）上生成构建文件。
- **自动管理依赖**：无需手动写复杂的依赖关系树。
- **灵活可扩展**：可以轻松地集成其他工具（如 GoogleTest、Valgrind）

## Makefile

当然可以！以下是完整的 `project/` 项目结构，包括源代码、头文件、Makefile，以及编译生成的 `build` 目录（用来存放中间文件和最终的可执行文件）。

------

### 项目结构

```
project/
├── include/
│   ├── module1.h
│   └── module2.h
├── src/
│   ├── main.c
│   ├── module1.c
│   └── module2.c
├── Makefile
└── build/  (在第一次运行 make 后生成)
```

------

### 代码部分

#### **1. main.c**

```c
#include <stdio.h>
#include "module1.h"
#include "module2.h"

int main() {
    printf("Hello from main!\n");
    function_from_module1();
    function_from_module2();
    return 0;
}
```

#### **2. module1.c**

```c
#include <stdio.h>
#include "module1.h"

void function_from_module1() {
    printf("Function from module1\n");
}
```

#### **3. module2.c**

```c
#include <stdio.h>
#include "module2.h"

void function_from_module2() {
    printf("Function from module2\n");
}
```

#### **4. module1.h**

```c
#ifndef MODULE1_H
#define MODULE1_H

void function_from_module1();

#endif
```

#### **5. module2.h**

```c
#ifndef MODULE2_H
#define MODULE2_H

void function_from_module2();

#endif
```

------

### Makefile

```makefile
# Compiler and flags
CC = gcc
CFLAGS = -Wall -g -Iinclude

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

# Find source files and corresponding object files
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SOURCES))

# Final executable
TARGET = $(BUILD_DIR)/program

# Default rule
all: $(TARGET)

# Linking the final executable
$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^

# Compiling object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Ensure the build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build artifacts
clean:
	rm -rf $(BUILD_DIR)
```

------

### 执行步骤

1. **进入项目目录**：

   ```bash
   cd project
   ```

2. **编译项目**：

   ```bash
   make
   ```

3. **运行程序**：

   ```bash
   ./build/program
   ```

4. **清理生成的文件**：

   ```bash
   make clean
   ```

------

### Makefile 的解析

1. **自动发现源文件**：

   ```makefile
   SOURCES = $(wildcard $(SRC_DIR)/*.c)
   OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SOURCES))
   ```

   - `$(wildcard $(SRC_DIR)/*.c)`：找到 `src/` 中所有 `.c` 文件。
   - `$(patsubst ...)`：将 `src/` 中的 `.c` 文件替换为 `build/` 中的 `.o` 文件。

2. **编译和链接**：

   - 每个 `.c` 文件生成一个 `.o` 文件，目标目录是 `build/`。
   - 所有 `.o` 文件链接成最终的可执行文件。

3. **自动创建目录**：

   - 在目标规则中使用 `| $(BUILD_DIR)`，确保 `build/` 目录在生成 `.o` 文件前已经存在。

4. **清理规则**：

   - `clean` 会删除 `build/` 目录及其中的文件。

------

### 编译和运行结果

**运行 `make`**：

```bash
gcc -Wall -g -Iinclude -c src/main.c -o build/main.o
gcc -Wall -g -Iinclude -c src/module1.c -o build/module1.o
gcc -Wall -g -Iinclude -c src/module2.c -o build/module2.o
gcc -o build/program build/main.o build/module1.o build/module2.o
```

**运行程序**：

```bash
./build/program
```

输出：

```
Hello from main!
Function from module1
Function from module2
```

**清理生成文件**：

```bash
make clean
```

清理后 `build/` 目录被删除。

------

### 总结

此项目结构简洁明了，Makefile 自动化管理了所有的编译、链接和清理任务。你只需专注于编写代码，而不必担心手动管理编译过程。

## CMakeList.txt

运行 CMake 的作用是 **生成适合目标构建环境的构建文件**（如 `Makefile`、Visual Studio 工程文件等），从而简化和标准化跨平台项目的构建流程。CMake 是一种配置工具，它解析项目的 `CMakeLists.txt` 文件，根据其中的指令生成构建文件，供后续使用构建工具（如 `make` 或 IDE）编译项目。

------

### **CMake 的核心作用**

1. **跨平台构建支持**：

   - CMake 能根据系统环境（如 Linux、Windows、macOS）自动生成对应的构建文件。例如：
     - Linux: 生成 `Makefile`。
     - Windows: 生成 Visual Studio 的 `.sln` 工程文件。
     - macOS: 支持 Xcode 构建系统。

2. **简化复杂项目的管理**：

   - 对于包含多个子目录、多个模块的大型项目，CMake 可以通过配置文件（`CMakeLists.txt`）集中管理所有编译规则和依赖关系。

3. **动态设置编译选项**：

   - CMake 支持添加自定义编译选项和编译器标志，例如调试模式（`-g`）或优化模式（`-O2`）。

   - 示例：

     ```bash
     cmake -DCMAKE_BUILD_TYPE=Debug ..
     ```

4. **依赖关系处理**：

   - 自动检查项目的外部依赖（如库和头文件）并链接。例如，依赖 `Boost` 或 `OpenCV` 的项目，可以通过简单指令找到并配置这些库。

5. **模块化开发支持**：

   - CMake 支持多个模块和子项目的独立管理。例如：
     - 主项目调用多个子项目的 `CMakeLists.txt`，实现分布式开发。

6. **统一生成构建文件**：

   - CMake 使用统一语法（`CMakeLists.txt`）配置项目的构建规则，无需针对不同平台单独维护构建脚本。

------

### **CMake 的执行流程**

#### **1. 配置阶段（运行 `cmake`）**

```bash
cmake ..
```

- 解析 `CMakeLists.txt` 文件，生成目标构建文件（如 `Makefile`）。
- 检查系统是否有所需的工具链（如编译器）和依赖项。

#### **2. 构建阶段（运行 `make` 或等效工具）**

```bash
make
```

- 使用 CMake 生成的构建文件进行编译，输出最终的可执行文件或库。

------

### **运行 CMake 的典型用途**

1. **生成构建文件**： 例如，在 `build/` 目录中运行：

   ```bash
   cmake ..
   ```

   - 生成项目的 `Makefile` 或其他构建文件。

2. **指定编译选项**：

   - 使用 `-D`设置变量：

     ```bash
     cmake -DCMAKE_BUILD_TYPE=Release ..
     ```

   - 通过选项选择优化级别或调试模式。

3. **指定编译器**：

   - 使用 `CMAKE_C_COMPILER` 和 `CMAKE_CXX_COMPILER`指定编译器：

     ```bash
     cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
     ```

4. **设置自定义安装路径**：

   - 设置目标文件的安装路径：

     ```bash
     cmake -DCMAKE_INSTALL_PREFIX=/custom/install/path ..
     ```

------

### **运行 CMake 的输出**

- 配置完成后，CMake 会输出类似以下的日志：

  ```
  -- The C compiler identification is GNU 9.4.0
  -- The CXX compiler identification is GNU 9.4.0
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /path/to/project/build
  ```

- 这表示 CMake 成功生成了构建系统文件（如 `Makefile`）。

------

### **总结**

运行 CMake 的作用在于通过配置文件（`CMakeLists.txt`），结合系统环境，生成跨平台的构建文件，使得项目构建更加高效、灵活和可移植。

## 单独的 main.c

如果 `main.c` 文件位于项目的根目录下，而其他模块的源文件仍在 `src/` 文件夹中，你可以对 `CMakeLists.txt` 做以下修改以适应这种情况：

------

### **修改后的 `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(MyProject)

# 设置包含头文件路径
include_directories(include)

# 查找 src/ 中的所有源文件
file(GLOB SRC_SOURCES "src/*.c")

# 添加 main.c 文件路径
set(MAIN_SOURCE "${CMAKE_SOURCE_DIR}/main.c")

# 将 main.c 和 src/ 中的所有源文件合并
set(SOURCES ${MAIN_SOURCE} ${SRC_SOURCES})

# 生成可执行文件
add_executable(program ${SOURCES})
```

------

### **解释**

1. **定义单独的 `main.c` 路径**：

   ```cmake
   set(MAIN_SOURCE "${CMAKE_SOURCE_DIR}/main.c")
   ```

   - `CMAKE_SOURCE_DIR` 是 CMake 的内置变量，指向 `CMakeLists.txt` 所在的项目根目录。

2. **保持 `src/\*.c` 文件的动态查找**：

   ```cmake
   file(GLOB SRC_SOURCES "src/*.c")
   ```

   - 使用 `file(GLOB ...)` 查找 `src/` 文件夹下的所有 `.c` 文件。

3. **将 `main.c` 与 `src/\*.c` 文件路径合并**：

   ```cmake
   set(SOURCES ${MAIN_SOURCE} ${SRC_SOURCES})
   ```

4. **生成可执行文件**：

   ```cmake
   add_executable(program ${SOURCES})
   ```

   - 将所有源文件（`main.c` 和 `src/*.c`）编译为可执行文件 `program`。

------

### **项目结构示例**

假设项目目录结构如下：

```
project/
├── CMakeLists.txt
├── main.c
├── src/
│   ├── module1.c
│   ├── module2.c
├── include/
│   ├── module1.h
│   ├── module2.h
└── build/ (CMake 构建文件夹)
```

------

### **构建步骤**

1. **创建构建文件夹**：

   ```bash
   mkdir build && cd build
   ```

2. **运行 CMake**：

   ```bash
   cmake ..
   ```

3. **编译项目**：

   ```bash
   make
   ```

4. **运行生成的可执行文件**：

   ```bash
   ./program
   ```

