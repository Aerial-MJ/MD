## init

在 Linux 系统中，`init` 文件通常是指系统初始化过程中的关键组件，它负责系统启动时的初始化操作。具体来说，`init` 是 Linux 系统启动时执行的第一个进程，它的进程 ID (PID) 为 1，并且负责启动和管理其他用户空间进程。

`init` 有几种常见的形式，包括传统的 `init`、`systemd` 等。不同的 Linux 发行版可能使用不同的 `init` 系统。下面是关于这些不同类型的 `init` 的详细介绍。

### 1. **传统的 init（SysVinit）**

传统的 `init` 是 Unix 风格的初始化系统，通常被称为 **SysVinit**。它通过一系列的启动脚本来初始化系统、启动进程以及管理服务。

#### **文件和目录结构：**

- `/etc/inittab`：这是 SysVinit 使用的配置文件，定义了系统的运行级别、启动脚本等信息。
- `/etc/init.d/`：存放系统服务脚本的目录。在该目录下，脚本用于启动、停止、重启等服务。
- `/etc/rc.d/` 或 `/etc/rc[runlevel].d/`：根据不同的运行级别启动相应的脚本。

#### **典型配置文件示例 (`/etc/inittab`)：**

```bash
# Default runlevel.
id:3:initdefault:

# System initialization
si::sysinit:/etc/init.d/rcS

# Run level 3: multi-user mode
l3:3:wait:/etc/init.d/rc 3
```

- `id:3:initdefault:` 表示默认的运行级别是 3（多用户模式）。
- `si::sysinit:/etc/init.d/rcS` 表示系统初始化时要执行 `/etc/init.d/rcS` 脚本。
- **`/etc/init.d/rc 3`**：这是要执行的脚本。`/etc/init.d/rc` 是一个脚本，它负责启动在运行级别 3 中需要的服务。`3` 是传递给脚本的参数，表示在运行级别 3 时需要启动哪些服务。该脚本会根据当前的运行级别执行相应的操作，通常它会调用一系列服务启动脚本（例如 `/etc/init.d/` 目录中的脚本）来启动系统服务。

#### **如何切换运行级别：**

SysVinit 使用 `init` 命令来切换运行级别。常见的运行级别包括：

- `0`：关机
- `1`：单用户模式
- `2`：多用户模式（无网络）
- `3`：多用户模式（有网络）
- `5`：图形界面模式
- `6`：重启

```bash
init 3  # 切换到运行级别 3
```

------

### 2. **systemd（现代的 init 系统）**

`systemd` 是现代 Linux 发行版（如 Ubuntu、CentOS、Debian 等）广泛使用的初始化系统。它旨在提供并行化启动过程，简化服务管理，并且更加高效。

**关键特点：**

- **并行化启动：** systemd 允许同时启动多个服务，以减少启动时间。

- **单位文件（unit files）：** systemd 使用配置文件称为单位文件（unit files）来描述服务。单位文件通常存放在 `/etc/systemd/system/` 或 `/lib/systemd/system/` 目录下。

- ```
  chenjiarui202422080422@ubuntu:/etc/systemd/system$ cat sshd.service 
  [Unit]
  Description=OpenBSD Secure Shell server
  Documentation=man:sshd(8) man:sshd_config(5)
  After=network.target auditd.service
  ConditionPathExists=!/etc/ssh/sshd_not_to_be_run
  
  [Service]
  EnvironmentFile=-/etc/default/ssh
  ExecStartPre=/usr/sbin/sshd -t
  ExecStart=/usr/sbin/sshd -D $SSHD_OPTS
  ExecReload=/usr/sbin/sshd -t
  ExecReload=/bin/kill -HUP $MAINPID
  KillMode=process
  Restart=on-failure
  RestartPreventExitStatus=255
  Type=notify
  RuntimeDirectory=sshd
  RuntimeDirectoryMode=0755
  
  [Install]
  WantedBy=multi-user.target
  Alias=sshd.service
  
  ```

- **依赖管理：** systemd 管理进程和服务之间的依赖关系，确保服务按顺序启动。

#### **常见命令：**

- 启动服务：

  ```bash
  systemctl start 服务名
  ```

- 停止服务：

  ```bash
  systemctl stop 服务名
  ```

- 重启服务：

  ```bash
  systemctl restart 服务名
  ```

- 查看服务状态：

  ```bash
  systemctl status 服务名
  ```

- 查看默认目标（类似运行级别）：

  ```bash
  systemctl get-default
  ```

**默认目标：**

- **multi-user.target**：类似于传统的运行级别 3，多用户模式。
- **graphical.target**：类似于传统的运行级别 5，图形界面模式。
- **reboot.target**：系统重启。
- **halt.target**：系统关机。

**系统启动文件：**

- `/etc/systemd/system/`: 存放用户自定义的单位文件。
- `/lib/systemd/system/`: 存放系统级的单位文件。

**示例服务文件（`/etc/systemd/system/my_service.service`）：**

```ini
[Unit]
Description=My Custom Service
After=network.target

[Service]
ExecStart=/usr/local/bin/my_service

[Install]
WantedBy=multi-user.target
```

- `ExecStart`：定义启动服务的命令。
- `After`：指定服务依赖的其他服务，确保在 `network.target` 后启动。

**启动服务：**

```bash
systemctl enable my_service  # 设置为开机自启动
systemctl start my_service   # 启动服务
```

------

### 3. **Upstart（过时的 init 系统）**

`Upstart` 是 Ubuntu 在较早版本中使用的初始化系统，后被 `systemd` 取代。它也基于事件驱动，在事件发生时启动或停止服务。

------

### 总结

- **传统 `init`（SysVinit）**：基于运行级别的顺序启动，使用脚本进行服务管理，配置文件是 `/etc/inittab`。
- **`systemd`**：现代的初始化系统，支持并行启动和更高效的服务管理，使用单位文件来描述服务和目标，配置文件存放在 `/etc/systemd/system/`。
- **`Upstart`**：过时的初始化系统，主要在 Ubuntu 的早期版本中使用。

不同的 Linux 发行版使用不同的初始化系统。`systemd` 已成为许多现代发行版的标准，而 SysVinit 和 Upstart 在较老的系统中依然存在。如果你使用的系统是基于 `systemd`，那么你应该学习 `systemd` 的管理方法。如果是老版本的 Linux，你可能会遇到 SysVinit。

## 开机启动

在 Linux 系统中，`init` 是开机时执行的第一个进程，其进程 ID (PID) 为 1。`init` 进程的任务是启动系统并初始化所有其他进程。其初始化过程包括了启动内核、加载设备驱动、挂载文件系统、启动各种服务等。具体的初始化流程如下：

当计算机开机后，**Bootloader**（如 GRUB）会加载操作系统的内核到内存中并启动它。内核首先完成硬件检测、驱动加载、文件系统挂载等操作。这时，内核会启动第一个用户空间进程，也就是 `init` 进程。

内核启动后会执行 `/sbin/init` 或 `/etc/init`（具体路径与发行版和系统配置相关）。`init` 是用户空间的第一个进程，进程号为 1。`init` 在启动时会负责后续的系统初始化工作。

`init` 会根据配置文件进行初始化，传统的 SysVinit 和现代的 `systemd` 有不同的初始化方法。

#### **SysVinit**

传统的 `init`（即 SysVinit）会使用 `/etc/inittab` 文件来决定系统的初始化流程。`init` 会按步骤执行以下任务：

- **读取配置文件：** `init` 会首先读取 `/etc/inittab` 文件，该文件定义了不同的运行级别以及每个运行级别需要启动的服务。

- **启动运行级别：** `init` 会根据 `/etc/inittab` 文件的配置启动相应的运行级别。例如，如果配置文件中的默认运行级别是 3（多用户模式），`init` 就会切换到运行级别 3，并执行与该运行级别相关的启动脚本。

  运行级别的定义：

  - 0：关机
  - 1：单用户模式
  - 2：多用户模式（无网络）
  - 3：多用户模式（有网络）
  - 4：用户自定义
  - 5：图形用户界面（GUI）
  - 6：重启

- **启动脚本：** `init` 会执行与当前运行级别对应的启动脚本。例如，在运行级别 3 时，`init` 会执行 `/etc/init.d/rc` 脚本，并传递相应的参数来启动服务。

#### **systemd**

现代的 `init` 系统（即 `systemd`）与传统的 SysVinit 有较大不同。`systemd` 使用的是一种称为 **单位文件（unit files）** 的配置文件，来描述服务、挂载点、设备等。`systemd` 会并行启动服务，极大地提高了启动效率。

`systemd` 的启动过程：

- **读取配置文件：** `systemd` 会从配置文件 `/etc/systemd/system/` 和 `/lib/systemd/system/` 目录加载服务和单位文件。这些文件定义了各种服务的启动顺序和依赖关系。
- **并行启动：** `systemd` 会在启动过程中并行启动服务，而不是像传统 `init` 那样按顺序启动。它会根据服务之间的依赖关系来决定启动顺序。
- **目标（targets）：** `systemd` 通过目标（target）来表示不同的运行级别。例如：
  - `multi-user.target`：类似于运行级别 3（多用户模式）。
  - `graphical.target`：类似于运行级别 5（图形界面模式）。

####  启动系统服务

无论是 SysVinit 还是 `systemd`，`init` 都会根据配置启动各种系统服务。例如：

- 启动网络服务，确保网络连接。
- 启动用户空间的进程，如登录管理器。
- 启动后台服务（如数据库服务、Web 服务器等）。

在所有系统服务启动完毕后，`init` 会启动一个或多个登录进程。例如，在图形界面模式下，`init` 会启动一个显示管理器（如 `gdm` 或 `lightdm`）来显示登录界面。而在没有图形界面的多用户模式下，它会启动 `getty` 进程，等待用户登录。

完成上述所有步骤后，系统就进入了正常的运行状态。用户可以登录系统并开始使用计算机。