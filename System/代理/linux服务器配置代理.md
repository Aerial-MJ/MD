# 1

你在服务器上跑的是 Linux，而且是非 root 用户，那我们选择一个：

>  **免安装、能直接运行、支持你的架构的版本**

------

##  最可能适用的版本（你大概率是 64 位 x86 架构）

请下载：

### 👉 `mihomo-linux-amd64-alpha-d036d98.gz`

这是给 **x86_64 / amd64 架构的 Linux** 准备的压缩包版本，适合大部分云服务器或本地 Linux 桌面（比如 Ubuntu, Debian, CentOS）。

------

## 🎯 下载 & 解压命令：

```bash
wget https://github.com/MetaCubeX/mihomo/releases/download/alpha-d036d98/mihomo-linux-amd64-alpha-d036d98.gz
gzip -d mihomo-linux-amd64-alpha-d036d98.gz
mv mihomo-linux-amd64-alpha-d036d98 mihomo
chmod +x mihomo
```

这时候你就得到了一个可执行文件 `./mihomo`，可以直接运行，不需要 root 权限！

------

## 🚀 下一步：运行 mihomo 并挂上你的订阅

你订阅地址是这样的格式：

```
https://aappi.xn--ihq550b15yw2rtoa.xn--fiqs8s/api/v1/client/subscribe?token=282c8d6fc1eabf58790f12c5bc89fe35
```

这个是订阅地址，不能直接用，需要通过订阅转换工具（Subconverter）变成 Clash 的配置文件，或者我可以帮你转换。

# 2

## 📦 我可以帮你转换一下

你只需要告诉我你希望选择哪个节点（自动 / 手动都行），我可以帮你生成 `config.yaml`，你直接用 mihomo 就能跑起来。

或者你自己用在线转换服务，比如这个：

- https://sub.mahod.cc/
- https://acl4ssr.netlify.app/
- [ACL4SSR 在线订阅转换](https://acl4ssr-sub.github.io/)

选择 Clash 格式 → 粘贴订阅 → 下载 Clash config 文件即可。

------

需要我：

- 🔁 帮你生成完整配置文件？
- 🧩 帮你写好启动脚本？
- 🐍 帮你配置 Python 的代理环境？

都没问题，告诉我你当前代理节点大致地区（比如香港/日本/自动），我来给你配一套 😎

非常棒，进度飞快！你已经有了：

-  可运行的 `./mihomo`
-  创建好了配置目录：`~/.config/clash`

接下来我们来搞定：

------

##  步骤 1：把你的订阅链接转换成 Clash 的配置文件

你的订阅地址是：

```
https://aappi.xn--ihq550b15yw2rtoa.xn--fiqs8s/api/v1/client/subscribe?token=282c8d6fc1eabf58790f12c5bc89fe35
```

Clash 不支持直接读订阅，需要转换成 Clash 的 `config.yaml`，这一步我们可以用在线转换工具来完成。

------

### 🔁 方法 A（推荐）：使用 [Subconverter 在线转换](https://sub.mahod.cc/)

1. 打开这个链接：
    👉 [https://sub.mahod.cc](https://sub.mahod.cc/)
2. 在页面中：
   - **输入订阅链接**
   - **格式选择：Clash.Meta 或 Clash**
   - **点击生成配置**
   - 点击 `复制 YAML 配置`，或者下载配置文件
3. 然后你把它保存到你服务器的目录中：

```bash
vim ~/.config/clash/config.yaml
```

然后粘贴进去你复制的配置内容（粘贴后按 `ESC`，输入 `:wq` 保存退出）。

------

##  步骤 2：运行 mihomo（以后台方式）

```bash
nohup ./mihomo -d ~/.config/clash > mihomo.log 2>&1 &
```

说明：

- `-d` 参数指定配置目录
- `nohup` 让它在后台运行
- 日志会写入 `mihomo.log`

------

##  步骤 3：确认代理端口启动成功

默认情况下，`mihomo` 会监听：

- HTTP 代理端口：`127.0.0.1:7890`
- SOCKS 代理端口：`127.0.0.1:7891`

你可以验证：

```bash
curl -x http://127.0.0.1:7890 https://www.google.com
```

如果返回了 HTML，那就通了 🎉

------

##  步骤 4：让 Python 使用代理访问 Yahoo 数据

在你的 Python 脚本最前面加入：

```python
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
```

然后照常运行 `YahooDownloader().fetch_data()` 就 OK 了！

------

##  附加：常用管理命令

### 停止 mihomo：

```bash
ps aux | grep mihomo
kill <PID>
```

### 修改代理策略（比如选节点）：

你可以用 Clash 的控制面板（需要额外配置），或者用 CLI 控制方式也可以，我可以后续告诉你如何设置。

------

现在你可以开始跑 FinRL 的数据下载了！

如果你已经粘贴好了订阅配置但无法联网，或者想让我帮你生成 config.yaml，也完全可以～要不要我帮你生成一个直接能用的 config 文件？你只要复制就能跑。

# 3

好问题！我们在 [Subconverter](https://sub.mahod.cc/) 这样的订阅转换网站填写这些选项时，选得对，生成的配置才能直接用于 `mihomo`（Clash.Meta）。

我来逐项告诉你怎么选 ：

------

##  填写方式如下：

### 🔗 **订阅链接**：

粘贴你的订阅地址：

```
https://aappi.xn--ihq550b15yw2rtoa.xn--fiqs8s/api/v1/client/subscribe?token=282c8d6fc1eabf58790f12c5bc89fe35
```

------

### 📦 **生成格式**：

👉 **选这个：`Clash.Meta`**

> 如果没有 Clash.Meta 选项，就选 `Clash`，大部分兼容，`mihomo` 支持。

------

### 🔗 **短链选择**：

❌ **不要勾选**（留空）

> 短链是给移动端生成订阅 URL 用的，不是我们要的。

------

### ⚙️ **规则配置**：

选择这个推荐的规则组（干净、好用）：

👉 **`ACL4SSR_Online_Full.yaml`**

> 如果你只想访问外网，不走国内网站，可以选 `ACL4SSR_Online_Mini.yaml`（速度更快点）

------

### 其它选项保持默认即可（分组策略、节点排序等）

------

##  最终操作

1. 选好之后点【生成配置】
2. 然后点击【下载配置文件】
3. 或者点击【复制配置】粘贴到你服务器的文件里：

```bash
vim ~/.config/clash/config.yaml
```

粘贴内容，保存退出。

------

##  运行 mihomo：

```bash
nohup ./mihomo -d ~/.config/clash > mihomo.log 2>&1 &
```

然后试试：

```bash
curl -x http://127.0.0.1:7890 https://www.google.com
```

能访问说明一切 OK！

------

需要我给你一键生成好的 `config.yaml`？我可以在线帮你转换好，你直接复制粘贴即可。你想用哪种代理节点（香港 / 日本 / 自动 / 美国）为主？我来给你配好一份。

# 4

明白了！你目前只有一个订阅链接：

```
https://aappi.xn--ihq550b15yw2rtoa.xn--fiqs8s/api/v1/client/subscribe?token=282c8d6fc1eabf58790f12c5bc89fe35
```

但没有现成的 `config.yaml` 配置文件。为了让 `mihomo` 正常工作，我们需要将这个订阅链接转换为 Clash/Mihomo 可用的配置文件。以下是详细的步骤：

------

##  步骤 1：使用在线工具转换订阅链接

我们将使用 [ACL4SSR 在线订阅转换工具](https://acl4ssr-sub.github.io/) 来完成转换。

1. 打开 [ACL4SSR 在线订阅转换工具](https://acl4ssr-sub.github.io/)。
2. 在页面中，找到“订阅链接”输入框，粘贴你的订阅链接。
3. 在“生成格式”中，选择 **Clash**。
4. 在“规则配置”中，选择 **ACL4SSR_Online_Full.yaml**（如果你只想代理国外流量，可以选择 **ACL4SSR_Online_Mini.yaml**）。
5. 点击“生成配置”按钮。
6. 页面会生成一个新的链接，点击该链接即可下载生成的 `config.yaml` 文件。

------

##  步骤 2：保存配置文件到服务器

将下载的 `config.yaml` 文件上传到你服务器的 `~/.config/clash/` 目录下。如果你在服务器上直接操作，可以使用以下命令：

```bash
wget -O ~/.config/clash/config.yaml "生成的配置文件链接"
```

请将 `"生成的配置文件链接"` 替换为你在上一步中获得的链接。

------

##  步骤 3：运行 mihomo

确保 `mihomo` 可执行文件位于你的当前目录下，然后运行以下命令启动它：

```bash
nohup ./mihomo -d ~/.config/clash > mihomo.log 2>&1 &
```

这将使 `mihomo` 在后台运行，并将日志输出到 `mihomo.log` 文件中。

------

##  步骤 4：验证代理是否正常工作

你可以使用以下命令测试代理是否正常：

```bash
curl -x http://127.0.0.1:7890 https://www.google.com
```

如果返回了 HTML 内容，说明代理设置成功。

------

##  步骤 5：配置 Python 使用代理

在你的 Python 脚本中，添加以下代码以使用代理：

```python
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
```

这样，`YahooDownloader().fetch_data()` 等函数就可以通过代理访问外网了。

------

如果你在使用过程中遇到任何问题，或者需要进一步的帮助，请随时告诉我！

# 5

好的！我来 **详细讲解一下改法一（使用 Clash 的 HTTP API 切换节点）**，让你在 **Linux 或服务器中通过命令行轻松换节点**。

------

## 🧠 1. 背景：为什么能用 curl 控制 Clash？

你在配置文件中写了：

```yaml
external-controller: :9090
```

这表示 Clash 开启了一个监听在本地 9090 端口的“控制接口”，它提供了 HTTP API，用来远程查看状态、切换节点、测试延迟等操作。

------

## 🧰 2. 准备工作

###  前提条件：

- Clash 已经在后台运行 
- 监听了 9090 端口 
- 你知道要切换到哪个节点，比如 `🇭🇰 香港01`

------

## 🎯 3. 实际操作步骤（一步步来）

### ✏️ 第一步：确认当前正在运行的节点组名

你 YAML 里写的是：

```yaml
- name: 🚀 节点选择
```

所以这个是你要操作的节点组，**必须完全匹配，包括 emoji、空格、大小写**。

------

### 🧪 第二步：测试一下 Clash 控制接口是否通

运行这个命令看看能不能获取所有节点组：

```bash
curl http://127.0.0.1:9090/proxies
```

如果你看到返回了 JSON 数据，比如有 `"🚀 节点选择"`，说明控制端口没问题。

------

### 🚀 第三步：发送 PUT 请求切换节点

比如你要切换到 “🇭🇰 香港01”，就运行下面的命令：

```bash
curl -X PUT http://127.0.0.1:9090/proxies/🚀%20节点选择 \
  -H "Content-Type: application/json" \
  -d '{"name": "🇭🇰 香港01"}'
```

注意事项：

- `🚀%20节点选择` 是 `🚀 节点选择` 的 URL 编码（空格必须变成 `%20`）
- `-d` 后面是目标节点的名字，必须和你 `proxies` 列表里的 name 完全一致（包括 emoji）
- 你也可以换成其它节点，比如 `"🇸🇬 狮城01"` 或 `"🇺🇸 美国02"`

------

###  第四步：验证切换成功

你可以这样测试代理是否生效了：

```bash
curl -x http://127.0.0.1:7890 https://ipinfo.io
```

看返回的 IP 和国家是否跟你切换的节点一致（比如你选香港，就应该显示香港的 IP）。

------

## 🔒 如果有 `secret` 密码怎么办？

如果你在配置文件里设置了控制接口密码：

```yaml
external-controller: :9090
secret: "abc123"
```

那么你的命令需要添加 `Authorization` 头：

```bash
curl -X PUT http://127.0.0.1:9090/proxies/🚀%20节点选择 \
  -H "Authorization: Bearer abc123" \
  -H "Content-Type: application/json" \
  -d '{"name": "🇭🇰 香港01"}'
```

------

## 🔁 Bonus：切换多个节点组怎么办？

你可能还有 `🌍 国外媒体`、`♻️ 自动选择` 等组，也可以用同样方法切：

```bash
curl -X PUT http://127.0.0.1:9090/proxies/🌍%20国外媒体 \
  -d '{"name": "🇭🇰 香港01"}'
```

------

## 总结一句话：

> 只要你能访问 Clash 的控制端口 `9090`，就能用 `curl` 发 PUT 请求自由切换节点，比改 YAML 还方便，还不用重启 Clash。

# 步骤

我们将使用 [ACL4SSR 在线订阅转换工具](https://acl4ssr-sub.github.io/) 来完成转换。

打开 [ACL4SSR 在线订阅转换工具](https://acl4ssr-sub.github.io/)。



**执行**

```
nohup ./clash -d ~/.config/clash > mihomo.log 2>&1 &
```





运行这个命令看看能不能获取所有节点组：

```bash
curl http://127.0.0.1:9090/proxies
```

如果你看到返回了 JSON 数据，比如有 `"🚀 节点选择"`，说明控制端口没问题。



**测试节点**

你可以使用以下命令测试代理是否正常：

```bash
curl -x http://127.0.0.1:7890 https://www.google.com
```

如果返回了 HTML 内容，说明代理设置成功。





**切换节点**

```
curl -X PUT "http://127.0.0.1:9090/proxies/GLOBAL" \
  -H "Content-Type: application/json" \
  -d '{"name": "🇭🇰 香港Z01"}'
```

比如你要切换到 “🇭🇰 香港01”，就运行下面的命令：

```bash
curl -X PUT http://127.0.0.1:9090/proxies/🚀%20节点选择 \
  -H "Content-Type: application/json" \
  -d '{"name": "🇭🇰 香港01"}'
```

在终端中执行：

```
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

> 如果你是用 `zsh` 或 `bash`，这两条命令都可以直接执行。

测试是否生效：

```
curl -I https://www.google.com
```

如果返回 `HTTP/1.1 200 Connection established`，说明走代理成功。

如果你只想让某个命令走代理，可以这样写：

```
http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 curl https://www.google.com
```

或者：

```
--proxy http://127.0.0.1:7890
```





```
curl -x http://127.0.0.1:7890 https://ipinfo.io
```

```
# 方式 1（推荐）
curl -x http://127.0.0.1:7890 https://api.ip.sb/ip

# 方式 2
curl -x http://127.0.0.1:7890 https://ifconfig.me

# 方式 3
curl -x http://127.0.0.1:7890 https://ipapi.co/json

# 方式 4
curl -x http://127.0.0.1:7890 https://ipinfo.io/ip  # 只显示IP，有时仍可用
```

