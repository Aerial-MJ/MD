# `origin` 是什么意思？

## 简单理解

`origin` 就是**远程仓库的别名/昵称**。

你的代码仓库存在两个地方：

```
你的电脑（本地）          远程服务器（比如 GitLab/GitHub）
┌─────────────┐          ┌─────────────────────┐
│  本地仓库    │  ←────→  │  远程仓库            │
│             │          │  这个远程仓库的       │
│             │          │  别名就叫 "origin"   │
└─────────────┘          └─────────────────────┘
```

> 就像你给同事存了个手机号，备注叫"张三"，`origin` 就是你给远程服务器起的备注名。

---

## 为什么叫 origin？

这是 Git 的**默认命名约定**，当你 `git clone` 一个仓库时，Git 自动把远程地址命名为 `origin`。

你可以验证一下：
```bash
git remote -v
```
会输出类似：
```
origin  http://gitlab.xxx.com/yourteam/yourproject.git (fetch)
origin  http://gitlab.xxx.com/yourteam/yourproject.git (push)
```
这就是 `origin` 实际指向的地址。

---

## 为什么有的有 `remotes/origin/`，有的没有？

| 显示形式 | 类型 | 在哪里 | 能直接工作吗 |
|---|---|---|---|
| `* feature/dev_chenjiarui` | **本地分支** | 你自己电脑上 | ✅ 可以直接提交、改代码 |
| `master` | **本地分支** | 你自己电脑上 | ✅ 可以直接提交、改代码 |
| `remotes/origin/master` | **远程分支镜像** | 远程服务器上 | ❌ 只能看，不能直接在上面工作 |

---

## 用生活类比理解

```
本地分支                    远程分支 (remotes/origin/xxx)
─────────────────────────────────────────────────────
你手里的文件副本             服务器上同事的文件

feature/dev_chenjiarui  ←→  remotes/origin/feature/dev_chenjiarui
master                  ←→  remotes/origin/master
```

- **本地分支** = 你桌上的草稿纸，随时可以写
- **远程分支** = 服务器上的存档，需要 `pull` 拉下来才能用

---

## 你的情况具体解释

```bash
* feature/dev_chenjiarui        ← 你本地的开发分支（你现在在这里）
  master                        ← 你本地的 master 分支

  remotes/origin/master         ← 远程服务器上的 master
  remotes/origin/dev_tjh        ← 同事 tjh 推上去的远程分支
  remotes/origin/feature/dev_chenjiarui  ← 你自己推到远程的备份
```

所以当你执行：
```bash
git pull origin master
#          ↑      ↑
#        别名    远程分支名
#      (远程服务器)
```
意思就是：**从叫 `origin` 的那个远程服务器，把 `master` 分支的代码拉下来合并到我当前分支**。


## 合并分支
你需要把远程其他分支的代码合并到你当前分支。常用操作如下：

### 1. 先查看有哪些远程分支

```bash
git branch -r
```

找到你想要合并的远程分支，比如 `origin/dev` 或 `origin/main`。

### 2. 合并远程分支到当前分支

假设你想合并 `origin/dev`：

```bash
git pull origin dev
```

这相当于 `git fetch origin dev` + `git merge origin/dev`，会自动合并。

### 3. 如果只想查看某个分支的改动，不合并

```bash
git log origin/dev --oneline -10
```

### 4. 如果有冲突

合并时如果提示冲突，需要手动解决：

```bash
# 查看冲突文件
git status

# 编辑冲突文件，解决后
git add .
git commit
```

---

> **提示**：你之前 `git pull` 显示 `Already up to date.`，是因为你当前分支对应的远程分支没有新提交。其他分支的更新需要显式指定分支名来拉取合并。



## 理解 Git 分支列表

你的分支分为两类：

### 本地分支（没有 `remotes/` 前缀）

```
* feature/dev_chenjiarui    ← 星号表示你当前所在的分支
  master
```

这是你本地有的分支，`*` 标记当前分支。

### 远程分支（以 `remotes/origin/` 开头）

这些是远程仓库（origin）上的分支，是其他同事的分支。**它们只是远程的"镜像"，你本地不能直接切换到上面工作。**

---

## 你想合并别人的代码，怎么做？

假设你想把 `master` 的最新代码合并到你当前的 `feature/dev_chenjiarui` 分支：

```bash
# 第1步：先更新远程信息
git fetch

# 第2步：把 origin/master 合并到你当前分支
git pull origin master
```

或者如果你想合并某个同事的分支，比如 `remotes/origin/dev_tjh`：

```bash
git pull origin dev_tjh
```

---

## 常用场景速查

| 需求 | 命令 |
|------|------|
| 把 master 最新代码合到我的分支 | `git pull origin master` |
| 把某同事的分支合到我的分支 | `git pull origin dev_tjh` |
| 切换到本地的 master 分支 | `git checkout master` |
| 基于远程分支创建本地分支 | `git checkout -b my_feature origin/feature/dev_gaahou` |

---

## 简单理解

```
你当前在:  feature/dev_chenjiarui (本地)
                │
                │  git pull origin master
                │  (把 master 的代码拉过来合并)
                ▼
          origin/master (远程)
```

> **最常见的需求**就是定期执行 `git pull origin master`，把主分支的最新代码同步到你的开发分支，避免落后太多导致最终合并时冲突太大。


```
没听懂，我现在
* feature/dev_chenjiarui
  master
  remotes/origin/HEAD -> origin/master
  remotes/origin/cr
  remotes/origin/dev/xuyasheng
  remotes/origin/dev/xuyasheng-malcop-adrrmining
  remotes/origin/dev_chenhang18
  remotes/origin/dev_chenxinrui03
  remotes/origin/dev_dingshuai
  remotes/origin/dev_gty
  remotes/origin/dev_huaminyao
  remotes/origin/dev_jixianpeng
  remotes/origin/dev_jyj
  remotes/origin/dev_lijinku03
  remotes/origin/dev_lijunlang
  remotes/origin/dev_shahongzhou
  remotes/origin/dev_tjh
  remotes/origin/dev_tjh_backup
  remotes/origin/dev_tjh_tmp
  remotes/origin/dev_xuyasheng
  remotes/origin/dev_zcr
  remotes/origin/dev_zhangmengyang
  remotes/origin/dev_zhangshuai77
  remotes/origin/dev_zhaomingming
  remotes/origin/develop/cyh
  remotes/origin/develop/dev_heshiyong
  remotes/origin/develop/dev_lulv
  remotes/origin/develop/gaahou
  remotes/origin/develop/gongkaiyi
  remotes/origin/develop/mijiaxuan
  remotes/origin/develop/shanhaijiao
  remotes/origin/feature/chenxuemei
  remotes/origin/feature/dev/gaahou
  remotes/origin/feature/dev_chenjiarui
  remotes/origin/feature/dev_chenxuemei
  remotes/origin/feature/dev_gaahou
  remotes/origin/feature/dev_tjh_test
  remotes/origin/feature/mijiaxuan
  remotes/origin/feature/postage_xgb
  remotes/origin/feature/quyu_feat
  remotes/origin/feature/replace
  remotes/origin/feature/sunchuchu
  remotes/origin/feature/xuyasheng/mytask-qianyi
  remotes/origin/hotfix/code-cmt
  remotes/origin/hotfix/dev_chenhang18
  remotes/origin/hotfix/emptyfix
  remotes/origin/hotfix/modify-upstream
  remotes/origin/hotfix/poi-cluster-bug-fix
  remotes/origin/hotfix/poipairfix
  remotes/origin/hotfix/usrdensity
  remotes/origin/master
  remotes/origin/wangzhiyu_test
  remotes/origin/xuyaheng/data_task_all
  remotes/origin/xuyasheng/AllCateUserMarkov
  remotes/origin/xuyasheng/all_cate_iforest
  remotes/origin/xuyasheng/cate-modify
  remotes/origin/xuyasheng/category-modify
  remotes/origin/xuyasheng/crowd-geohash6
  remotes/origin/xuyasheng/dev_jyj
  remotes/origin/xuyasheng/grid-fq-mining
  remotes/origin/xuyasheng/jh-old-task-yh
  remotes/origin/xuyasheng/jh_if_model_optimization
  remotes/origin/xuyasheng/model-for-shangou
  remotes/origin/xuyasheng/model_for_shangou
  remotes/origin/xuyasheng/poi-community-detection
  remotes/origin/xuyasheng/refund
  remotes/origin/xuyasheng/replace-data-source
  remotes/origin/xuyasheng/sku_cluster_result
  remotes/origin/xuyasheng/temp
  remotes/origin/xuyasheng/temp-x
  remotes/origin/xuyasheng/test
  remotes/origin/xuyasheng/tmp
  remotes/origin/xuyasheng/tx_antispam_xgb_model
  remotes/origin/xuyasheng/user-together-buy
  remotes/origin/xuyasheng/usrdensity
* feature/dev_chenjiarui
  master
  remotes/origin/HEAD -> origin/master
  remotes/origin/cr
  remotes/origin/dev/xuyasheng
  remotes/origin/dev/xuyasheng-malcop-adrrmining
  remotes/origin/dev_chenhang18
  remotes/origin/dev_chenxinrui03
  remotes/origin/dev_dingshuai
  remotes/origin/dev_gty
  remotes/origin/dev_huaminyao
  remotes/origin/dev_jixianpeng
  remotes/origin/dev_jyj
  remotes/origin/dev_lijinku03
  remotes/origin/dev_lijunlang
  remotes/origin/dev_shahongzhou
  remotes/origin/dev_tjh
  remotes/origin/dev_tjh_backup
  remotes/origin/dev_tjh_tmp
  remotes/origin/dev_xuyasheng
  remotes/origin/dev_zcr
  remotes/origin/dev_zhangmengyang
  remotes/origin/dev_zhangshuai77
  remotes/origin/dev_zhaomingming
  remotes/origin/develop/cyh
```


# 安装torch
## 逐一解答

### 1. `--index-url` vs `--extra-index-url` 的区别

| 参数 | 含义 |
|------|------|
| `--index-url` | **主包索引**，pip 优先从这里找包 |
| `--extra-index-url` | **额外包索引**，pip 同时也从这里找包 |

**问题在于**：你的 `pip.conf` 里配了内网源作为默认 index，pip 会**同时**搜索两个源。所以你虽然命令行指定了 `--index-url https://download.pytorch.org/whl/cu121`，pip 还是会去内网源找，就出现了你看到的日志里大量请求走 `data-source-pip.sankuai.com` 的情况。

> **最简单的解决办法**就是临时把 `pip.conf` 改名，装完再改回来：
> ```bash
> mv ~/.pip/pip.conf ~/.pip/pip.conf.bak
> pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
>   --index-url https://download.pytorch.org/whl/cu121 --timeout 300
> mv ~/.pip/pip.conf.bak ~/.pip/pip.conf
> ```

### 2. `--no-cache-dir` 是什么意思

pip 下载的包会缓存到 `~/.cache/pip/` 目录。`--no-cache-dir` 表示**不使用缓存**，每次都重新下载。

- **好处**：避免缓存损坏导致的问题
- **对你来说**：其实**建议去掉这个参数**，保留缓存反而好——如果下载中断，缓存的半截文件虽然不能直接用，但 pip 可能会利用已下载的部分

### 3. `--no-build-isolation` 是什么意思

pip 安装某些包时，会创建一个**临时隔离环境**来构建（编译）包。`--no-build-isolation` 禁用这个隔离机制，直接用当前环境的构建工具。

- **对你来说**：PyTorch 的 whl 是预编译好的，不需要构建，**这个参数对安装 PyTorch 没什么影响**，可以去掉。

### 4. 能不能断点续传？

**pip 本身不支持断点续传**，这是它最大的痛点之一。大文件下载中断就只能从头再来。

替代方案：

| 方案 | 是否支持断点续传 |
|------|:---:|
| `pip install` | ❌ |
| `wget -c` 下载 whl 再本地安装 | ✅ |
| `curl -C -` 下载 whl 再本地安装 | ✅ |
| conda install | ✅（部分支持） |

**推荐做法**：用 `wget` 或 `curl` 手动下载 whl 文件（支持断点续传），然后本地安装：

```bash
# 用 wget 下载（-c 表示断点续传）
wget -c https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp312-cp312-linux_x86_64.whl

wget -c https://download.pytorch.org/whl/cu121/torchvision-0.20.1%2Bcu121-cp312-cp312-linux_x86_64.whl

wget -c https://download.pytorch.org/whl/cu121/torchaudio-2.5.1%2Bcu121-cp312-cp312-linux_x86_64.whl

# 下载完成后本地安装
pip install torch-2.5.1+cu121-cp312-cp312-linux_x86_64.whl
pip install torchvision-0.20.1+cu121-cp312-cp312-linux_x86_64.whl
pip install torchaudio-2.5.1+cu121-cp312-cp312-linux_x86_64.whl
```

> 中断了就再执行一次同样的 `wget -c` 命令，会从断点继续下载，不用从头来！


## 安装 torch  torchvision  torchaudio
# 用 wget 下载（-c 表示断点续传）
wget -c https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp312-cp312-linux_x86_64.whl

wget -c https://download.pytorch.org/whl/cu121/torchvision-0.20.1%2Bcu121-cp312-cp312-linux_x86_64.whl

wget -c https://download.pytorch.org/whl/cu121/torchaudio-2.5.1%2Bcu121-cp312-cp312-linux_x86_64.whl

# 下载完成后本地安装
pip install torch-2.5.1+cu121-cp312-cp312-linux_x86_64.whl --no-deps

pip install torchvision-0.20.1+cu121-cp312-cp312-linux_x86_64.whl --no-deps

pip install torchaudio-2.5.1+cu121-cp312-cp312-linux_x86_64.whl --no-deps


``` bash
# 查看Mac本机ip 找到inet xx后面的xx 复制该ip
ifconfig en0
# 1.有外网访问权限的机器开代理端口(比如连入公司Wifi的Mac办公机）
sudo pip install proxy.py  
proxy --port 8420 --hostname 0.0.0.0
# 2.codelab上的机器实际上是pord环境 需要添加 代理机的ip 推荐公司网络环境的Mac的ip  ip查询命令：ifconfig en0
# 记得修改ip
export http_proxy=http://xx:8420
export https_proxy=https://xx:8420

#验证
echo $http_proxy
#发送一条请求 测试一下开没开成功
curl "http://jaminzhang.github.io/nginx/Nginx-resolver-DNS-resolve-timed-out-problem-analysis-and-solve/"


# 注：代理情况下有些操作无法实现，比如唯内网可用的hope等命令，需要取消代理
unset http_proxy
unset https_proxy
```