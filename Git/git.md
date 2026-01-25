## git

### 远程分支连接

```git
git init  //初始化本地仓库
git remote add origin https://github.com/xxxxx/TA.git //连接远程仓库 origin（远程仓库的源头）
git remote rm origin //删除远程分支连接
git remote -v   //查看远程分支连接
```

### git分支操作

```git
git branch -r //远程分支
git branch -a //所有分支
git branch  //本地分支
git branch ** //创建分支

//创建的分支会以你目前的分支为基础，你不用commit，你创建的分支会是你现有分支的基础，当然你现有分支没有commit的数据，新分支也不会有。
//删除本地分支
git branch -d  local_branch_name

//删除远程分支
git push origin --delete [branch_name]
或者推送空分支到远程（删除远程分支另一种实现）git push origin :远程分支

git checkout -b ** 创建并切换
git checkout ** //切换分支

git merge branch
//git merge：合并分支。
//操作：切换到master分支，然后输入git merge a命令，将a分支合并到master分支
```

### git提交项目

```git
git status //本地仓库状态
git add .  //将所有的文件添加到本地的仓库上
git commit -m "你的叙述信息"   //将项目提交到本地项目中
git push (-u) origin master:master   //push第一次push需要-u，将本地项目提交到远程分支上（本地：远程）

git pull origin master:master
//git pull等于git fetch 和 git merge
//将远程分支的master拉到本地的master分支上，并且将远程分支的内容和本地分支的内容进行合并（远程：本地）
git fetch
//每次pull之前都要先看一下仓库的状态

#使用git拉代码时可以使用 -b 指定分支
#指定拉 master 分支代码
git clone -b master https://github.com/xxxxx/TA.git
```

### git fetch

`git fetch` 是 Git 的一个命令，它可以从远程仓库中下载最新的变更并将其存储到本地仓库中。这个命令**不会自动将远程仓库的变更与本地分支合并**，而是只会更新远程跟踪分支以反映远程仓库中的新变更。

下面是该命令的用法：

- `git fetch`：这个命令会从你指定的远程仓库中获取所有的变更。它不会将这些变更与你的本地分支合并，只会将它们下载到本地仓库中。
- `git fetch <remote>`：这个命令会从你指定的特定远程仓库中获取所有的变更。例如，`git fetch origin` 将会从名为 'origin' 的远程仓库中获取所有的变更。
- `git fetch <remote> <branch>`：这个命令会从你指定的特定远程仓库的特定分支中获取所有的变更。例如，`git fetch origin main` 将会从名为 'origin' 的远程仓库的 'main' 分支中获取所有的变更。

### github fork

如果你fork其他人的项目（相当于你将人家的项目复制到你的git上，然后自行再做修改），如果你觉得自己修改的不错，想向源代码的主人提供你自己修改的代码，你需要向fork的源头git仓库提交pull request命令。

![image-20230413184008827](../Image/image-20230413184008827.png)

源头git仓库觉得你修改的不错，并且执行git pull命令拉取你修改的源文件。这样源代码主人就认可了你的修改，并将他的源代码pull成了你修改的样子。

"Fork（分支）"是软件开发和版本控制系统（如Git）中使用的术语。它涉及创建一个代码库的新副本，通常是从远程存储库，并独立于原始代码库进行更改。

当用户fork（分支）一个存储库时，他们实际上在自己的帐户下创建了代码库的新副本。这使他们可以在不影响原始代码库或其他开发人员工作的情况下更改代码。分支后的存储库是原始存储库的精确副本，包括所有文件、分支和提交历史记录。

当开发人员想要为开源项目做出贡献或与其他开发人员协作时，通常会fork（分支）存储库。他们可以在分支存储库中更改代码，然后提交拉取请求到原始项目，这使得原始项目的维护者可以审查更改并可能将其合并到主代码库中。

## 远程分支和本地分支

你提到的这两个命令确实和 `git` 中的远程仓库管理有关，涉及到 **远程分支** 和 **本地分支** 的关系。让我们一步一步分析一下。

### 1. **`git push -u origin master:master`**

这个命令的作用是将 **本地的 `master` 分支** 推送到 **远程的 `master` 分支**。

- **`git push`**: 将本地仓库的更改推送到远程仓库。
- **`-u`**: 这个选项告诉 Git 记住这个推送设置，使得以后可以简化为 `git push`（不用每次都指定远程仓库和分支）。简单来说，`-u` 设置了“追踪”关系，之后你可以直接使用 `git push` 和 `git pull`，Git 会自动推送和拉取你设置的远程分支。
- **`origin`**: 是远程仓库的名称，默认情况下，Git 使用 `origin` 作为远程仓库的名称。你可以通过 `git remote -v` 来查看远程仓库的详细信息。
- **`master:master`**: 左边是本地分支（`master`），右边是远程分支（`master`）。这表示将本地的 `master` 分支推送到远程的 `master` 分支。

实际上，你推送到 `origin/master` 就是将本地 `master` 分支的内容推送到远程 `origin` 仓库的 `master` 分支。

### 2. **`git pull origin master:master`**

这个命令的作用是从 **远程的 `master` 分支** 拉取最新的内容，并将其合并到 **本地的 `master` 分支**。

- **`git pull`**: 等价于 `git fetch` 和 `git merge`。首先，它会从远程仓库拉取（`fetch`）最新的代码，然后会尝试将远程分支的更改与本地分支进行合并（`merge`）。
- **`origin`**: 同样是指远程仓库的名称，表示你要从 `origin` 仓库拉取数据。
- **`master:master`**: 这里的意思是从远程仓库的 `master` 分支拉取数据并合并到本地的 `master` 分支。

### 3. **关于 `origin/master`**

- **`origin/master`** 是指 **远程仓库 `origin` 上的 `master` 分支**。它是本地 Git 仓库对远程仓库 `master` 分支的一个引用。
- `origin/master` 不代表本地的 `master` 分支，而是表示远程 `origin` 上的 `master` 分支的状态（通常是 Git 自动管理的“远程跟踪分支”）。
- 你本地的 `master` 分支可以通过执行 `git fetch` 或 `git pull` 来更新为远程仓库的 `master` 分支（也就是更新 `origin/master`）。

### 总结

- `origin/master` 是远程 `origin` 仓库中的 `master` 分支的**本地引用**，它通常用来跟踪**远程仓库 `master` 分支的状态**。
- `git push origin master:master` 是将本地 `master` 分支的内容推送到远程 `origin` 仓库的 `master` 分支。
- `git pull origin master:master` 是从远程 `origin` 仓库的 `master` 分支拉取并合并到本地 `master` 分支。

## 分支合并原则

### 1. **将本地的其他分支推送到远程的其他分支**

假设你在本地有一个分支 `feature-xyz`，并且你想将它推送到远程的 `dev` 分支。你可以使用以下命令：

```bash
git push origin feature-xyz:dev
```

**解释**：

- **`origin`**: 远程仓库的名字（默认是 `origin`）。
- **`feature-xyz`**: 本地分支的名字。
- **`dev`**: 远程仓库的目标分支。

这样，Git 会将你本地 `feature-xyz` 分支的内容推送到远程 `dev` 分支。需要注意的是，远程 `dev` 分支上的内容会被本地的 `feature-xyz` 分支内容覆盖，除非你有合并或其他操作。

如果远程仓库没有 `dev` 分支，Git 会创建一个新的 `dev` 分支。

### 2. **拉取并合并分支（`git pull`）遵循什么原则？**

#### **`git pull` 的工作原理**

`git pull` 其实是 **`git fetch`** + **`git merge`**，它会首先从远程仓库拉取最新的提交，并尝试将这些提交与本地分支合并。

- **`git pull origin <branch>`**: 将远程仓库 `origin` 上的 `<branch>` 分支内容拉取到当前本地分支并合并。

例如：

```bash
git pull origin dev
```

这会从远程仓库 `origin` 的 `dev` 分支拉取内容，并将它们合并到你当前的本地分支中。

#### **合并时遵循的原则**

1. **快速前进合并（Fast-forward Merge）**：
   如果本地分支落后于远程分支（没有任何本地的独立提交），Git 会做一个快速前进合并。这意味着本地分支的 `HEAD` 会直接跳到远程分支的最新提交，没有实际的合并。

   例如，如果本地 `master` 分支落后于远程 `master`，Git 会把本地分支直接指向远程的最新提交：

   ```bash
   git pull origin master
   ```

2. **非快速前进合并（Non-fast-forward Merge）**：
   如果你本地的分支有新提交，而远程分支也有新的提交（这时你和其他人可能都有自己的修改），Git 会进行普通的合并。Git 会将本地和远程的修改合并成一个新的提交（merge commit）。

   例如，`feature-xyz` 本地分支和远程的 `dev` 分支有各自的提交，你可以使用 `git pull origin dev` 将远程的内容拉取到本地并自动进行合并。

   在这种情况下，Git 会尝试自动合并两者的内容。如果没有冲突，Git 会自动创建一个合并提交（merge commit）。如果有冲突，Git 会提示你解决冲突。

3. **拉取不同分支合并**：
   如果你想将某个远程分支的内容合并到另一个本地分支，Git 会将这个远程分支的内容与当前分支合并。例如，你正在本地的 `feature-xyz` 分支上工作，而想要拉取远程 `dev` 分支的最新内容：

   ```bash
   git checkout feature-xyz
   git pull origin dev
   ```

   这会将远程 `dev` 分支的内容拉取到本地的 `feature-xyz` 分支中，并尝试合并它们。

#### **如何解决合并冲突**

在拉取并合并时，如果 Git 无法自动合并（即发生冲突），它会提示你手动解决冲突。解决冲突后，你需要执行以下步骤：

1. 手动修改冲突文件。
2. 使用 `git add <file>` 标记冲突已解决。
3. 最后，使用 `git commit` 提交合并结果。

```bash
git add <file>  # 添加解决冲突后的文件
git commit      # 完成合并提交
```

### 3. **总结**

- **推送到远程其他分支**：使用 `git push origin <local-branch>:<remote-branch>` 将本地的某个分支推送到远程的其他分支。

  例如：

  ```bash
  git push origin feature-xyz:dev
  ```

- **拉取并合并分支**：使用 `git pull origin <branch>` 将远程的 `<branch>` 分支拉取并合并到本地分支。Git 会根据本地分支和远程分支的关系决定是做快速前进合并，还是普通合并。

  例如：

  ```bash
  git pull origin dev
  ```

- **合并冲突**：如果发生合并冲突，Git 会提示你手动解决冲突，解决后提交合并结果。

希望这些解释能帮你更清楚地理解 `git push` 和 `git pull` 的使用，以及它们如何影响本地和远程分支的合并操作！

