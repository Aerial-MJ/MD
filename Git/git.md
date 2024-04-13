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