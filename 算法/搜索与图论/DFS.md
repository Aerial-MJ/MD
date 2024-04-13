# DFS

## 排列数字

```c++
#include<iostream>
#include<cstring>
#include<algorithm>

using namespace std;

int n;

void dfs(int p[],int n,int idx){
    if(idx==n)
        return;
    if(idx==n-1){
        for(int i=0;i<n;i++) cout<<p[i]<<' ';
        cout<<endl;
    }
    for(int i=idx;i<n;i++){
        swap(p[idx],p[i]);
        dfs(p,n,idx+1);
        swap(p[idx],p[i]);
    }
    
}

int main(){
    cin>>n;
    
    int p[n];
    for(int i=0;i<n;i++){
        p[i]=i+1;
    }
    
    dfs(p,n,0);
    
    return 0;
    
}
```
**字典序排列**

```c++
#include<iostream>
#include<cstring>
#include<algorithm>

using namespace std;

int n;
const int N=10;
int path[N];
int st[N];

void dfs(int idx,int n){
    if(idx==n) {
        for(int i=0;i<n;i++) cout<<path[i]<<' ';
        cout<<endl;
        return ;
    }
    else{
        for(int i=0;i<n;i++){
            if(!st[i]){
                path[idx]=i+1;
                st[i]=1;
                dfs(idx+1,n);
                st[i]=0;
            }
        }
        
    }
}

int main(){
    memset(st,0,sizeof(st));
    cin>>n;
    
    dfs(0,n);
    return 0;
    
}
```

### 子集树和排列树

**子集树**就是一个确定的集合中的元素是**有还是没有**，即可以理解为**0101的属性串**。

**排列树**就是所有元素都有，并且将这些元素拿来排成特定的序列。

## N皇后问题

```c++
#include<iostream>
#include<algorithm>
#include<cstring>

using namespace std;

int n;
const int N=11;
int p[N][N];
bool col[N],dig[2*N],undig[2*N];

void dfs(int n,int idx){//idx模拟行
    if(n==idx){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(p[i][j]==1) cout<<'Q';
                else cout<<'.';
            }
            cout<<endl;
        }
        cout<<endl;
        return;
    }
    
    else{
        for(int i=0;i<n;i++){//模拟列
            if(!col[i]&&!dig[i+idx]&&!undig[n - idx + i]){
                col[i]=dig[i+idx]=undig[n - idx + i]=true;
                p[idx][i]=1;
                dfs(n,idx+1);
                col[i]=dig[i+idx]=undig[n - idx + i]=false;
                p[idx][i]=0;
            }
            
        }
    }
}
int main(){
    cin>>n;
    memset(p,0,sizeof(p));
    
    dfs(n,0);
    
    return 0;
}
```

## 二叉树的遍历

二叉树的遍历分为以下三种：

先序遍历：遍历顺序规则为【根左右】

中序遍历：遍历顺序规则为【左根右】

后序遍历：遍历顺序规则为【左右根】

