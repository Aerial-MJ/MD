# 树状DP

**简单的说就是在树的结构上来进行动态规划**

## 例题

Ural 大学有 N 名职员，编号为 `1∼N`。

他们的关系就像一棵以校长为根的树，父节点就是子节点的直接上司。

每个职员有一个快乐指数，用整数 Hi 给出，其中 `1≤i≤N`。

现在要召开一场周年庆宴会，不过，没有职员愿意和直接上司一起参会。

在满足这个条件的前提下，主办方希望邀请一部分职员参会，使得所有参会职员的快乐指数总和最大，求这个最大值。

```cpp
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 6010;

int n;
int h[N], e[N], ne[N], idx;
int happy[N];
int f[N][2];
bool has_fa[N];

void add(int a, int b){
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u){
    //状态转移f[i][1]   选择了i，且i子树的快乐指数之和
    //状态转移f[i][0]   选择了i，且i子树的快乐指数之和
    //1选择i   0不选择i
    f[u][1] = happy[u];

    for (int i = h[u]; i!=-1; i = ne[i])
    {
        int j = e[i];
        dfs(j);
		
        f[u][1] += f[j][0];
        f[u][0] += max(f[j][0], f[j][1]);
    }
}


int main(){
    cin>>n;

    for (int i = 1; i <= n; i ++ )  cin>>happy[i];

    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i ++ ){
        int a, b;
        cin>>a>>b;
        //b是父节点，a是子节点，因为用链式向量法代表图的话，指向的那个是父节点，这样b向量就可以找到所有的子节点
        add(b, a);
        has_fa[a] = true;
    }

    int root = 1;
    //找到根节点
    while (has_fa[root]) root ++ ;
    //对根节点进行深度搜索
    dfs(root);

    cout<<max(f[root][0], f[root][1]);

    return 0;
}

```



