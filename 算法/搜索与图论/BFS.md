# BFS

普通队列`queue`都是**从队尾进队，队首出队**

所以`queue`的`STL`操作是

```c++
#include<queue>
#include<iostream>

using namespace std;

int main(){
    queue<int> que;
    
    q.front();//返回queue头部的引用
    q.back();//返回queue尾部的引用
    q.push(x);//向queue的尾部添加元素
    q.pop();//删除queue的头部元素
    q.size();
    q.empty();
    
}
```

## 走迷宫

```c++
#include<iostream>
#include<cstring>
#include<algorithm>
#include<queue>
using namespace std;

const int N=110;

typedef pair<int,int> PII;

int g[N][N],d[N][N];//d[x][y]用来存储(x,y)这一点到坐标原点的最短距离
int n,m;

int bfs(){
    queue<PII> q; //用来暂时储存广度优先搜索的路径
    memset(d,-1,sizeof(d));
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
    d[0][0] = 0;
    q.push({0, 0});
    
    while(q.size()){
        auto t = q.front();
        q.pop();
        
        for (int i = 0; i < 4; i ++ ){
            int x = t.first + dx[i], y = t.second + dy[i];
            
            if (x >= 0 && x < n && y >= 0 && y < m && g[x][y] == 0 && d[x][y] == -1){
                d[x][y] = d[t.first][t.second] + 1;
                q.push({x, y});
            }
            
        }
    }
    return d[n-1][m-1];
}


int main(){
    cin>>n>>m;
    
    
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < m; j ++ )
            cin >> g[i][j];

    cout << bfs() << endl;
    
    return 0;
}
```

## 八数码/拼图

```c++
#include<iostream>
#include<cstring>
#include<algorithm>
#include<unordered_map>
#include<queue>

using namespace std;

void bfs(string state){
    string end = "12345678x";
    
    queue<string> q;
    unordered_map<string, int> d; //该路径所走的长度

    q.push(state);
    d[state] = 0;
    
    int dx[4]={-1,0,1,0};
    int dy[4]={0,1,0,-1};
    
    while(q.size()){
        //引用
        string t=q.front();
        q.pop();
        int distance=d[t];
        if(t==end) {
            cout<<distance;
            return ;
        }
        int pos=t.find('x');
        for(int i=0;i<4;i++){
            int x=pos/3;
            int y=pos%3;
            x+=dx[i];
            y+=dy[i];
            if(x>=0&&x<3&&y>=0&&y<3){
                swap(t[pos],t[x*3+y]);
                if(!d.count(t)){
                    //第一次永远是最短的距离，因为这是广度优先搜索
                    d.insert({t,distance+1});
                    q.push(t);
                }
                swap(t[pos],t[x*3+y]);
            }
        }
    }
    cout<<"-1"<<endl;
    return ;
}

int main(){
    char s;    
    string state;    
    for(int i=0;i<9;i++){
        cin>>s;
        state+=s;
    }
    
    bfs(state);
    return 0;
    
}


```

