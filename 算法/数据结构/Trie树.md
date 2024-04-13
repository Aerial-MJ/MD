# Trie树

维护一个字符串集合，支持两种操作：

1. `I x` 向集合中插入一个字符串 x；
2. `Q x` 询问一个字符串在集合中出现了多少次。

共有 N 个操作，所有输入的字符串总长度不超过 10<sup>5</sup>，字符串仅包含小写英文字母。

```cpp
#include <iostream>

using namespace std;

const int N = 100010;

int son[N][26], cnt[N], idx;
char str[N];

void insert(char *str)
{
    int p = 0;
    //p相当于root，son[0][]的位置
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++ ;
}

int query(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}

int main()
{
    int n;
    scanf("%d", &n);
    while (n -- )
    {
        char op[2];
        scanf("%s%s", op, str);
        if (*op == 'I') insert(str);
        else printf("%d\n", query(str));
    }

    return 0;
}

```

**自己的代码**

```cpp
#include<iostream>
#include<cstring>
#include<algorithm>

using namespace std;

const int N=1e5+10;

int son[N][26],cnt[N],idx;

void insert(string str){
    int p=0;
    
    for(int i=0;i<str.size();i++){
        int u=str[i]-'a';
        if(!son[p][u]) son[p][u]=++idx;
        p=son[p][u];
    }
    cnt[p]++;
}

void query(string str){
    int p=0;
    
    for(int i=0;i<str.size();i++){
        int u=str[i]-'a';
        if(!son[p][u]){
            cout<<0<<endl;
            return;
        }
        else{
            p=son[p][u];
        }
    }
    
    cout<<cnt[p]<<endl;
    return;
}

int main(){
    int n;
    cin>>n;
    char c;
    string str;
    
    for(int i=0;i<n;i++){
        cin>>c>>str;
        
        if(c=='I') insert(str);
        else{
            query(str);
        }
    }
    
    return 0;
}
```

