# Hash

## 模拟散列表

<[840. 模拟散列表 - AcWing题库](https://www.acwing.com/problem/content/description/842/)>

### 开放寻址法

```c++
#include <cstring>
#include <iostream>

using namespace std;

//使用质数，减少hash冲突
const int N = 200003, null = 0x3f3f3f3f;

int h[N];

int find(int x)
{
    int t = (x % N + N) % N;
    while (h[t] != null && h[t] != x)
    {
        t ++ ;
        if (t == N) t = 0;
    }
    return t;
}

int main()
{
    memset(h, 0x3f, sizeof h);

    int n;
    scanf("%d", &n);

    while (n -- )
    {
        char op[2];
        int x;
        scanf("%s%d", op, &x);
        if (*op == 'I') h[find(x)] = x;
        else
        {
            if (h[find(x)] == null) puts("No");
            else puts("Yes");
        }
    }

    return 0;
}

```

### 拉链法

```c++
#include <cstring>
#include <iostream>

using namespace std;

const int N = 100003;

int h[N], e[N], ne[N], idx;

void insert(int x)
{
    int k = (x % N + N) % N;
    e[idx] = x;
    ne[idx] = h[k];
    h[k] = idx ++ ;
}

bool find(int x)
{
    int k = (x % N + N) % N;
    for (int i = h[k]; i != -1; i = ne[i])
        if (e[i] == x)
            return true;

    return false;
}

int main()
{
    int n;
    scanf("%d", &n);

    memset(h, -1, sizeof h);

    while (n -- )
    {
        char op[2];
        int x;
        scanf("%s%d", op, &x);

        if (*op == 'I') insert(x);
        else
        {
            if (find(x)) puts("Yes");
            else puts("No");
        }
    }

    return 0;
}

```

### 直接使用unordered_set

```c++
#include<iostream>
#include<algorithm>
#include<cstring>
#include<unordered_set>

using namespace std;

int main(){
    int n;
    cin>>n;
    unordered_set<int> p;
    for(int i=0;i<n;i++){
        char x;
        int y;
       
        cin>>x>>y;
        if(x=='I'){
            p.insert(y);
        }
        else{
            cout<<(p.count(y)==1?"Yes":"No"); 
            cout<<endl;
        }
    }
    
    return 0;
}
```

## 字符串Hash

```c++
#include <iostream>
#include <algorithm>

using namespace std;

typedef unsigned long long ULL;

const int N = 100010, P = 131;

int n, m;
char str[N];
ULL h[N], p[N];


ULL get(int l, int r)
{
    return h[r] - h[l - 1] * p[r - l + 1];
}

int main()
{
    scanf("%d%d", &n, &m);
    scanf("%s", str + 1);

    p[0] = 1;
    
    //构建hash表
    for (int i = 1; i <= n; i ++ )
    {
        h[i] = h[i - 1] * P + str[i];
        p[i] = p[i - 1] * P;
    }

    while (m -- )
    {
        int l1, r1, l2, r2;
        scanf("%d%d%d%d", &l1, &r1, &l2, &r2);

        if (get(l1, r1) == get(l2, r2)) puts("Yes");
        else puts("No");s
    }

    return 0;
}
```

1. 把字符串看成是一个 P 进制数，每个字符的 ASCII 码对应数的一位
2. ASCII 范围 0 - 127，最少 128 进制，经验上取 131 或 13331 冲突率低
3. 字符串很长，对应的数太大，通过模 2^64 把它映射到 [0, 2^64 - 1]
4. 用 unsigned long long 存储，溢出相当于对 2^64 取模，省略了手动运算
5. 该方法的好处是，可以利用前缀哈希直接求出子串哈希（减去高位）

![image-20230401160355411](C:/Users/19409/AppData/Roaming/Typora/typora-user-images/image-20230401160355411.png)

## Pair重写hash函数并创建unordered_map

```cpp
struct Hash{
  template<typename T,typename U>
  size_t operator()(const pair<T,U>& A) const{
      return hash<T>()(A.first) ^ hash<T>()(A.second);
  }
};

unordered_map<pair<int,int>,int,Hash> mp;
```

