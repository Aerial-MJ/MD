## 三者的关系

> **Cookie ≠ Session ≠ Token**
>  它们**不在同一个地方**，作用也完全不同。

| 名称       | 存在哪                     | 由谁生成    | 发给谁                 | 常见用途         |
| ---------- | -------------------------- | ----------- | ---------------------- | ---------------- |
| Cookie     | **浏览器本地**             | 服务端 / JS | 每次请求自动带给服务端 | 会话标识、反爬   |
| Session    | **服务端内存/缓存**        | 服务端      | 不直接暴露             | 存用户状态       |
| JSESSIONID | Cookie 里的一个值          | Java 服务端 | 客户端                 | Session 的“钥匙” |
| Token      | 本地存储 / Cookie / Header | 服务端      | 客户端                 | 登录 / API 鉴权  |

## Cookie

### 本质一句话：

> **Cookie 是“客户端带给服务器的小纸条”**

### 存在哪？

- 浏览器：内存 / 磁盘
- requests：`session.cookies`

### 你现在遇到的：

```
document.cookie = "C3VK=011bd1; path=/; max-age=300;"
```

这是：

- **前端 JS 写 Cookie**
- 存在客户端
- 下次请求自动带：

```
Cookie: C3VK=011bd1
```

👉 **你现在绕的反爬，100% 是 Cookie 级别**

## Session

### 本质一句话：

> **Session 是“服务器存储的用户状态”**

### Session 不在浏览器里！

> **Session 永远在服务器**

### 典型流程（Java 网站）：

1. 浏览器第一次访问
2. 服务端创建 Session 对象
3. 生成一个 ID（比如 `ABC123`）
4. 把这个 ID 放进 Cookie发给浏览器

```
Set-Cookie: JSESSIONID=ABC123
```

**以后浏览器每次请求：**

```
Cookie: JSESSIONID=ABC123
```

**服务端：根据JSESSIONID找到对应的Session，以获取用户状态**

> “哦，是你，用 ABC123 找 Session”

## Token

### Token 的本质：

> **“你是谁 + 你能干啥”的一次性凭证**

### 常见形式：

- JWT（`eyJhbGciOi...`）
- UUID
- Base64

### 存哪？

| 场景     | 存储位置                      |
| -------- | ----------------------------- |
| 前端 SPA | localStorage / sessionStorage |
| API      | HTTP Header                   |
| 传统网站 | Cookie                        |

### 典型请求：

```
Authorization: Bearer eyJhbGciOiJIUzI1...
```

### 例子

这是一个 **Token**，更具体地说，是 **JWT（JSON Web Token）**。

**JWT = 服务端签发的一张“数字身份证”**

#### 三段分别是

| 段        | 作用     | 说明                   |
| --------- | -------- | ---------------------- |
| Header    | 算法信息 | 用什么算法签名         |
| Payload   | 用户信息 | 你是谁、权限、过期时间 |
| Signature | 防伪签名 | 防止被篡改             |

#### 假装“解码”一段（不需要密钥）

JWT 的 **前两段只是 Base64**，随便解：

Header（第一段）

```
{
  "alg": "HS256",
  "typ": "JWT"
}
```

Payload（第二段）

```
{
  "userId": 123,
  "role": "admin",
  "exp": 1700000000
}
```

意思是：

> 👤 用户 123
>  🔐 身份：admin
>  ⏰ 1700000000 之前有效

#### 浏览器 / 前端 JS：

```
fetch("/api/user/info", {
  headers: {
    "Authorization": "Bearer " + token
  }
})
```

#### Python requests：

```
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1..."
}

requests.get("https://api.example.com/user/info", headers=headers)
```

👉 **服务端只认这个 Header**

## 在 requests 里怎么“模拟浏览器状态”

### 浏览器的 3 个“存储层”

| 浏览器         | requests 对应                |
| -------------- | ---------------------------- |
| Cookie         | `requests.Session().cookies` |
| sessionStorage | ❌（必须 JS）                 |
| localStorage   | ❌（必须 JS）                 |

### 你现在只需要这一个：

```
session = requests.Session()
session.cookies.set("C3VK", value, domain, path)
```

## 什么时候你才需要 Selenium / Playwright？

只有当：

- Token 在 `localStorage`
- JS 计算签名（hash / crypto）
- 参数绑定时间戳 + 混淆
- Canvas / 指纹校验

👉 **你这个站远远没到这个级别**









































