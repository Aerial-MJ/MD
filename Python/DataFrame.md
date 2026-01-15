# DataFrame

```python
import pandas as pd

# 创建一个DataFrame
data = {'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'gender': ['female', 'male', 'male', 'male', 'female'],
        'year': [2017, 2018, 2019, 2020, 2021],
        'math': [88, 92, 94, 90, 86],
        'english': [84, 90, 92, 88, 90],
        'science': [92, 94, 90, 92, 93]}
print(type(data))
df = pd.DataFrame(data)
df = df.set_index(['gender', 'year'])   #设置索引

# df.reset_index()   展平索引

# 输出多层次索引DataFrame
print("结构信息：")
df.info()

print("\n维度：", df.shape)

print("\n列名：", df.columns)

print("\n索引：", df.index)

print("\n前几行数据：")
print(df.head())
```

```tex
<class 'dict'>
结构信息：
<class 'pandas.core.frame.DataFrame'>
MultiIndex: 5 entries, ('female', 2017) to ('female', 2021)
Data columns (total 4 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   name     5 non-null      object
 1   math     5 non-null      int64 
 2   english  5 non-null      int64 
 3   science  5 non-null      int64 
dtypes: int64(3), object(1)
memory usage: 614.0+ bytes

维度： (5, 4)

列名： Index(['name', 'math', 'english', 'science'], dtype='object')

索引： MultiIndex([('female', 2017),
            (  'male', 2018),
            (  'male', 2019),
            (  'male', 2020),
            ('female', 2021)],
           names=['gender', 'year'])

前几行数据：
                name  math  english  science
gender year                                 
female 2017    Alice    88       84       92
male   2018      Bob    92       90       94
       2019  Charlie    94       92       90
       2020    David    90       88       92
female 2021      Eva    86       90       93
```

## if row["label"]

**问题**

```
if row["label"]:
    print("跳过")
    continue
```

如果 `label` 列是 **NaN（空值）**，那么：

```
bool(np.nan) == True
```

也就是说：

- **本来是空值**
- 却被当成 True
- 直接被跳过 ❌

这会导致 **大量未处理的数据被误跳过**

**正确写法**

用 `pd.isna()` 判断：

```
if not pd.isna(row["label"]):
    print("跳过")
    continue
```

或者更严格（防止空字符串）：

```
if pd.notna(row["label"]) and str(row["label"]).strip() != "":
    print("跳过")
    continue
```

下面这些都会被认为是 **NA（缺失值）**：

```
None
np.nan
pd.NA
```

### 判断一个值是不是空

```
import pandas as pd
import numpy as np

pd.isna(None)      # True
pd.isna(np.nan)    # True
pd.isna("")        # False
pd.isna("娱乐")    # False
```

## iterrows

```python
for index, row in df.iterrows(): 
    if row["label"]: 
        print("跳过") 
        continue 
    print(f"==================处理第{index+1}条数据==================") 
    payload = {
        "inputs": { "order_information": row['cleaned_goodsname'] },
        "response_mode": "blocking", 
        "user": "test-user-001" 
    } 
    resp = requests.post(url, headers=headers, json=payload) 
    result = resp.json()
```

Pandas 的 index **从 0 开始**
你说的 **第 125 行（人类视角）= index 124**

```
for index, row in df.iloc[124:].iterrows():
    ...
```

这是 **最干净、最安全** 的方式
不影响原 index
不用改 CSV
随时可调起始行

**如果你说的就是 index=125（不关心人类行号）**

```
for index, row in df.iloc[125:].iterrows():
    ...
```
