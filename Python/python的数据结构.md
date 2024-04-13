# Python的数据结构

## 列表，元组，集合

```python
例如：# 列表
a = [1, 2, 3]
a[0] = 4
print(a) # [4, 2, 3]

# 元组
b = (1, 2, 3)
# b[0] = 4  # TypeError: 'tuple' object does not support item assignment

# 集合
c = {1, 2, 3}
# c[0] = 4  # TypeError: 'set' object does not support item assignment
```

总结：

- 列表和元组都是有序的，但列表可以更改里面的元素，元组不能
- 集合是无序的，可以进行集合运算

选择使用哪种数据结构取决于你的具体需求，如果需要更改数据，可以使用列表；如果不需要更改数据，可以使用元组；如果需要进行集合运算，可以使用集合。

集合 (set) 支持常见的集合运算，如并集、交集、差集等。

下面是一些例子：

```text
# 定义两个集合
a = {1, 2, 3}
b = {3, 4, 5}

# 并集
c = a.union(b)
print(c) # {1, 2, 3, 4, 5}

# 交集
d = a.intersection(b)
print(d) # {3}

# 差集
e = a.difference(b)
print(e) # {1, 2}

# 对称差集
f = a.symmetric_difference(b)
print(f) # {1, 2, 4, 5}
```

结果就是：

- 并集 (union) 运算返回两个集合中所有不重复的元素。
- 交集 (intersection) 运算返回两个集合中相同的元素。
- 差集 (difference) 运算返回在第一个集合中但不在第二个集合中的元素。
- 对称差集 (symmetric_difference) 运算返回两个集合中不同的元素。

这些运算都是可以连续使用的，例如：

```text
a = {1, 2, 3}
b = {3, 4, 5}
c = {3, 6, 7}

d = a.intersection(b).intersection(c)
print(d) # {3}
```

这个例子中就是求出了 a,b,c 三个集合的交集

## set

是的，集合是一种无序且不包含重复元素的数据结构。在Python中，可以使用大括号 `{}` 或者 `set()` 函数来创建集合。集合中的元素是不可变的（immutable），这意味着集合中不能包含可变对象（例如列表），但可以包含不可变对象（例如整数、字符串、元组等）。

下面是一些关于集合的基本操作和特性：

1. 创建集合：
```python
my_set1 = {1, 2, 3, 4, 5}
my_set2 = set([1, 2, 3, 4, 5])  # 使用set()函数创建集合
```

2. 集合中的元素是唯一的：
```python
my_set = {1, 2, 3, 4, 5, 5}
print(my_set)  # 输出: {1, 2, 3, 4, 5}，重复元素被自动去重
```

3. 遍历集合：
```python
my_set = {1, 2, 3, 4, 5}
for item in my_set:
    print(item)
```

4. 添加和删除元素：
```python
my_set = {1, 2, 3}
my_set.add(4)  # 添加元素
print(my_set)  # 输出: {1, 2, 3, 4}

my_set.remove(2)  # 删除元素
print(my_set)  # 输出: {1, 3, 4}
```

总之，集合是一个非常有用的数据结构，特别适合用于处理无序且不包含重复元素的数据。