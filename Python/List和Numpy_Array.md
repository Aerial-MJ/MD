## List-列表

```python
my_list = [1, 2, 3, 4, 5]
print(len(my_list))  # 输出：5

my_list.append(6)
print(my_list)  # 输出：[1, 2, 3, 4, 5, 6]

my_list.extend([7, 8, 9])
print(my_list)  # 输出：[1, 2, 3, 4, 5, 6, 7, 8, 9]

my_list.pop()
print(my_list)  # 输出：[1, 2, 3, 4, 5, 6, 7, 8]

```

## Numpy Array-数组

```python
import numpy as np

# 创建一个长度为 5 的数组
a = np.array([1, 2, 3, 4, 5])
print(a)  # 输出：[1 2 3 4 5]

# 使用 resize() 方法改变数组的长度
a.resize(8)
print(a)  # 输出：[1 2 3 4 5 0 0 0]

# 使用 append() 方法添加新元素
a = np.append(a, [6, 7, 8])
print(a)  # 输出：[1 2 3 4 5 0 0 0 6 7 8]

# 使用 insert() 方法插入新元素
a = np.insert(a, 2, [9, 10])
print(a)  # 输出：[1 2 9 10 3 4 5 0 0 6 7 8]

# 使用 delete() 方法删除指定位置的元素
a = np.delete(a, [3, 4])
print(a)  # 输出：[1 2 9 5 0 0 6 7 8]
```

## 两者区别

Python中的List和Array都是用来存储数据的容器，但是它们之间还是有一些区别的：

1. 存储方式：

   - List 可以存储不同类型的数据，而且可以动态改变大小，它是用链表实现的，每个元素在内存中是分开存储的。

   - Array 只能存储相同类型的数据，并且一旦创建就无法改变大小，它是用连续的内存块来存储数据的，因此访问元素比List更快。


2. 访问元素：

   - List中的元素可以通过索引和切片来访问，但是访问速度比Array慢。

   - Array中的元素只能通过索引来访问，但是访问速度比List快。


3. 用途：

   - List主要用于需要动态增加或减少元素的情况，或者需要存储不同类型的数据。

   - Array主要用于需要频繁访问元素的情况，或者需要存储大量同类型的数据。


总之，List和Array都有各自的优缺点，选择哪个容器要根据具体的应用场景来决定。
