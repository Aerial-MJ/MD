## 共轭转置

![image-20240918233012829](../../Image/image-20240918233012829.png)

对于一个复矩阵 $A = (a_{ij})$，其**共轭矩阵**（也称为**共轭转置矩阵**或**厄米矩阵**）记作 $A^H$，其定义为：
$$
A^H = \overline{A}^T
$$
其中，$\overline{A}$ 是矩阵 $A$ 的元素的复共轭，$T$ 表示转置操作。

**共轭矩阵的元素**

如果矩阵 $A$ 的元素为 $a_{ij}$，那么 $A^H$ 的元素为 $ \overline{a_{ij}}$。即：
$$
(A^H)_{ij} = \overline{a_{ji}}
$$

**示例**

假设矩阵 $A$ 为：
$$
A = \begin{pmatrix}
    1 + i & 2 - i \\
    3 & 4 + 2i
\end{pmatrix}
$$

则 $A$ 的共轭矩阵 $A^H$ 为：
$$
A^H = \begin{pmatrix}
    1 - i & 3 \\
    2 + i & 4 - 2i
\end{pmatrix}
$$

## Hermite矩阵

**自共轭矩阵**

Hermite 矩阵的定义：$H = (h_{ij})$，其中满足 $h_{ij} = \overline{h_{ji}}$，即矩阵的元素关于主对角线共轭对称。

举个例子，一个 $3 \times 3$ 的 Hermite 矩阵可以表示为：
$$
H = \begin{pmatrix}
    h_{11} & h_{12} & h_{13} \\
    \overline{h_{12}} & h_{22} & h_{23} \\
    \overline{h_{13}} & \overline{h_{23}} & h_{33}
\end{pmatrix}
$$

## 酉矩阵

**复数域上的正交矩阵**

一个复矩阵 $U$ 被称为**酉矩阵**，如果它满足以下条件：
$$
U^\dagger U = U U^\dagger = I
$$
其中，$U^\dagger$ 是矩阵 $U$ 的**共轭转置矩阵**，$I$ 是单位矩阵。这意味着矩阵 $U$ 的共轭转置是它的逆矩阵。

### 酉矩阵的性质

1. **长度保持**：酉矩阵对应的线性变换保持向量的长度不变，即对于任何向量 $x$，都有：
$$
\|Ux\| = \|x\|
$$

2. **内积保持**：酉矩阵保持向量的内积，即对于任何两个向量 $x$ 和 $y$，有：
$$
\langle Ux, Uy \rangle = \langle x, y \rangle
$$

### 示例

一个常见的 $2 \times 2$ 酉矩阵是：
$$
U = \begin{pmatrix}
    e^{i\theta} & 0 \\
    0 & e^{-i\theta}
\end{pmatrix}
$$
其中 $e^{i\theta}$ 表示一个复数的相位因子。