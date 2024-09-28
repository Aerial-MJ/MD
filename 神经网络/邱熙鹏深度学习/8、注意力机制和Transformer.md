# 注意力机制

![image-20240928172331932](C:/Users/19409/AppData/Roaming/Typora/typora-user-images/image-20240928172331932.png)

自下而上：不用主动关注，自带显著特征，只要我们的输入信息刺激性足够强，就能被关注到，而且对周围信息有抑制效应

自上而下：有明显关注的任务，带着任务从输入信息中选择一个信息出来（即我们此处讨论的**attention**）

## 人工神经网络的注意力机制

![image-20240928174556483](../../Image/image-20240928174556483.png)

**soft  attention**

 ![image-20240928174852790](../../Image/image-20240928174852790.png)

**连续函数，方便求导**

打分函数$s(x_n,q)$ ：**计算x和q的相似度**

![image-20240928180131943](../../Image/image-20240928180131943.png)

**hard attention**：一般不考虑，离散无法求导

**键值对注意力  key-value pair attention**

![image-20240928180427004](C:/Users/19409/AppData/Roaming/Typora/typora-user-images/image-20240928180427004.png)

**多头注意力  multi-head attention**

![image-20240928180854609](../../Image/image-20240928180854609.png)

**指针网络**

![image-20240928180751305](../../Image/image-20240928180751305.png)

## Transformer
