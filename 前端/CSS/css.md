# CSS

## 伪类和伪元素

CSS 伪类和伪元素是用于选择 HTML 元素的特殊选择器。它们可以使得我们在不添加额外 HTML 元素的情况下，实现一些特殊的效果和样式。以下是伪类和伪元素的定义和区别：

1. 伪类（Pseudo-Class）：**伪类用于描述某些特定状态下的元素样式**，例如链接的状态（已访问、未访问、悬停等）和表单元素的状态（选中、未选中等）等。常见的伪类包括 `:hover`、`:active`、`:focus`、`:visited` 等。
2. 伪元素（Pseudo-Element）：伪元素用于**在元素内部创建一些不同于元素本身的内容，并可以应用样式。**常见的伪元素包括 `::before` 和 `::after`，它们可以用于创建一些装饰性的元素或者添加一些文本内容。

区别：

- 伪类用于描述元素的**状态或行为**，而伪元素用于描述元素的**内容**。
- 伪类前缀为一个冒号 `:`，而伪元素前缀为两个冒号 `::`。
- 由于历史原因，伪元素在旧版浏览器中需要使用单冒号 `:`，而在 CSS3 中规定使用双冒号 `::`，但为了兼容性，双冒号和单冒号在现代浏览器中都可以使用。

以下是一个使用伪元素和伪类的例子，用于为某个 HTML 元素添加一些特殊样式：

```css
/* 为链接添加悬停状态下的样式 */
a:hover {
  color: red;
  text-decoration: underline;
}

/* 在元素内容前添加一些装饰性元素 */
p::before {
  content: ">> ";
  font-weight: bold;
}

/* 在元素内容后添加一些文本内容 */
p::after {
  content: " <<";
  font-weight: bold;
}
```

## 总结

```css
letter-spaceing --字间距
line-height  --行高（行间距）

box-shadow: 0 0 5px 5px rgb(0 0 0 / 8%) --阴影
filter：brightness(0.5)  --高斯模糊

opacity --透明度
z-index: 1  --上下层的位置

font-size  --字体大小
font-weight  --字体粗细

transition --过渡 （区别动画和转换）
cursor: pointer --鼠标移上去的状态

//其中转换是静止的，转换(2D和3D)和过渡可以连用 
//转化+过渡≈动画效果
div
{
	width:100px;
	height:75px;
	background-color:red;
	border:1px solid black;
	cursor:pointer；
	transition:1s;
}

//transiton：xxs xxs  （前面是持续时间，后面是delay-time（延迟多少秒, 才开始动画））
transition-delay: 1s; 延迟多少秒, 才开始动画

div:hover{
	transform:rotateY(130deg);
	-webkit-transform:rotateY(130deg); /* Safari and Chrome */
}

display：block/inline/flex
float：right/left --漂浮状态
position：relative/absolute/fixed  相对定位/绝对定位/固定定位
 
background  集合了下面多属性的集合属性
background-color  --背景色
background-size:  百分比 ---宽 长
background-image：  --背景图片
background-repeat： no-repeat

```

## 高斯模糊

```css
/* URL to SVG filter */
filter: url("filters.svg#filter-id");

#具体来说，filter 属性用于指定应用到元素的图形滤镜，url() 函数用于引用滤镜的定义文件。在这个例子中，"filters.svg#filter-id" 表示引用名为 filter-id 的 SVG 滤镜，它定义在名为 filters.svg 的 SVG 文件中。

#SVG（Scalable Vector Graphics）是一种基于 XML 的矢量图形格式，它可以用于创建各种图形和动画效果。滤镜是 SVG 中的一种特殊元素，它可以对 SVG 图形进行各种特殊效果的处理，例如模糊、扭曲、亮度、对比度等。使用 CSS 中的 filter 属性可以将 SVG 中定义的滤镜应用到 HTML 元素上，从而实现各种特殊效果。

/* <filter-function> values */
filter: blur(5px);
filter: brightness(0.4);
filter: contrast(200%);
filter: drop-shadow(16px 16px 20px blue);
filter: grayscale(50%);
filter: hue-rotate(90deg);
filter: invert(75%);
filter: opacity(25%);
filter: saturate(30%);
filter: sepia(60%);

/* Multiple filters */
filter: contrast(175%) brightness(3%);

/* Use no filter */
filter: none;

/* Global values */
filter: inherit;
filter: initial;
filter: revert;
filter: unset;
```

**下面是一个例子，使用 SVG 滤镜和 CSS `filter` 属性创建一个简单的图片效果：**

**给example.jpg 添加一个图形滤镜**

HTML 代码：

```html
<img src="example.jpg" class="filter-img" />
```

CSS 代码：

```css
/* 定义 SVG 滤镜 */
svg {
  height: 0;
  width: 0;
  position: absolute;
}

.filter-img {
  /* 引用名为 "filter-1" 的 SVG 滤镜 */
  filter: url("#filter-1");
}
```

SVG 代码

```svg
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <!-- 定义名为 "filter-1" 的滤镜 -->
    <filter id="filter-1">
      <feGaussianBlur stdDeviation="3" />
      <feColorMatrix type="matrix"
        values="0.8 0 0 0 0
                0 0.8 0 0 0
                0 0 0.8 0 0
                0 0 0 1 0" />
    </filter>
  </defs>
</svg>
```

在这个例子中，SVG 滤镜定义了一个高斯模糊和颜色矩阵效果，可以将图片进行模糊处理和颜色调整。CSS 中的 `filter` 属性通过引用滤镜的 ID，将滤镜应用到了图片元素上。最终，图片就会呈现出经过滤镜处理后的效果。

**html代码**

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    .filtered-image {
      filter: url(#my-filter);
    }
  </style>
</head>
<body>
  <img src="example.jpg" class="filtered-image">

  <svg width="0" height="0" xmlns="http://www.w3.org/2000/svg">
    <filter id="my-filter">
      <!-- 在这里定义你想要的滤镜效果 -->
      <!-- 例如，下面的示例将图像转为黑白效果 -->
      <feColorMatrix type="matrix"
        values="0.33 0.33 0.33 0 0
                0.33 0.33 0.33 0 0
                0.33 0.33 0.33 0 0
                0    0    0    1 0"/>
    </filter>
  </svg>
</body>
</html>
```

