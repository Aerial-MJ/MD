## HTML标签

在 HTML 中，行级内联标签（inline elements）和区块标签（block elements）有不同的显示和布局特性。以下是它们的主要区别：

### 行级内联标签（Inline Elements）

- **定义**：行级内联标签通常包含短内容，如文本或链接，不会单独占用一行。
- **布局**：与其他内容（文本或内联元素）在同一行内显示，不会在前后产生换行。
- **尺寸**：宽度和高度由其内容决定，不能设置 `width` 和 `height` 属性（某些元素如图片除外）。
- **常见内联元素**：
  - `<a>`：超链接
  - `<span>`：行内容器
  - `<strong>`：强调文本（加粗）
  - `<em>`：强调文本（斜体）
  - `<img>`：图像
  - `<code>`：代码片段

**示例**：

```html
<p>This is an <a href="#">inline link</a> inside a paragraph.</p>
```

### 区块标签（Block Elements）

- **定义**：区块标签通常包含较长内容，如段落或列表，占据整个宽度。
- **布局**：默认情况下独占一行，在前后产生换行。
- **尺寸**：可以设置 `width` 和 `height` 属性，默认宽度为父容器的 100%。
- **常见区块元素**：
  - `<div>`：通用区块容器
  - `<p>`：段落
  - `<h1>` 到 `<h6>`：标题
  - `<ul>`：无序列表
  - `<ol>`：有序列表
  - `<li>`：列表项
  - `<section>`：文档区段
  - `<article>`：文章

**示例**：

```html
<div>
  <h1>This is a block-level heading</h1>
  <p>This is a paragraph inside a div.</p>
</div>
```

### 行级内联标签和区块标签的区别

1. **显示方式**：
   - 行级内联标签：在同一行内显示，不会换行。
   - 区块标签：独占一行，会在前后产生换行。

2. **尺寸控制**：
   - 行级内联标签：宽度和高度由内容决定，通常不能设置 `width` 和 `height` 属性。
   - 区块标签：可以设置 `width` 和 `height` 属性，默认宽度为父容器的 100%。

3. **嵌套规则**：
   - 行级内联标签：可以嵌套在区块标签中，但通常不能包含区块标签。
   - 区块标签：可以包含行级内联标签或其他区块标签。

### 例子对比

**行级内联标签示例**：

```html
<p>This is a paragraph with an <em>inline emphasized</em> word and an <a href="#">inline link</a>.</p>
```

**区块标签示例**：

```html
<div>
  <h2>Heading</h2>
  <p>This is a paragraph within a div. Below is a list:</p>
  <ul>
    <li>List item 1</li>
    <li>List item 2</li>
  </ul>
</div>
```

在实际使用中，行级内联标签和区块标签的选择取决于内容的结构和布局需求。

## HTML实体

HTML 实体（HTML Entities）用于表示在 HTML 中难以直接使用的字符。这些实体以 `&` 开头，以 `;` 结尾，中间是实体名称或实体编号。

### 常见 HTML 实体

1. **空格和制表符**
   - `&nbsp;`：不间断空格（non-breaking space）
   - `&ensp;`：半个空格（en space）
   - `&emsp;`：一个空格（em space）

2. **符号**
   - `&lt;`：小于号（<）
   - `&gt;`：大于号（>）
   - `&amp;`：和号（&）
   - `&quot;`：双引号（"）
   - `&apos;`：单引号（'）

3. **其他常见字符**
   - `&cent;`：美分符号（¢）
   - `&pound;`：英镑符号（£）
   - `&yen;`：日元符号（¥）
   - `&euro;`：欧元符号（€）
   - `&copy;`：版权符号（©）
   - `&reg;`：注册商标符号（®）

### 实体编号

除了实体名称，还可以使用实体编号来表示字符。实体编号以 `&#` 开头，后跟字符的 Unicode 编码，以 `;` 结尾。例如：

- `&#60;`：小于号（<）
- `&#62;`：大于号（>）
- `&#38;`：和号（&）
- `&#34;`：双引号（"）
- `&#39;`：单引号（'）

### 使用示例

以下是一些使用 HTML 实体的示例：

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HTML Entities</title>
</head>
<body>
    <p>Less than: &lt;</p>
    <p>Greater than: &gt;</p>
    <p>Ampersand: &amp;</p>
    <p>Double quote: &quot;</p>
    <p>Non-breaking space: &nbsp;</p>
    <p>Copyright: &copy; 2024</p>
</body>
</html>
```

转换后，在浏览器中显示为：

- Less than: <
- Greater than: >
- Ampersand: &
- Double quote: "
- Non-breaking space: (一个不可见的空格)
- Copyright: © 2024

### 特殊字符在 Markdown 中的处理

在 Markdown 中使用特殊字符时，可以直接使用 HTML 实体来确保它们正确显示。例如：

```markdown
This is a less than sign: &lt;
This is an ampersand: &amp;
```

转换后在 HTML 中显示为：

- This is a less than sign: <
- This is an ampersand: &

使用 HTML 实体可以确保在处理特殊字符时不会引起格式问题或显示错误。

## 特殊字符自动转义

在 HTML 文件中，有两个字符需要特殊处理： `<` 和 `&` 。 `<` 符号用于起始标签，`&` 符号则用于标记 HTML 实体，如果你只是想要使用这些符号，你必须要使用实体的形式，像是 `<` 和 `&`。

`&` 符号其实很容易让写作网页文件的人感到困扰，如果你要打「AT&T」 ，你必须要写成「`AT&T`」 ，还得转换网址内的 `&` 符号，如果你要链接到：

```
http://images.google.com/images?num=30&q=larry+bird
```

你必须要把网址转成：

```
http://images.google.com/images?num=30&amp;q=larry+bird
```

才能放到链接标签的 `href` 属性里。不用说也知道这很容易忘记，这也可能是 HTML 标准检查所检查到的错误中，数量最多的。

Markdown 允许你直接使用这些符号，它帮你自动转义字符。如果你使用 `&` 符号的作为 HTML 实体的一部分，那么它不会被转换，而在其它情况下，它则会被转换成 `&`。所以你如果要在文件中插入一个著作权的符号，你可以这样写：

```
&copy;
```

Markdown 将不会对这段文字做修改，但是如果你这样写：

```
AT&T
```

Markdown 就会将它转为：

```
AT&amp;T
```

类似的状况也会发生在 `<` 符号上，因为 Markdown 支持 [行内 HTML](https://markdown.com.cn/basic-syntax/#内联-html) ，如果你使用 `<` 符号作为 HTML 标签的分隔符，那 Markdown 也不会对它做任何转换，但是如果你是写：

```
4 < 5
```

Markdown 将会把它转换为：

```
4 &lt; 5
```

需要特别注意的是，在 Markdown 的块级元素和内联元素中， `<` 和 `&` 两个符号都会被自动转换成 HTML 实体，这项特性让你可以很容易地用 Markdown 写 HTML。（在 HTML 语法中，你要手动把所有的 `<` 和 `&` 都转换为 HTML 实体。）