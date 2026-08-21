# 图片字节码 vs 像素矩阵：Parquet 里到底存的是什么

> 起因：搞不清楚「字节码」「像素值」「Parquet 里存的形态」之间的关系。下面逐条确认。

---

## 1. "只要存了就是字节码，网上下载的也算"—— ✅ 对

不管是磁盘文件、网络下载的响应体（`requests.get(url).content`），还是内存里传输的数据，**只要是以二进制形式存在/传输，都是字节码**。这是个很通用的概念，跟"图片"没有特殊关系。

## 2. "Image.open() 之后不是字节码，是像素值"—— ✅ 对

```python
img = Image.open(path)   # img 现在是 PIL.Image 对象，内部是像素矩阵，不再是字节码
```

`Image.open()` 把字节码**解码**成了具体的像素数值（每个像素的 RGB 值），这是完全不同的两种数据形态。

## 3. "字节码是不是就是像素编码之后的东西"—— ✅ 基本对，可以这么理解

```
像素矩阵 --[编码/压缩算法，如JPEG]--> 字节码
字节码   --[解码，如Image.open()]  --> 像素矩阵
```

字节码就是"像素数据经过某种编码格式（JPEG/PNG/BMP等）转换后的二进制表示"，这个理解是对的。

---

## 4. Parquet 里存的到底是像素还是字节码？—— **存的是字节码**

这是关键点。虽然你在 Python 代码里操作的 `img` 变量是 PIL.Image 对象（像素矩阵），但 **HuggingFace `datasets` 库在真正 `.to_parquet()` 落盘时，会自动把 PIL.Image 对象重新编码成字节码再存**，不会直接存像素矩阵。

原因很简单：像素矩阵**未压缩、体积巨大**（比如一张 1120×1120 的图，未压缩要占 `1120×1120×3 ≈ 3.7MB`），如果直接存 Parquet 文件会大到无法接受；而编码成 JPEG/PNG 字节码后，通常只有几百 KB，体积小很多。

对应 `verify_verl_real_prompt_len.py` 里的代码就能验证这一点：

```python
def to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if img.get("bytes") is not None:
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            return Image.open(img["path"]).convert("RGB")
    raise TypeError(f"未知图片格式: {type(img)}")
```

这段代码专门处理"从 Parquet 读出来的图片是 `{"bytes": ..., "path": ...}` 这种字典格式"的情况——`img["bytes"]` 就是**存在 Parquet 里的字节码**，读的时候还要再用 `Image.open(io.BytesIO(...))` 重新解码一次，才能变回像素矩阵。

---

## 完整总结表

| 数据形态 | 是字节码还是像素？ | 出现在哪个环节 |
|---|---|---|
| 磁盘上的 `.jpg` 文件 | 字节码 | 原始存储 |
| `requests.get(url).content` | 字节码 | 网络下载 |
| `Image.open()` 返回的对象 | 像素矩阵 | 内存中处理 |
| `img.resize()` 之后 | 像素矩阵 | 内存中处理 |
| **Parquet 文件里的内容** | **字节码**（`{"bytes": ...}`） | 落盘存储 |
| 训练时读出来再 `Image.open()` | 像素矩阵 | 送入模型前 |

---

## 一句话总结

**只要是"落盘/存储/传输"，最终都会变回字节码；只有在内存里被 `Image.open()` 打开、准备做图像处理运算的那个中间阶段，才是像素矩阵的形态。** Parquet 也不例外，遵循同样的规律。

### 画成流程图

```
原始 .jpg (字节码)
    │  Image.open()  →  解码
    ▼
PIL.Image (像素矩阵, 内存中)
    │  resize / preprocess / augment
    ▼
PIL.Image (像素矩阵, 内存中)
    │  datasets.to_parquet()  →  重新编码
    ▼
Parquet 文件 (字节码, {"bytes": ..., "path": ...})
    │  datasets.load_dataset()  →  读出字典
    ▼
{"bytes": ...} (字节码, 内存中)
    │  Image.open(io.BytesIO(img["bytes"]))  →  再解码
    ▼
PIL.Image (像素矩阵, 送入模型前)
```
