# LSB


## 配置环境脚本

```cmd
. init.sh
```

## 基本用法

在初始化函数中 传入背景图片路径 和 水印图片路径
```python
lsb = LSB('images/bg1.png', 'images/wm.png', 2)  # 参数解释: 背景图片路径 水印图片路径 嵌入比特位数
```

合成
```python
synthesis = lsb.embed()  # 获取嵌入后的合成图片
```
提取
```python
synthesis = lsb.embed()  # 获取嵌入后的合成图片
ebg, ewm = lsb.extract(synthesis)  # 获取提取出的背景图片，水印图片
```

使用show会将图片用matplotlib展示
```python
lsb.show()
```


