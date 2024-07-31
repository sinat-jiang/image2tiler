## ReadMe

This rep is based on https://github.com/nuno-faria/tiler

### 1 - Tiler 项目使用

1. 先安装依赖：`pip install -r requirements.txt`

2. 先制作一张你自己想要的基础像素图，放在 `/tiles/your_tile_file/your_tile.png`，然后运行 `python gen_tiles.py path/to/your_tile.png`，生成多个同尺寸但色彩不同的基础像素图块，这个图块就是之后要画到真实图片上的图块，也叫 tiler，你也可以理解为马赛克。`tiles` 下本身也预置了许多不同的基础马赛克图像（tiler）。

3. 有了基础 tilers 图像块后，就可以通过 `python tiler.py` 来进行像素风格转换了。其中图片和其他参数修改为直接在 `tiler.py` 文件中指定了。

### 2 - 增加第二种像素风格转换实现方式

原始的项目功能自然更为强大，提供了自定义 tiler 的功能，这样可以使用自己喜欢的基础马赛克像素块来对图像进行像素化。

但最常见的像素化其实就是简单的平铺正方形像素块，这里的实现思路就是对原始图片，在三个颜色通道上，对指定的正方形区域大小，取像素平均值，来替代原来的真实值，实现像素化。这种实现方式代码运行速度快，便于对视频进行像素风格转换。

运行：`python simple_pixelate.py`