# GEN TILES CONFS

# number of divisions per channel (R, G and B)，分割点数，而不是分成几份
# DEPTH = 4 意味着每个颜色通道被分成 5 个部分,总共 5 * 5 * 5 = 125 个颜色，更大的 DEPTH 值意味着更多的颜色
DEPTH = 4
# list of rotations, in degrees, to apply over the original image
ROTATIONS = [0]


#############################


# TILER CONFS

# number of divisions per channel
# (COLOR_DEPTH = 32 -> 32 * 32 * 32 = 32768 colors)
COLOR_DEPTH = 32
# Scale of the image to be tiled (1 = default resolution)
IMAGE_SCALE = 1                 # 只会改变图像尺寸 w 和 h，但不会改变 tile 相对于图片的大小
# tiles scales (1 = default resolution)
# RESIZING_SCALES = [0.5, 0.4, 0.3, 0.2, 0.1]
RESIZING_SCALES = [0.1]         # 只保留一种大小的像素，并且可以依此调整像素块的大小【一般设 0.2 效果还行，如果图片本身尺寸较小，可以设为 0.02-0.05 看看效果】

# number of pixels shifted to create each box (tuple with (x,y))
# if value is None, shift will be done accordingly to tiles dimensions
PIXEL_SHIFT = None
# if tiles can overlap
OVERLAP_TILES = False
# render image as its being built
RENDER = False                  # 一个布尔标志,指示是否在构建过程中渲染图像
# multiprocessing pool size
POOL_SIZE = 8

# out file name
# OUT = 'out.png'
# image to tile (ignored if passed as the 1st arg)
IMAGE_TO_TILE = None
# folder with tiles (ignored if passed as the 2nd arg)
TILES_FOLDER = None
