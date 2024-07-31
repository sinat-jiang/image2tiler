"""
通过平均化像素值实现图片像素化
"""

from PIL import Image
import os
import random
import numpy as np


# r, g, b 通道像素区间
color_gaps = 32      # 4 * 4 * 4
color_range_list = [(256 // color_gaps) * i for i in range(color_gaps + 1)]      # e.g. [0, 64, 128, 192, 256]


def ret_color(x):
    for idx, v in enumerate(color_range_list):
        if x < v:
            return color_range_list[idx - 1]
    return color_range_list[-1]


def pixelate_image(img, pixel_size, pix_cal_type='mean'):
    # 打开图像
    image = Image.open(img)
    
    # 获取图像模式
    mode = image.mode
    
    # 获取图像尺寸
    width, height = image.size
    
    # 创建一个新的图像
    pixelated_image = Image.new('RGB', (width, height))
    
    # 遍历每个像素并进行像素化
    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            r_list, g_list, b_list = [], [], []
            for py in range(pixel_size):
                for px in range(pixel_size):
                    if x + px < width and y + py < height:
                        # 根据图像模式获取像素值
                        if mode == 'RGB':
                            r_, g_, b_ = image.getpixel((x + px, y + py))
                        elif mode == 'RGBA':
                            r_, g_, b_, _ = image.getpixel((x + px, y + py))
                        else:
                            raise ValueError(f"Unsupported image mode: {mode}")
                        
                        r_list.append(r_)
                        g_list.append(g_)
                        b_list.append(b_)

            if pix_cal_type == 'mean':
                r = sum(r_list) // (pixel_size * pixel_size)
                g = sum(g_list) // (pixel_size * pixel_size)
                b = sum(b_list) // (pixel_size * pixel_size)
            elif pix_cal_type == 'max':
                r = max(r_list)
                g = max(g_list)
                b = max(b_list)
            elif pix_cal_type == 'min':
                r = min(r_list)
                g = min(g_list)
                b = min(b_list)
            elif pix_cal_type == 'median':
                r = sorted(r_list)[len(r_list) // 2]
                g = sorted(g_list)[len(g_list) // 2]
                b = sorted(b_list)[len(b_list) // 2]
            elif pix_cal_type == 'random':
                r = random.choice(r_list)
                g = random.choice(g_list)
                b = random.choice(b_list)
            elif pix_cal_type == 'range':
                r_list = [ret_color(x) for x in r_list]
                g_list = [ret_color(x) for x in g_list]
                b_list = [ret_color(x) for x in b_list]
                r = sorted(r_list)[len(r_list) // 2]
                g = sorted(g_list)[len(g_list) // 2]
                b = sorted(b_list)[len(b_list) // 2]
            else:
                raise ValueError(f"Unsupported image pixelate caculate mode: {mode}")
            
            # 将该平均颜色值应用到当前像素块
            for py in range(pixel_size):
                for px in range(pixel_size):
                    if x + px < width and y + py < height:
                        pixelated_image.putpixel((x + px, y + py), (r, g, b))
    
    return pixelated_image


def pixelate_image_parallel(image_path, out_image_path, pixel_size, pix_cal_type='median'):
    """
    优化代码，加速执行效率
    """
    # 打开图像
    image = Image.open(image_path)
    # 获取图像模式
    mode = image.mode
    # 获取图像尺寸
    width, height = image.size

    # 使用 numpy 获取图像数据
    img_data = np.array(image)
    if img_data.shape[-1] == 4:
        img_data = img_data[:, :, :3]
    
    # 根据 pixel_size 对图像进行分块
    new_width = width // pixel_size if width % pixel_size == 0 else (width // pixel_size) + 1
    new_height = height // pixel_size if height % pixel_size == 0 else (height // pixel_size) + 1

    # 创建新的图像数据（只保存 RGB 图像）
    pixelated_data = np.zeros((height, width, 3), dtype=np.uint8)

    # 计算新图像的像素值
    for y in range(new_height):
        for x in range(new_width):
            y_start, y_end = y * pixel_size, (y + 1) * pixel_size
            x_start, x_end = x * pixel_size, (x + 1) * pixel_size
            if y_end > height:
                y_end = height
            if x_end > width:
                x_end = width
            block = img_data[y_start:y_end, x_start:x_end]
            if pix_cal_type == 'mean':
                pixelated_data[y_start:y_end, x_start:x_end] = block.mean(axis=(0, 1))
            elif pix_cal_type == 'max':
                pixelated_data[y_start:y_end, x_start:x_end] = block.max(axis=(0, 1))
            elif pix_cal_type == 'min':
                pixelated_data[y_start:y_end, x_start:x_end] = block.min(axis=(0, 1))
            elif pix_cal_type == 'median':
                pixelated_data[y_start:y_end, x_start:x_end] = np.median(block, axis=[0, 1])
            elif pix_cal_type == 'random':
                pixelated_data[y_start:y_end, x_start:x_end] = np.random(block, axis=[0, 1])
            elif pix_cal_type == 'range':
                vectorized_function = np.vectorize(ret_color)
                processed_array = vectorized_function(block)
                pixelated_data[y_start:y_end, x_start:x_end] = np.median(processed_array, axis=[0, 1])
            else:
                raise ValueError(f"Unsupported image pixelate caculate mode: {mode}")

    # 创建新的图像对象并保存
    pixelated_image = Image.fromarray(pixelated_data)
    pixelated_image.save(out_image_path)
    print('success')


if __name__ == '__main__':
    # 像素块的大小（正方形边长）
    pixel_size = 3             # 5, 7, 10, 13, 16, 20
    image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'simple_images', 'wukong.jpg')
    # image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'standard_images', 'jz9.jpg')
    # image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'phone_background', 'rw7.jpg')
    out_image_path = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).replace('.', f'_pix_{pixel_size}_out.'))
    # median 和 range+color_gaps[=5] 的效果较好，但这种方式的色彩不如 tiler 项目的色彩鲜艳
    pixelated_image = pixelate_image_parallel(image_path, out_image_path, pixel_size, pix_cal_type='range')