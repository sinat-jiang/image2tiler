"""
视频像素化
"""
import os
import cv2
from PIL import Image, ImageSequence
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from simple_pixelate import pixelate_image_parallel
from tiler import tiler_pixlate, load_tiles


root = os.path.dirname(__file__)


def gif_frames_extract(gif_path, output_folder):
    """
    gif 拆解

    Params:
    - gif_path:
    - output_folder:
    """
    # 获取 gif 文件名
    gif_name = os.path.splitext(os.path.basename(gif_path))[0]

    # 检查输出文件夹
    output_path = output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    imgs = Image.open(gif_path)
    pbar = tqdm(desc='抽取帧数：')
    count = 0
    for i, f in enumerate(ImageSequence.Iterator(imgs)):
        image_name = os.path.join(output_path, f"{gif_name}_frame_{count}.png")
        duration = f.info['duration']
        f.save(image_name)
        count += 1
        pbar.update(1)

    print('gif split success.')
    return duration


def frames_to_gif(frames_path, gif_path, duration=1000):
    """
    组成 gif
    
    Params:
    - frames_path:
    - gif_path:
    """
    im_list = os.listdir(frames_path)
    im_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))    # 对帧图片进行排序
    img = Image.open(os.path.join(frames_path, im_list[0]))
    img_size = img.size                                 # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致
    # print('image size:', img_size)

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G')        # opencv 版本是 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')            # opencv 版本是 3
    videoWriter = cv2.VideoWriter(gif_path, fourcc, fps, img_size)
    images = []
    for i in tqdm(im_list, desc='video synthesizing:'):
        im_name = os.path.join(frames_path, i)
        images.append(Image.open(im_name))
    
    # merge to gif
    head_image = images[0]
    head_image.save(gif_path, format='GIF', save_all=True, append_images=images[1:], optimize=True, duration=duration, loop=0)

    print('gif synthesize finished.')


def video_frames_extract(video_path, output_folder, ff=None):
    """
    拆帧

    Params:
    - video_path:
    - output_folder:
    - fps:
    """
    # 获取视频文件名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 新建文件夹
    output_path = output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    # 获取&设置帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('视频默认帧率：', fps)
    if ff is None:
        # 设置帧间隔为 1，即每帧的图片都会被抽取出来
        frame_interval = 1
    else:
        frame_interval = int(ff)
    print('设置帧间隔：', frame_interval)

    # 逐帧提取并保存
    pbar = tqdm(desc='抽取帧数：')
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if count % frame_interval == 0:
                image_name = os.path.join(output_path, f"{video_name}_frame_{count}.jpg")
                cv2.imwrite(image_name, frame)
            count += 1
            pbar.update(1)
        else:
            break
    cap.release()
    print('video split success.')
    return fps


def frames_to_video(frames_path, video_path, fps=25):
    """
    组帧
    
    Params:
    - frames_path:
    - video_path:
    - fps: 每秒多少帧，推荐和原视频保持一致
    """
    im_list = os.listdir(frames_path)
    im_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))    # 对帧图片进行排序
    img = Image.open(os.path.join(frames_path, im_list[0]))
    img_size = img.size                                 # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致
    # print('image size:', img_size)

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G')        # opencv 版本是 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')            # opencv 版本是 3
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
    for i in tqdm(im_list, desc='video synthesizing:'):
        im_name = os.path.join(frames_path, i)
        # frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
    videoWriter.release()
    print('video synthesize finished.')


def frames_to_pixelate(frames_path, new_frames_save_path, kwargs, type='simple', max_workers=20):
    """
    依次将每张图片做像素风格转换
    
    Params: 
    - frames_path: 存放帧图像的路径
    - new_frames_save_path: 转换后的图像存储路径
    - args: 风格转换的参数
    - type: 转换类型，黑白 or 彩色
    """

    if not os.path.exists(new_frames_save_path):
        os.makedirs(new_frames_save_path)

    # 读取所有文件
    frames = os.listdir(frames_path)
    # frames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))    # 对帧图片进行排序
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        if type == 'tiler':
            # 提前加载好 tiler
            tiles = load_tiles(kwargs['tiles_paths'], kwargs)

        # pool = ProcessPoolExecutor(max_workers=max_workers)

        # sumbit all tasks
        all_task = []
        for frame in frames:
            frame = os.path.join(frames_path, frame)
            # 依次进行转换并存储
            output_image = os.path.join(new_frames_save_path, os.path.basename(frame))
            if type == 'simple':
                all_task.append(pool.submit(pixelate_image_parallel, frame, output_image, **kwargs))
            elif type == 'tiler':
                all_task.append(pool.submit(tiler_pixlate, frame, kwargs['tiles_paths'], output_image, kwargs, tiles))
            else:
                print('请指定正确的转换类型')

        # catch the results of all tasks and manual update the tqdm process bar
        with tqdm(desc=f'Image2pixelate: ', ncols=80, total=len(frames)) as pbar_b:
            for future in as_completed(all_task):
                # future.result()
                pbar_b.update(1)
    
    print('All frames are converted to pixelate style.')


if __name__ == '__main__':
    
    # 1 拆帧
    # 1.1 视频
    # video_path = f'./test_videos/luori1.mp4'                 # 视频文件路径
    # video_path = f'./test_videos/sunset/Evening_Beach_Sunset_Fireworks.mp4'                 # 视频文件路径
    # video_path = f'./test_videos/sence_animation/dmkj.mp4'                                # 视频文件路径
    # output_folder = os.path.join(os.path.dirname(video_path), f"{os.path.basename(video_path).split('.')[0]}_frames")    # 输出文件夹路径
    # output_folder = os.path.join(os.path.dirname(video_path), f"{os.path.basename(video_path).split('.')[0]}_p3")    # tmp tmp tmp

    # 1.2 gif
    video_path = f'./test_videos/gif-sence-anime/train.gif'                 # gif 文件路径
    output_folder = os.path.join(os.path.dirname(video_path), f"{os.path.basename(video_path).split('.')[0]}_frames")    # 输出文件夹路径
    
    # if video_path.split('.')[-1] in ['mp4']:
    #     fps = video_frames_extract(video_path, output_folder, ff=None)
    #     print('fps:', fps)
    # elif video_path.split('.')[-1] in ['gif']:
    #     duration = gif_frames_extract(video_path, output_folder)
    #     print('duration:', duration)

    # 2 像素风格转化
    type = 'simple'     # simple or tiler
    new_frames_save_path = os.path.join(os.path.dirname(output_folder), f"{os.path.basename(video_path).split('.')[0]}_{type}_pixelate_frames")
    # new_frames_save_path = os.path.join(os.path.dirname(output_folder), f"{os.path.basename(video_path).split('.')[0]}_{type}_pixelate_frames_p3_ps6")    # tmp tmp tmp
    # params for simple pixelate
    kwargs = {
        'pixel_size': 7,                # 像素块大小，6,10,16,20
        'pix_cal_type': 'range',         # 像素块值计算模式，默认 median
        'color_gaps': 13                 # range 模式下每个颜色通道的分割数，HD 图片推荐 13，简单图片推荐 5
    }
    # params for tiler pixelate
    # kwargs = {
    #     # number of divisions per channel, (COLOR_DEPTH = 32 -> 32 * 32 * 32 = 32768 colors)
    #         'COLOR_DEPTH': 64,    
    #     # Scale of the image to be tiled (1 = default resolution)      
    #         'IMAGE_SCALE': 1,                 # 只会改变图像尺寸 w 和 h，但不会改变 tile 相对于图片的大小
    #     # tiles scales (1 = default resolution), e.g. RESIZING_SCALES = [0.5, 0.4, 0.3, 0.2, 0.1]。根据原始 tiler 的大小进行放缩，原始 tiler 为 100x100 时，设为 0.1 表示 10x10
    #         'RESIZING_SCALES': [0.06],        # 当只有一个元素时，表示只保留一种大小的像素，并且可以依此调整像素块的大小【一般设 0.2 效果还行，如果图片本身尺寸较小，可以设为 0.02-0.05 看看效果】
    #     # number of pixels shifted to create each box (tuple with (x,y))
    #     # if value is None, shift will be done accordingly to tiles dimensions
    #         'PIXEL_SHIFT': None,
    #     # if tiles can overlap
    #         'OVERLAP_TILES': False,
    #     # render image as its being built
    #         'RENDER': False,                  # 一个布尔标志,指示是否在构建过程中渲染图像
    #     # multiprocessing pool size
    #         'POOL_SIZE': 8,
    #         'tiles_paths': [os.path.join(root, 'tiles', 'squares', 'gen_square')]      # tiler 路径
    # }
    # frames_to_pixelate(frames_path=output_folder, new_frames_save_path=new_frames_save_path, kwargs=kwargs, type=type, max_workers=6 if type == 'simple' else 5)

    # 组帧
    fps = 30
    duration = 40
    video_path = os.path.join(os.path.dirname(video_path), os.path.basename(video_path).replace('.', f'2pixelate_{type}.'))
    if video_path.split('.')[-1] in ['mp4']:
        frames_to_video(frames_path=new_frames_save_path, video_path=video_path, fps=fps)
    elif video_path.split('.')[-1] in ['gif']:
        frames_to_gif(frames_path=new_frames_save_path, gif_path=video_path, duration=duration)
