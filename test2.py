import os


if __name__ == '__main__':
    frames_path = r'D:\D0\coding\workspace_AI\ai_toys\CV\mosaic\image2tiler\test_videos\sunset\Evening_Beach_Sunset_Fireworks_simple_pixelate_frames'
    # 读取所有文件
    frames = os.listdir(frames_path)

    already_trans_idxs = set()
    for frame in frames:

        idx = frame.split('.')[0].split('_')[-1]
        already_trans_idxs.add(int(idx))

    all_idxs = set([i for i in range(1145)])

    dif_set = all_idxs.difference(already_trans_idxs)

    print(already_trans_idxs)
    print(all_idxs)

    a = list(dif_set)
    a.sort()
    print(a)

