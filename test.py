import zipfile

import itertools

def extract_zip(zip_file, password):
    try:
        zip_file.extractall(path=".", pwd=password.encode())
        return True

    except Exception as e:

        return False

zip_path = f"E:\BaiduNetdiskDownload\视频剪辑\AE\AE安装包\普通电脑\Win版 AE 2023\Win版 AE 2023.zip"

# characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{};':\",./<>?"
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

for length in range(2, 9):       # 根据情况适当调整密码长度的范围
    for password in itertools.product(characters, repeat=length):
        password = "".join(password)
        print("trying:", password)
        with zipfile.ZipFile(zip_path) as zip_file:
            if extract_zip(zip_file, password):
                print(f"密码破解成功：{password}")
                break

print("密码破解完成") 