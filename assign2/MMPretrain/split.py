import glob
import random

from PIL import Image
import numpy as np
import os

# 1.初始化设置
split_ratio = 0.8

desired_size = 128
file_path = './dataset'

# 2.使用glob进行通配链接,使用os。join链接所有的目录
img_dir = glob.glob(os.path.join(file_path, '*'))
img_dir = [d for d in img_dir if os.path.isdir(d)]

catalogue_class = len(img_dir)
# 对此文件进行目录

# 3.接下来就是对图像进行缩放,这是一个类
for file in img_dir:
    # 通过对这个字符串使用split进行切割，获得最后一部分，就是类名
    file1 = file
    file = file.split('/')[-1]
    # 对这个类进行构建train&test
    os.makedirs(f'data/fruit/train/{file}', exist_ok=True)
    os.makedirs(f'data/fruit/val/{file}', exist_ok=True)
    
    os.makedirs(f'data/fruit/test/{file}', exist_ok=True)
    # 4.对这个类进行划分  0.8 0.1 0.1
    img_list = glob.glob(os.path.join(file1, '*.jpg'))
    random.shuffle(img_list)
    train_num = int(len(img_list) * split_ratio)
    val_num = int(len(img_list) * (1 - split_ratio) / 2)
    train_list = img_list[:train_num]
    val_list = img_list[train_num:train_num + val_num]
    test_list = img_list[train_num + val_num:]
    # 5.对这个类进行划分
    for img in train_list:
        last = img.split('/')[-1]
        img = Image.open(img)
        
        img.save(os.path.join(f'data/fruit/train/{file}', last))
    for img in val_list:
        last = img.split('/')[-1]
        
        img = Image.open(img)
        
        img.save(os.path.join(f'data/fruit/val/{file}', last))
    for img in test_list:
        last = img.split('/')[-1]
        
        img = Image.open(img)
        
        img.save(os.path.join(f'data/fruit/test/{file}', last))
    
    
print('Done')
   
