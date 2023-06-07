**题目**：基于 ResNet50 的水果分类

**背景**：使用基于卷积的深度神经网络 ResNet50 对 30 种水果进行分类

**任务**

1. 划分训练集和验证集
2. 按照 MMPreTrain CustomDataset 格式组织训练集和验证集
3. 使用 MMPreTrain 算法库，编写配置文件，正确加载预训练模型
4. 在水果数据集上进行微调训练
5. 使用 MMPreTrain 的 ImageClassificationInferencer 接口，对网络水果图像，或自己拍摄的水果图像，使用训练好的模型进行分类
6. 需提交的验证集评估指标（不能低于 60%）

- ResNet-50
  [![Resnet-50](https://user-images.githubusercontent.com/94358981/243633153-f76b4aa5-e4d6-4c02-bff9-09d974268bfa.png)](https://user-images.githubusercontent.com/94358981/243633153-f76b4aa5-e4d6-4c02-bff9-09d974268bfa.png)



### 对数据集进行划分

调用以前写的代码,不想重新再写

```python
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
   

```



## 配置文件

修改成adam优化器,其他没动,按照resnet50的代码更改过来

```python
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=33,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
    init_cfg = dict(type='Pretrained',checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth')
    )


# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=33,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/fruit/train/',


        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
test_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/fruit/test/',


        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/fruit/val/',


        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy', topk=(1))

# If you want standard test, please manually configure the test dataset

test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)



# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1,max_keep_ckpts=5,save_best='auto'),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
work_dir = './exp'
#直接在cfg设置workdir
```



## 验证效果

```python
 mim test mmpretrain res50.py  --checkpoint exp/best_accuracy_top1_epoch_74.pth

```



```shell
Loads checkpoint by local backend from path: exp/best_accuracy_top1_epoch_74.pth
06/07 15:38:36 - mmengine - INFO - Load checkpoint from exp/best_accuracy_top1_epoch_74.pth
06/07 15:38:40 - mmengine - INFO - Epoch(test) [4/4]    accuracy/top1: 91.2060  data_time: 0.1837  time: 0.7654
Testing finished successfully.

```



## 测试推理

![](https://fastly.jsdelivr.net/gh/weijia99/blog_image@main/16861238165741686123815846.png)
