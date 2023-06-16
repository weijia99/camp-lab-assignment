******
**作业：ControlNet 的 N 种玩法****
假设你是某装修公司的设计师，客户发了你毛坯房的照片，想让你设计未来装修好的效果图。
先将毛坯房照片，用 OpenCV 转为 Canny 边缘检测图，然后输入 ControlNet，用 Prompt 咒语控制生成效果。
将毛坯房图、Canny 边缘检测图、咒语 Prompt、ControlNet 生成图，做成一页海报，发到群里。作业：ControlNet 的 N 种玩法





## 设置推理代码

参考官网的文件

```python
import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('configs/controlnet/controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()

prompt = 'Room with blue walls and a yellow ceiling.'
control_url = 'https://user-images.githubusercontent.com/28132635/230288866-99603172-04cb-47b3-8adb-d1aa532d1d2c.jpg'
control_img = mmcv.imread(control_url)
control = cv2.Canny(control_img, 100, 200)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)

output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'sample_{idx}.png')
controls = output_dict['controls']
for idx, control in enumerate(controls):
    control.save(f'control_{idx}.png')
```

我们只需要进行修改提示词还有图片就可以。由于众所周知的原因，可能会下载模型失败，所以，需要自己设置http proxy。修改代码如下。

```python
import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image
import os
from mmagic.registry import MODELS
from mmagic.utils import register_all_modules
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
register_all_modules()

cfg = Config.fromfile('configs/controlnet/controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()

prompt = 'Room with blue walls and a yellow ceiling.'
control_url = 'https://user-images.githubusercontent.com/28132635/230288866-99603172-04cb-47b3-8adb-d1aa532d1d2c.jpg'
control_url = './test.jpeg'

control_img = mmcv.imread(control_url)
control = cv2.Canny(control_img, 100, 200)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)

output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'sample_{idx}.png')
controls = output_dict['controls']
for idx, control in enumerate(controls):
    control.save(f'control_{idx}.png')
```

首先是读入图片，然后canny边缘检测，之后按照c的尺度，进行三个连接着一起，变成image。然后对图像进行推理。效果如下。

![](https://fastly.jsdelivr.net/gh/weijia99/blog_image@main/16869011801241686901180048.png)

canny：

![](https://fastly.jsdelivr.net/gh/weijia99/blog_image@main/16869012192891686901219033.png)



最终效果：

![](https://fastly.jsdelivr.net/gh/weijia99/blog_image@main/16869012463001686901246225.png)