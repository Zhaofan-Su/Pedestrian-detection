## 目标检测：行人检测


概述
---

机器视觉要解决的中心问题就是如何从图像中解析出计算机可以理解的信息。计算机对于图像的理解主要有三个层次：分类、检测和分割。目标检测处于图像理解的中层次。

图像分类关心的是整体，给出的是整张图片的内容描述，而目标检测则关注特定的物体目标，要求同时获得这一目标的类别信息和位置信息。目标检测是对图片前景和背景的理解，我们需要从背景中分离出目标，并确定这一目标的类别和位置。


项目运行方式
---

进入项目文件夹
```sh
cd Pedestrian-detection
```

`demo.py`文件用于效果演示
```sh
python demo.py 
```
`demo.py`有如下参数可供使用
| 参数 | 默认值 | 描述 |
| ------ | ------ | ------ |
| -v[--version] | RFB_vgg | 选择版本 | 
| -s[--size] | 300 | 输入参数的大小 |
| -d[--dataset] | VOC | 选择数据集类型 |
| -m[--trained_model] | r'weights/7690.pth' | 已训练的模型 |
| --save_folder | eval/ | 数据保存路径 |
| --video | True | 是否使用视频 |

`train_RFB.py`文件用于模型训练
```sh
python train_RFB.py
```
`train_RFB.py`重要参数介绍
| 参数 | 默认值 | 描述 |
| ------ | ------ | ------ |
| -b[--batch_size] | 64 | 定义训练的batch size |
| --cuda | True | 是否使用cuda训练模型 |
| --ngpu | 4 | GPU个数 |
| --lr[--learning-rate] | 1e-2 | 学习率 |
| …… | …… | …… |
使用如下命令查看所有参数
```sh
python train_RFB.py -h
```
`test_RFB.py`文件用于测试模型
```sh
python test_RFB.py
```
`test_RFB.py`重要参数介绍
| 参数 | 默认值 | 描述 |
| ------ | ------ | ------ |
| -m[--trained_model] | weights/7690.pth | 已训练的模型 |
| --cuda | True | 是否使用cuda测试模型 |
| --cpu | False | 是否使用cpu测试模型 |
| …… | …… | …… |
使用如下命令查看全部参数
```sh
python test_RFB.py -h
```
<br/>
<br/>


---

本项目使用[CUHK Occlusion Dataset](http://mmlab.%3C/b%3Eie.cuhk.edu.hk/datasets/cuhk_occlusion/index.html) 

[行人检测数据集打包下载](https://pan.baidu.com/s/1o8aanoQ) &nbsp;密码：xkka