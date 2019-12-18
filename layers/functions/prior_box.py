import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = 300 # 输入RFBNet的图像尺度 300
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = 4
        self.variance = [0.1, 0.2]
        self.feature_maps =  [38, 19, 10, 5, 3, 1]
        # 两个1:1长宽比的anchor
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.clip = True
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = [] # anchor
        # 对特征金字塔的各个检测分支，每个feature map上each-pixel都做密集anchor采样
        # k: 0 1 2 3 4 5
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        # 遍历特征图
        for k, f in enumerate(self.feature_maps):     # k: 0, f: 38
            for i, j in product(range(f), repeat=2):  # 笛卡尔乘积，可以开始密集anchor采样了
                # i   = 0 1 ... 37
                # j   = 0 1 ... 37
                if k < 4:
                    # steps: 8,   16,   32,    64,     100, 300
                    # f_k ： 37.5 18.75 9.375  4.6875  3.0  1.0
                    f_k = self.image_size / self.steps[k]
                    # unit center x,y
                    # 从 0.5 到 37.5
                    cx = (j + 0.5) / f_k # 除以 f_k 达到归一化的目的
                    cy = (i + 0.5) / f_k # 以上三步操作，就相当于从feature map位置映射至原图，float型

                    s_k = self.min_sizes[k]/self.image_size
                    # [cx, cy, w, h]
                    # 由于人的比例是 w / h 在 0.1-1之间，h更大一些
                    mean += [cx, cy, s_k * sqrt(0.2), s_k / sqrt(0.2)]
                    mean += [cx, cy, s_k * sqrt(0.5), s_k / sqrt(0.5)]
                    mean += [cx, cy, s_k * sqrt(0.8), s_k / sqrt(0.8)]
                    mean += [cx, cy, s_k * sqrt(1.1), s_k / sqrt(1.1)]

                    # aspect_ratio: 0.4
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_prime * sqrt(0.6), s_k_prime / sqrt(0.6)]
                else:
                    f_k = self.image_size / self.steps[k]
                    cx = (j + 0.5) / f_k  # 除以 f_k 达到归一化的目的
                    cy = (i + 0.5) / f_k  # 以上三步操作，就相当于从feature map位置映射至原图，float型
                    s_k = self.min_sizes[k] / self.image_size
                    mean += [cx, cy, s_k * sqrt(0.3), s_k / sqrt(0.3)]
                    mean += [cx, cy, s_k * sqrt(0.6), s_k / sqrt(0.6)]
                    mean += [cx, cy, s_k * sqrt(0.9), s_k / sqrt(0.9)]
                    mean += [cx, cy, s_k * sqrt(1.2), s_k / sqrt(1.2)]

            # 总结：
            # 1 feature map上each-pixel对应4 / 6个anchor，长宽比：2:1 + 1:2 + 1:3 + 3:1 + 1:1 + 1:1，后两个1:1的anchor对应的尺度有差异；
            # 2 跟SSD还是严格对应的，每个feature map上anchor尺度唯一(2:1 + 1:2 + 1:3 + 3:1 + 1:1这五个anchor的尺度还是相等的，面积相等)，仅最后的1:1 anchor尺度大一点；
            # 3 所有feature map上所有预定义的不同尺度、长宽比的anchor保存至mean中；

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

if __name__ == '__main__':
    cfg = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    'variance' : [0.1, 0.2],

    'clip' : True,
    }
    priorbox = PriorBox(cfg)

    with torch.no_grad():
        priors = priorbox.forward()
    print(priors.shape)
