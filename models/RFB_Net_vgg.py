import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

# conv + bn + relu三剑客，in_planes：输出feature map通道数；out_planes：输出feature map通道数
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class LightVGG(nn.Module):
    def __init__(self, in_planes=3, stride=1, group=1, bn=True):
        super(LightVGG, self).__init__()

        self.conv1_1 = BasicConv(3,  16, kernel_size=3, stride=1, padding=1, groups=1, bn=bn)
        self.conv1_2 = BasicConv(16, 16, kernel_size=3, stride=1, padding=1, groups=1, bn=bn)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2

        self.conv2_1 = BasicConv(16, 32, kernel_size=3, stride=1, padding=1, groups=1, bn=bn)
        self.conv2_2 = BasicConv(32, 32, kernel_size=3, stride=1, padding=1, groups=1, bn=bn)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 4

        self.conv3_1 = BasicConv(32,  64, kernel_size=3, stride=1, padding=1, groups=1, bn=bn)
        self.conv3_2 = BasicConv(64, 64, kernel_size=1, stride=1, groups=1, bn=bn)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) #8

        self.conv4_1 = BasicConv(64, 128, kernel_size=3, stride=1, padding=1, groups=1, bn=bn)
        self.conv4_2 = BasicConv(128, 128, kernel_size=1, stride=1, groups=1, bn=bn)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16

        self.conv5_1 = BasicConv(128, 128, kernel_size=3, stride=1, padding=1, groups=1, bn=bn)
        self.conv5_2 = BasicConv(128, 128, kernel_size=1, stride=1, groups=1, bn=bn)

    def forward(self, x):   # torch.Size([2, 3, 300, 300])
        x = self.conv1_1(x) # torch.Size([2, 32, 300, 300])
        #x = self.conv1_2(x) # torch.Size([2, 32, 300, 300])
        x = self.pool1(x)   # torch.Size([2, 32, 150, 150])

        x = self.conv2_1(x) # torch.Size([2, 64, 150, 150])
        #x = self.conv2_2(x) # torch.Size([2, 64, 150, 150])
        x = self.pool2(x)   # torch.Size([2, 64, 75, 75])

        x = self.conv3_1(x) # torch.Size([2, 128, 75, 75])
        x = self.conv3_2(x) # torch.Size([2, 128, 75, 75])
        x = self.pool3(x)   # torch.Size([2, 128, 38, 38])

        x = self.conv4_1(x) # torch.Size([2, 256, 38, 38])
        x = self.conv4_2(x) # torch.Size([2, 256, 38, 38])
        s1 = x              # torch.Size([2, 256, 38, 38])
        x = self.pool4(x)   # torch.Size([2, 256, 38, 38])

        x = self.conv5_1(x) # torch.Size([2, 256, 19, 19])
        #x = self.conv5_2(x) # torch.Size([2, 256, 19, 19])
        return s1, x


class RFBNet(nn.Module):
    def __init__(self, phase, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.phase = phase             # 'train'
        self.num_classes = num_classes # 2 类
        self.indicator = 3             # 3 个 RFB 模块
        # vgg network
        self.base = LightVGG(group=1)
        # conv_4
        self.Norm = BasicRFB_a(128, 128, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        sources = list() # 存储待检测的特征图，这些分支后要接loc + conf模块，用于bbox预测
        loc = list()     # 存储物体定位的结果
        conf = list()    # 存储物体分类的结果

        s1, x = self.base(x)

        s = self.Norm(s1)  # BasicRFB_a层 self.Norm = BasicRFB_a(256,256,stride = 1,scale=1.0)
        sources.append(s) # 该层做bbox预测，可以结合fig 5理解，s并未参与后面的RFBNet网络的构建，而是从x操作的

        # apply extra layers and cache source layer outputs，就是新增的RFB模块，但这里和论文中fig 5有差异
        # fig 5中conv7_fc是做了个二分支，输入RFB + RFB stride2,
        # 但结合以下代码，其实是输入后的RFB返回的输出，再接RFB stride2，没有二分支
        for k, v in enumerate(self.extras):
            # k: 0 1
            x = v(x) # torch.Size([2, 256, 19, 19])
            if k < self.indicator or k%2 ==0:
                # torch.Size([2, 256, 38, 38]) Conv4_3 -> BasicRFB_a
                # torch.Size([2, 256, 19, 19]) fc7 -> BasicRFB stride=1
                # torch.Size([2, 128, 10, 10])  BasicRFB stride=2
                # torch.Size([2, 128, 5, 5])    BasicRFB stride=2
                # torch.Size([2, 128, 3, 3])
                # torch.Size([2, 128, 1, 1])
                sources.append(x) # RFB模块后接loc + conf模块

        # apply multibox head to source layers，只有sources层上的分支也需要做bbox检测
        # source：需要做bbox预测的分支
        # loc：做bbox offsets预测的分支，可结合multibox函数理解
        # conf：做bbox cls预测的分支
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 相当于concate了所有检测分支上的检测结果
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds，这里直接返回softmax结果，就对应着confidence score
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes), # 等着loss计算再计算softmax loss吧 torch.Size([2, 11620, 21])
            )
        return output

extras = [128, 'S', 128, 'S', 128]

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    # extras: [256, 'S', 128, 'S', 128]
    for k, v in enumerate(cfg):
        # k:  0   1   2   3   4
        # v: 256 'S' 128 'S' 128
        if in_channels != 'S':
            if v == 'S':
                # 2.BasicRFB(in_channels=256, v=cfg[1+1]=128, scale=1.0)
                # 3.BasicRFB(in_channels=256, v=cfg[3+1]=128, scale=1.0)
                layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=2)]
            else:
                # 1.BasicRFB(in_channels=256, v=256, scale=1.0)
                layers += [BasicRFB(in_channels, v, scale = 1.0, visual=2)]
        in_channels = v

    # (3): BasicConv
    # (4): BasicConv
    # (5): BasicConv
    # (6): BasicConv
    # 附加层的最后四个普通卷积层，需要注意隔一个进行输出特征图
    layers += [BasicConv(128,64,kernel_size=1,stride=1)]
    layers += [BasicConv(64,128,kernel_size=3,stride=1)] ########### 输出特征图 ###########
    layers += [BasicConv(128,64,kernel_size=1,stride=1)]
    layers += [BasicConv(64,128,kernel_size=3,stride=1)] ########### 输出特征图 ###########

    return layers

def multibox(extra_layers, cfg, num_classes):
    """
    :param size: 300
    :param vgg: 基础模型
    :param extra_layers: 新添加层
    :param cfg: 特征图的每个像素点对应的anchor个数 [5, 5, 5, 5, 4, 4]
    :param num_classes: 2
    :return:
    """
    loc_layers = []
    conf_layers = []

    # 第一个要检测的特征图
    loc_layers += [nn.Conv2d(128, cfg[0] * 4, kernel_size=3, padding=1)]
    conf_layers +=[nn.Conv2d(128, cfg[0] * num_classes, kernel_size=3, padding=1)]

    i = 1
    indicator = 3 # 其实就是add_extras函数返回的新增层数extra_layers

    # 对应的参与检测的分支数，config.py内的feature_maps参数就很好理解了
    for k, v in enumerate(extra_layers):
        if k < indicator or k%2== 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i += 1
    return extra_layers, (loc_layers, conf_layers)

#################### 需要注意，这里要跟 prior_box.py 对应上
# number of boxes per feature map location，就是各个feature map上预定义的anchor数，可结合prior_box.py；理解
mbox = [5, 5, 5, 5, 4, 4] # number of boxes per feature map location


# RFBNet网络构建的接口函数
# net = build_net('train', img_dim, num_classes) # RFBNet网络构建
def build_net(phase, num_classes=2):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return

    return RFBNet(phase,
                  *multibox(add_extras(extras, 128), mbox, num_classes),
                  num_classes)
