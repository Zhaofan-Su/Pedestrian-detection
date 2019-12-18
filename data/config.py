import os.path


# 自己进行修改
VOCroot = '/home/gongyiqun/data/person' # path to VOCdevkit root dir


#RFB CONFIGS
VOC_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    'variance' : [0.1, 0.2],

    'clip' : True,
}


