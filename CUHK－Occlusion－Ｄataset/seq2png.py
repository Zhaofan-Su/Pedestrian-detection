import os
import glob
import cv2 as cv
 
# 保存图片以指定名字到指定位置
def save_img(fn, i, frame):
    cv.imwrite('{}_{}_{}.png'.format(
        out_dir, os.path.basename(fn).split('.')[0], i), frame)
    
    
#输出图片位置    
out_dir=r'C:\Users\Desktop\dataset\images'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#seq文件位置

for fn in sorted(glob.glob('C:/Users/Desktop/dataset/*.seq')):
    cap = cv.VideoCapture(fn)
    i = 0
    while True:
        # ret为标志位，bool型，是否读到数据，frame为视频帧图像数据
        ret, frame = cap.read()
        if not ret:
            break
        save_img(fn, i, frame)
        i += 1
    print(fn)