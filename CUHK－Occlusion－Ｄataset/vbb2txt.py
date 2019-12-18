import os, glob
import cv2
from scipy.io import loadmat
import numpy as np

vbb_dir=r'C:\Users\I354762\Desktop\dataset\labels'
images_dir = r'C:\Users\I354762\Desktop\dataset\images'

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0] + box[2]/2.0
    y = box[1] + box[3]/2.0
    w = box[2]
    h = box[3]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def visualizeBox(im, box, imname):
    # 在图像上画出圈定的方框
    cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), (0,0,255), 2)
    cv2.imwrite(imname,im)

for f in sorted(glob.glob('C:/Users/I354762/Desktop/dataset/labels/*.vbb')):
    vbb = loadmat(f)
    # 每一帧的object信息：id，pos，posv，occlusion，lock
    objLists = vbb['A'][0][0][1][0]
    # 所有类别
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    if 'person' in objLbl:
        person_index_list=np.where(np.array(objLbl)=="person")[0]
    elif 'object' in objLbl:
        person_index_list=np.where(np.array(objLbl)=="object")[0]
    i = 0 
    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            frame_name = "images_" + os.path.basename(f).split('.')[0] +"_" +str(i)+".png"
            boxes = []
            for Id, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                # matlab从1开始
                Id = int(Id[0][0]) - 1
                if not Id in person_index_list:
                    continue
                pos = pos[0].tolist()
                boxes.append(pos)
            if boxes:
                full_path ="C:/Users/I354762/Desktop/dataset/txts/%s"%("images_" + os.path.basename(f).split('.')[0] +"_" +str(i)+".txt")
                txt_file = open(full_path, 'a')
                im = cv2.imread(os.path.join(images_dir, frame_name))
                imsize = im.shape
                for box in boxes:
                    convert_size = convert(imsize, box)
                    txt_file.write('0' + ' ' + str(convert_size[0]) + ' ' + str(convert_size[1]) + ' ' + str(convert_size[2]) + ' ' +str(convert_size[3]))
                    txt_file.write('\n')
                    visualizeBox(im, box, frame_name)
        i += 1
    print(f)