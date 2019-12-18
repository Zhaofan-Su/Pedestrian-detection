import os, glob

txt_path = "C:/Users/Desktop/dataset/train.txt"
txt_file = open(txt_path,'a')
for f in sorted(glob.glob('C:/Users/Desktop/dataset/images/*.png')):
    full_name = os.path.basename(f)
    file_name = full_name.split("/")[-1]
    txt_file.write(file_name + '\n')
