import os

input_dir = "/vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang/images"
label_dir = "/vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang/labels"

for file in os.listdir(input_dir):
    fname, ext = os.path.splitext(file)
    img_file = os.path.join(input_dir, file)
    if not os.path.exists(os.path.join(label_dir, fname+'.txt')):
        os.system("rm -rf "+img_file)