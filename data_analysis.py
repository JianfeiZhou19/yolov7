import os
import string
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def count_labels(folder, classes):
    label_list = os.listdir(folder)
    label_count = defaultdict(int)
    for file in label_list:
        file_path = os.path.join(folder, file)
        if not file.endswith(".txt") or not os.path.exists(file_path):
            continue
        with open(file_path) as f:
            for line in f.readlines():
                label_count[classes[int(line.split(' ')[0])]] += 1
    return sorted(label_count.items(), key=lambda item:item[0])


if __name__ == "__main__":
    classes = [str(int(i)) for i in range(10)] + list(string.ascii_uppercase) + ["plate"]
    print(count_labels("/vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang/labels/train/", classes))