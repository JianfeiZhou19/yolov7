import xml.etree.ElementTree as ET
import glob
import os
import json
import string 

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]

def convers_voc2yolo(input_dir, output_dir, image_dir, classes):
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(input_dir, '*.xml'))
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue

        result = []

        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in classes:
                # classes.append(label)
                if label == "palte":
                    label = "plate"
                # continue
            index = classes.index(label)
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

    # generate the classes file as reference
    with open('classes.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(classes))

def split(dataset_dir, train_ratio=0.9):
    img_list = os.listdir(os.path.join(dataset_dir, "images"))
    os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "val"), exist_ok=True)
    for img in img_list[:int(train_ratio*len(img_list))]:
        os.system("mv "+os.path.join(dataset_dir, "images", img) + " " + os.path.join(dataset_dir, "images", "train"))
        os.system("mv "+os.path.join(dataset_dir, "labels", img.replace(".jpg", '.txt')) + " " + os.path.join(dataset_dir, "labels", "train"))

    for img in img_list[int(train_ratio*len(img_list)):]:
        os.system("mv "+os.path.join(dataset_dir, "images", img) + " " + os.path.join(dataset_dir, "images", "val"))
        os.system("mv "+os.path.join(dataset_dir, "labels", img.replace(".jpg", '.txt')) + " " + os.path.join(dataset_dir, "labels", "val"))


if __name__ == '__main__':
    # nums_recon
    # classes = [str(int(i)) for i in range(10)]
    # nums qian gang 
    classes = [str(int(i)) for i in range(10)] + list(string.ascii_uppercase) + ["plate"]
    # helmet detection
    # classes = ["crack","bar","round","lop","icf"]
    # input_dir = "/vepfs/Perception/Users/jianfei/self_exp/detectiondata/qiangang_aug/label/"
    # output_dir = "/vepfs/Perception/Users/jianfei/self_exp/detectiondata/qiangang_aug/labels/"
    # image_dir = "/vepfs/Perception/Users/jianfei/self_exp/detectiondata/qiangang_aug/image/"
    # convers_voc2yolo(input_dir, output_dir, image_dir, classes)
    # os.system('cp -r /vepfs/Perception/Users/jianfei/self_exp/detectiondata/qiangang/image/ /vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang/images/')
    # os.system('cp -r /vepfs/Perception/Users/jianfei/self_exp/detectiondata/qiangang/labels/ /vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang/labels/')
    split("/vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang/")