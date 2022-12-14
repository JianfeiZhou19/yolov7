import cv2
import albumentations as A
import xml.etree.ElementTree as ET
import os
import string

def voc_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2, mode='voc'):
    """Visualizes a single bounding box on the image"""
    if mode == 'voc':
        x_min, y_min, x_max, y_max = [round(i) for i in bbox]
    if mode == 'yolo':
        h, w, _ = img.shape
        w_half_len = (bbox[2] * w) / 2
        h_half_len = (bbox[3] * h) / 2
        x_min = int((bbox[0] * w) - w_half_len)
        y_min = int((bbox[1] * h) - h_half_len)
        x_max = int((bbox[0] * w) + w_half_len)
        y_max = int((bbox[1] * h) + h_half_len)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (0, 0, 255), -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=(255, 255, 255), 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name, mode, path='test.jpg'):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, mode=mode)
    cv2.imwrite(path, img)


def augmentation_voc_image(image_file, anno_file, transformers, classes):
    img = cv2.imread(image_file)
    tree = ET.parse(anno_file)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    bboxes = []
    category_ids=[]
    for obj in root.findall('object'):
        label = obj.find("name").text
        index = classes.index(label)
        pil_bbox = [int(x.text) for x in obj.find("bndbox")]
        bboxes.append(pil_bbox)
        category_ids.append(index)
    aug_res = transformers(image=img, bboxes=bboxes, category_ids=category_ids)
    aug_res['width'] = width
    aug_res['height'] = height
    return aug_res

def augmentation_yolo_image(image_file, anno_file, transformers, classes):
    img = cv2.imread(image_file)
    bboxes = []
    category_ids=[]
    with open(anno_file) as f:
        for line in f.readlines():
            line = line.strip('\n')
            label_index = int(line.split(' ')[0])
            label = classes[label_index]
            yolo_bbox = [float(i) for i in line.split(' ')[1:]]
            bboxes.append(yolo_bbox)
            category_ids.append(label_index)
    aug_res = transformers(image=img, bboxes=bboxes, category_ids=category_ids)
    return aug_res

def aug_voc2yolo(image_dir, label_dir, image_save_dir, label_save_dir, transformers, classes):
    file_names = os.listdir(image_dir)
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    for img in file_names:
        fname, ext = os.path.splitext(img)
        img_file = os.path.join(image_dir, img)
        anno_file = os.path.join(label_dir, fname + ".xml")
        aug_res = augmentation_voc_image(img_file, anno_file, transformers=transformers, classes=classes)
        aug_file_name = fname + "_aug" + ext
        img_save = os.path.join(image_save_dir, aug_file_name)
        # write img
        cv2.imwrite(img_save, aug_res['image'])
        # write label
        bboxes = aug_res['bboxes']
        category_ids = aug_res['category_ids']
        result = []
        for bbox, index in zip(bboxes, category_ids):
            yolo_bbox = voc_to_yolo_bbox(bbox, aug_res['width'], aug_res['height'])
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")
        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(label_save_dir, f"{fname}_aug.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

def augmentation_yolo(image_dir, label_dir, image_save_dir, label_save_dir, transformers, classes):
    file_names = os.listdir(image_dir)
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    for img in file_names:
        fname, ext = os.path.splitext(img)
        img_file = os.path.join(image_dir, img)
        anno_file = os.path.join(label_dir, fname + ".txt")
        aug_res = augmentation_yolo_image(img_file, anno_file, transformers=transformers, classes=classes)
        aug_file_name = fname + "_aug" + ext
        img_save = os.path.join(image_save_dir, aug_file_name)
        # write img
        cv2.imwrite(img_save, aug_res['image'])
        # write label
        bboxes = aug_res['bboxes']
        category_ids = aug_res['category_ids']
        # check by visualization
        # visualize(aug_res['image'], bboxes, category_ids, classes, mode='yolo')
        # import pdb;pdb.set_trace()
        result = []
        for bbox, index in zip(bboxes, category_ids):
            # convert data to yolo string
            bbox_string = " ".join([str(x) for x in bbox])
            result.append(f"{index} {bbox_string}")
        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(label_save_dir, f"{fname}_aug.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

def get_transforms(mode='yolo'):
    transformers =  A.Compose([
            A.HueSaturationValue(p=0.5),
            A.RandomBrightness(limit=(-0.2,-0.1), always_apply=False, p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                # A.MotionBlur(p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.Blur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=(1,3),p=0.5),
                # A.IAASuperpixels(p_replace=0.1,n_segments=500,p=0.5),
                # A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.Cutout(num_holes=30, max_h_size=20, max_w_size=20, fill_value=128, p=0.5),
            ], p=0.2),
            A.ShiftScaleRotate(rotate_limit=30, p=0.8),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
        ], p=1,
        bbox_params=A.BboxParams(format=mode, label_fields=['category_ids']))
    return transformers


if __name__ == '__main__':
    root_path = "/vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang/images/train/"
    save_path = "/vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang_aug/images/train/"
    label_root = "/vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang/labels/train/"
    yolo_save = "/vepfs/Perception/Users/jianfei/self_exp/yolov7/dataset/qiangang_aug/labels/train/"

    classes = [str(int(i)) for i in range(10)] + list(string.ascii_uppercase) + ["plate"]
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(yolo_save, exist_ok=True)
    file_names = os.listdir(root_path)
    trans = get_transforms(mode='yolo')
    # ??????????????????
    augmentation_yolo(root_path, label_root, save_path, yolo_save, trans, classes)
    
        