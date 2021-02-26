from glob import glob
from skimage.io import imread, imsave
import numpy as np
import json
import os
import os.path as path
from numba import njit


def get_classes():
    classes_csv = '''0,unlabeled,0,0,0,0,255,0
    1,ambiguous,111,74,0,0,255,0
    2,sky,70,130,180,1,0,0
    3,road,128,64,128,1,1,0
    4,sidewalk,244,35,232,1,2,0
    5,railtrack,230,150,140,0,255,0
    6,terrain,152,251,152,1,3,0
    7,tree,87,182,35,1,4,0
    8,vegetation,35,142,35,1,5,0
    9,building,70,70,70,1,6,0
    10,infrastructure,153,153,153,1,7,0
    11,fence,190,153,153,1,8,0
    12,billboard,150,20,20,1,9,0
    13,trafficlight,250,170,30,1,10,1
    14,trafficsign,220,220,0,1,11,0
    15,mobilebarrier,180,180,100,1,12,0
    16,firehydrant,173,153,153,1,13,1
    17,chair,168,153,153,1,14,1
    18,trash,81,0,21,1,15,0
    19,trashcan,81,0,81,1,16,1
    20,person,220,20,60,1,17,1
    21,animal,255,0,0,0,255,0
    22,bicycle,119,11,32,0,255,0
    23,motorcycle,0,0,230,1,18,1
    24,car,0,0,142,1,19,1
    25,van,0,80,100,1,20,1
    26,bus,0,60,100,1,21,1
    27,truck,0,0,70,1,22,1
    28,trailer,0,0,90,0,255,0
    29,train,0,80,100,0,255,0
    30,plane,0,100,100,0,255,0
    31,boat,50,0,90,0,255,0'''

    classes = []

    for line in classes_csv.splitlines():
        col = line.split(',')
        class_info = {}
        class_info['id'] = int(col[0])
        class_info['classname'] = col[1]
        class_info['red'] = int(col[2])
        class_info['green'] = int(col[3])
        class_info['blue'] = int(col[4])
        class_info['class_eval'] = int(col[5])
        class_info['trainid'] = int(col[6])
        class_info['instance_eval'] = int(col[7])
        classes.append(class_info)

    return classes


def generate_images_info(split='train'):
    coco_images = []
    img_files = glob(f'data/gta5/{split}/img/**/*')
    for index, img_file in enumerate(img_files):
        coco_images.append({
            "id": index, "width": 1920, "height": 1080, "file_name": img_file
        })
    return coco_images


def generate_labeled_images(classes, split='train'):
    # todo: multi processing
    img_files = glob(f'data/gta5/{split}/cls/**/*')
    for index, img_file in enumerate(img_files):
        img = imread(img_file)
        target_img = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        for class_info in classes:
            target_img[(img[:, :, 0] == class_info['red']) & (img[:, :, 1] == class_info['green']) & (
                img[:, :, 2] == class_info['blue'])] = class_info['trainid']
        target_file = img_file.replace('cls', 'lblcls')
        os.makedirs(path.dirname(target_file), exist_ok=True)
        imsave(target_file, target_img)
        if index % 100 == 0:
            print(index)


@njit(nogil=True, cache=True)
def _find_bounding_box(mask, shape):
    pos_up = -1
    pos_left = np.iinfo(np.int64).max
    pos_down = -1
    pos_right = -1
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if mask[i, j] != 0:
                if pos_up == -1:
                    pos_up = i
                else:
                    pos_down = i
                pos_left = min(pos_left, j)
                pos_right = max(pos_right, j)
    return pos_left, pos_up, pos_right, pos_down


def generate_bbox(classes, images_info, split='train'):
    id = 0
    annotations = []
    categories = []
    used_classes = set()
    for index, image_info in enumerate(images_info):
        inst_file = image_info['file_name'].replace('img', 'inst').replace('jpg', 'png')
        inst_img = imread(inst_file)
        for i in range(256 * (np.max(inst_img[:, :, 1]) + 1)):
            g = i // 256
            b = i % 256
            mask = (inst_img[:, :, 0] != 0) & (
                inst_img[:, :, 1] == g) & (inst_img[:, :, 2] == b)
            area = np.count_nonzero(mask)
            if area > 0:
                pos_left, pos_up, pos_right, pos_down = _find_bounding_box(
                    mask, mask.shape)
                y, x = np.nonzero(mask)
                category_id = classes[inst_img[y[0], x[0], 0]]['trainid']
                if category_id == 255:
                    continue
                annotations.append({
                    'id': id,
                    'image_id': image_info['id'],
                    'category_id': category_id,
                    'bbox': [pos_left, pos_up, pos_right - pos_left, pos_down - pos_up],
                    'iscrowd': 0,
                    'area': area,
                    'segmentation_color': [int(rgb) for rgb in inst_img[y[0], x[0]]]
                })
                id += 1
                used_classes.add(inst_img[y[0], x[0], 0])
        if index % 100 == 0:
            print(index)

    for class_id in used_classes:
        categories.append({
            'id': classes[class_id]['trainid'],
            'name': classes[class_id]['classname']
        })

    json_obj = {
        'images': images_info,
        'categories': categories,
        'annotations': annotations
    }

    with open(f'data/gta5/{split}/inst.json', 'w', encoding='utf-8') as f:
        json.dump(json_obj, f)


# generate labeled images (for sematic segmentation)
# generate_labeled_images(get_classes())
# generate_labeled_images(get_classes(), 'val')
# # generate bound box (for instance segmentation)
# generate_bbox(get_classes(), generate_images_info())
generate_bbox(get_classes(), generate_images_info('val'), 'val')
