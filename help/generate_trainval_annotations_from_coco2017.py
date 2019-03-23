#!/usr/bin/evn python
# coding:utf-8
import json
from tqdm import tqdm
import os

train_json = '../dataset/annotations/instances_train2017.json'
val_json = '../dataset/annotations/instances_val2017.json'
train_Jpeg = '../dataset/images/train2017/'
val_Jpeg = '../dataset/images/val2017/'
train_AnnoPath_txt = '../sample/coco2017_annotations_train_set.txt'
val_AnnoPath_txt = '../sample/coco2017_annotations_val_set.txt'


def generateAnn(data, ann_file, jpeg_path):
    num1 = 0
    with open(ann_file, 'w') as f:
        print('Train Converting Start...')

        for image in tqdm(data['images']):
            imgID = image['id']
            file_name = image['file_name']
            file_path = jpeg_path + file_name
            Height = image['height']
            Width = image['width']
            flag = 0
            for ann in data['annotations']:
                if ann['image_id'] == imgID:
                    flag = 1
                    cls = 0
                    for cat in data['categories']:
                        if cat['id'] == ann['category_id']:
                            cls = cat['name']
                    if cls == 0:
                        print('Can not find the category.')

                    x1 = round(ann['bbox'][0], 2)
                    y1 = round(ann['bbox'][1], 2)
                    x2 = round(x1 + ann['bbox'][2], 2)
                    y2 = round(y1 + ann['bbox'][3], 2)

                    if x2 > Width or y2 > Height:
                        print('There is something wrong with the bbox, error code:1.')
                        os._exit(0)
                    if x1 < 0 or y1 < 0:
                        print('There is something wrong with the bbox, error code:2.')
                        os._exit(0)
                    if x1 > x2 or y1 > y2:
                        print('There is something wrong with the bbox, error code:3.')
                        os._exit(0)

                    f.write('{},{},{},{},{},{}\n'.format(file_path, x1, y1, x2, y2, cls))

            if flag == 0:
                print('This image has no annotation.')
                num1 += 1
        print('There are {} image without annotations.'.format(num1))


if __name__ == '__main__':
    val_data = json.load(open(val_json, 'r'))
    print('There are {} validation samples.'.format(len(val_data['images'])))
    generateAnn(val_data, val_AnnoPath_txt, val_Jpeg)

    train_data = json.load(open(train_json, 'r'))
    print('There are {} training samples.'.format(len(train_data['images'])))
    generateAnn(train_data, train_AnnoPath_txt, train_Jpeg)
