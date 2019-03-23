import cv2
import numpy as np
import os.path
import pickle
import time
from tqdm import tqdm
import math
from help import show_picture as sp

ResizeShape_width = 576
ResizeShape_height = 480
Normalize = False

CNNModel = 'vgg16'
# 用于存储groundtruth的目录
GroundTruthPath = '../result/' + CNNModel + '/ground-truth/'
if not os.path.exists(GroundTruthPath):
    os.makedirs(GroundTruthPath)


def classMapping(cls):
    class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                  'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench',
                  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                  'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                  'bowl', 'banana', 'apple', 'sandwich', 'orange',
                  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                  'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                  'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                  'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                  'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']
    # class_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    #               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    #               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    #               'sheep', 'sofa', 'train', 'tvmonitor']
    if cls not in class_list:
        print('The current class is wrong.')
    else:
        return str(class_list.index(cls))


# 获取样本信息
def get_data(input_path):
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    image_num = 0
    image_width_sum = 0
    image_width_max = 0
    image_width_min = 2000
    image_height_sum = 0
    image_height_max = 0
    image_height_min = 2000

    # 把样本以字典形式存储
    with open(input_path, 'r') as f:
        print('Parsing annotation files')

        for line in tqdm(f):
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            # 注意，mapping设置
            if classMapping(class_name) not in class_mapping:
                class_mapping[classMapping(class_name)] = class_name

            if filename not in all_imgs:
                all_imgs[filename] = {}
                img = cv2.imread(filename)
                (rows, cols, channels) = img.shape
                # img = img.reshape((img.shape[0], img.shape[1], 1))
                resize_img = cv2.resize(img, (ResizeShape_width, ResizeShape_height))
                if Normalize:
                    resize_img = resize_img.astype('float32')
                    resize_img /= 127.5
                    resize_img -= 1.

                height_ratio = ResizeShape_height / rows
                width_ratio = ResizeShape_width / cols

                (rows, cols, channels) = resize_img.shape
                # print(resize_img.shape)
                # input('Stop!')
                if channels != 3:
                    print('There is a mistake.')
                    os._exit(0)

                image_num += 1
                image_width_sum += cols
                image_height_sum += rows

                if cols > image_width_max:
                    image_width_max = cols
                if cols < image_width_min:
                    image_width_min = cols
                if rows > image_height_max:
                    image_height_max = rows
                if rows < image_height_min:
                    image_height_min = rows

                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['channel'] = channels
                all_imgs[filename]['width_ratio'] = width_ratio
                all_imgs[filename]['height_ratio'] = height_ratio
                all_imgs[filename]['pixel'] = resize_img
                all_imgs[filename]['bboxes'] = []

            x1 = round(float(x1) * width_ratio, 2)
            x2 = round(float(x2) * width_ratio, 2)
            y1 = round(float(y1) * height_ratio, 2)
            y2 = round(float(y2) * height_ratio, 2)
            # 每个图有多个框
            all_imgs[filename]['bboxes'].append({'class': classMapping(class_name),
                                                 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})

        print('The max width is {}.'.format(image_width_max))
        print('The min width is {}.'.format(image_width_min))
        print('The average width is {}.'.format(image_width_sum / image_num))

        print('The max height is {}.'.format(image_height_max))
        print('The min height is {}.'.format(image_height_min))
        print('The average height is {}.'.format(image_height_sum / image_num))

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        for i in range(len(all_data)):
            # coords = []
            # # all_data[i]['pixel'])
            # for bn in range(len(all_data[i]['bboxes'])):
            #     coords.append([all_data[i]['bboxes'][bn]['x1'], all_data[i]['bboxes'][bn]['y1'],
            #                    all_data[i]['bboxes'][bn]['x2'], all_data[i]['bboxes'][bn]['y2']])
            #     # print(tmp_img.shape)
            #     # print(coords1)
            # sp.show_pic(all_data[i]['pixel'], coords)

            # 由于文件从不同目录读取来的，以免重名，给写出来的文件加上前缀
            # tmp_list = all_data[i]['filepath'].strip().split('/')
            # if 'R_HDS5_test_set' in tmp_list:
            #     mark = 'R_HDS5_'
            # elif 'S_HDS5_test_set' in tmp_list:
            #     mark = 'S_HDS5_'
            # elif 'S_MNIST_test_set' in tmp_list:
            #     mark = 'S_MNIST_'
            # else:
            #     print('Can not find the test sample.')
            #     exit(0)

            ground_truth_txt = GroundTruthPath + os.path.basename(all_data[i]['filepath']).split('.')[0] + ".txt"

            # 把groundtruth写到文件中去
            with open(ground_truth_txt, 'w') as f:
                for b in range(len(all_data[i]['bboxes'])):
                    f.write('{} {} {} {} {} \n'.format(all_data[i]['bboxes'][b]['class'],
                                                       all_data[i]['bboxes'][b]['x1'],
                                                       all_data[i]['bboxes'][b]['y1'],
                                                       all_data[i]['bboxes'][b]['x2'],
                                                       all_data[i]['bboxes'][b]['y2']))

        return all_data, class_mapping, classes_count


def main():
    # 注释文件
    ANNOTATIONS_FILE = '../sample/coco2017_annotations_val_set.txt'
    # 返回样本的所有信息，各类别数目，以及类别映射
    start = time.time()
    all_images, class_mapping, classes_count = get_data(ANNOTATIONS_FILE)
    cost_time = time.time() - start
    # 算下读图时间
    print('The cost time for reading an image is', str(cost_time / len(all_images)) + '.')
    # print(all_images[0])
    # print(classes_count)
    # print(class_mapping)

    # 由于官方没有test的标注，这里分别用test和val进行测试
    # 把样本的信息保存起来，以后可以快速读取，加速测试，以免等待读图
    Output = open('../sample/coco2017_valtest_sample.pkl', 'wb')
    pickle.dump(all_images, Output, -1)
    pickle.dump(class_mapping, Output, 0)
    pickle.dump(classes_count, Output, 0)
    Output.close()


if __name__ == '__main__':
    main()
