import cv2
import numpy as np
import os.path
import pickle
from tqdm import tqdm
import math
# ResizeShape_width = 480
# ResizeShape_height = 384

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

    if cls not in class_list:
        print('The current class is wrong.')
    else:
        return str(class_list.index(cls))


# 获取样本信息
def get_data(input_path, PATH_NOTICE=True):
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

            if PATH_NOTICE:
                filename = filename[1:]

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if classMapping(class_name) not in class_mapping:
                class_mapping[classMapping(class_name)] = class_name  # 注意，mapping设置

            if filename not in all_imgs:

                all_imgs[filename] = {}
                # print (os.path.isfile(filename))
                img = cv2.imread(filename)
                # print(img.shape)
                (rows, cols, channels) = img.shape

                # print(img.shape)
                # resize_img = cv2.resize(img, (ResizeShape_width, ResizeShape_height))
                # height_ratio = ResizeShape_height / rows
                # width_ratio = ResizeShape_width / cols

                # (rows, cols, channels) = resize_img.shape
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
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['channel'] = channels
                # all_imgs[filename]['width_ratio'] = width_ratio
                # all_imgs[filename]['height_ratio'] = height_ratio
                # all_imgs[filename]['pixel'] = resize_img
                all_imgs[filename]['bboxes'] = []
                # FLAG = np.random.randint(0, 9)
                # if FLAG > 2:
                #     all_imgs[filename]['imageset'] = 'train'
                # else:
                #     all_imgs[filename]['imageset'] = 'validation'

            # 每个图有多个框
            # x1 = max(int(math.ceil((float(x1) + 1) * width_ratio - 1)), 0)
            # x2 = min(int(math.floor(float(x2) * width_ratio)), cols - 1)
            # y1 = max(int(math.ceil((float(y1) + 1) * height_ratio - 1)), 0)
            # y2 = min(int(math.floor(float(y2) * height_ratio)), rows - 1)

            all_imgs[filename]['bboxes'].append(
                {'class': classMapping(class_name), 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})

            # all_imgs[filename]['bboxes'].append({'class': classMapping(class_name),
            #                                      'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        print('The max width is {}.'.format(image_width_max))
        print('The min width is {}.'.format(image_width_min))
        print('The average width is {}.'.format(image_width_sum / image_num))

        print('The max height is {}.'.format(image_height_max))
        print('The min height is {}.'.format(image_height_min))
        print('The average height is {}.'.format(image_height_sum / image_num))

        return all_data, class_mapping, classes_count


def generatePickle(anno_file, save_file):
    all_images, class_mapping, classes_count = get_data(anno_file, False)
    print(len(all_images))
    print(classes_count)
    print(class_mapping)

    for i in range(len(all_images)):
        all_images[i]['filepath'] = all_images[i]['filepath'][1:]

    # 把样本的信息保存起来，以后可以快速读取，以免等待读图
    Output = open(save_file, 'wb')
    pickle.dump(all_images, Output, -1)
    pickle.dump(class_mapping, Output, 0)
    pickle.dump(classes_count, Output, 0)
    Output.close()


def main():
    # 注释文件
    VAL_ANNOTATIONS_FILE = '../sample/coco2017_annotations_val_set.txt'
    VAL_SAVE_PICKLE_FILE = '../sample/coco2017_val_sample.pkl'
    generatePickle(VAL_ANNOTATIONS_FILE,VAL_SAVE_PICKLE_FILE)

    TRAIN_ANNOTATIONS_FILE = '../sample/coco2017_annotations_train_set.txt'
    TRAIN_SAVE_PICKLE_FILE = '../sample/coco2017_train_sample.pkl'
    generatePickle(TRAIN_ANNOTATIONS_FILE, TRAIN_SAVE_PICKLE_FILE)


if __name__ == '__main__':
    main()
