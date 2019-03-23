from help import handle_sample as hs
from help import get_trainval_data as getDATA
from help import get_img_cv2 as getIMG
import math
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import os

ResizeWidth = 576
ResizeHeight = 480
CHANNEL = 3
DSR = 16
CLASS = 80
ABS_LOCATION = False  # Todo: 每次运行要改的地方(1/3)
CrossRatio = 1.0  # Todo: 每次运行要改的地方(2/3)


def getInformation(data):
    X_path = []
    y = []
    original_size = []
    for i in range(len(data)):
        X_path.append(data[i]['filepath'])
        y.append(data[i]['bboxes'])
        original_size.append([data[i]['height'], data[i]['width'], data[i]['channel']])
    return X_path, y, original_size


def trainStandardScaler(label_r):
    label = label_r.reshape((label_r.shape[0] * label_r.shape[1] * label_r.shape[2], label_r.shape[3]))
    scale = StandardScaler().fit(label)
    label = scale.transform(label)
    label = label.reshape((label_r.shape[0], label_r.shape[1], label_r.shape[2], label_r.shape[3]))
    return label, scale


print('Load training samples...')
TRAIN_ANNOTATIONS_FILE = '../sample/coco2017_annotations_train_set.txt'
# 返回样本的所有信息，各类别数目，以及类别映射
train_images, class_mapping, train_classes_count = getDATA.get_data(TRAIN_ANNOTATIONS_FILE, False)
print(len(train_images), 'training samples are loaded.')

X_train_path, Y_train, train_size = getInformation(train_images)

num = 0
for m in tqdm(range(len(Y_train))):

    height_ratio = ResizeHeight / train_size[m][0]
    width_ratio = ResizeWidth / train_size[m][1]

    for n in range(len(Y_train[m])):
        Y_train[m][n]['x1'] = round(float(Y_train[m][n]['x1']) * width_ratio, 2)
        Y_train[m][n]['x2'] = round(float(Y_train[m][n]['x2']) * width_ratio, 2)
        Y_train[m][n]['y1'] = round(float(Y_train[m][n]['y1']) * height_ratio, 2)
        Y_train[m][n]['y2'] = round(float(Y_train[m][n]['y2']) * height_ratio, 2)
        if Y_train[m][n]['x2'] < Y_train[m][n]['x1'] or Y_train[m][n]['y2'] < Y_train[m][n]['y1']:
            print('The bbox is wrong.')
            os._exit(0)
        if Y_train[m][n]['x2'] == Y_train[m][n]['x1'] or Y_train[m][n]['y2'] == Y_train[m][n]['y1']:
            print('The bbox is a line.')
            num += 1

print('There are {} bboxes which are lines.'.format(num))

label_c, label_bbox = hs.relabel_2D_try_5(Y_train, ResizeHeight, ResizeWidth, DSR, CLASS, ABS_LOCATION, CrossRatio)
label_bbox, train_scale = trainStandardScaler(label_bbox)
# Todo: 每次运行要改的地方(3/3)
save_train_scale = open('../sample/train_scale_try_5_offset.pkl', 'wb')
pickle.dump(train_scale, save_train_scale)
save_train_scale.close()
