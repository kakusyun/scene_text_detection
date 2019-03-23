import pickle
import random
import numpy as np
import os
from network import vgg16 as cnn
from network import lstm as rnn
from network import output as out
from keras.models import Model
from keras.utils import plot_model, np_utils
from keras.layers import Input, core, Lambda
from sklearn.preprocessing import StandardScaler
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from help import handle_model as hm
from keras.models import load_model
from keras import backend as K
from help import handle_sample as hs
from help import get_trainval_data as getDATA
from help import get_img_cv2 as getIMG
import math
import copy
from help import show_picture as sp  # Todo:测试框放缩的时候使用
import cv2  # Todo:测试框放缩的时候使用

FromPickle = True
ImageAugmented = False
AugSetNumber = 2
TrainBatchSize = 32  # 1024集群内存不足
ValBatchSize = 32
Epochs = 1000
TRY = '01'  # Todo: 每次运行要改的地方(1/3)
DSR = 16  # down sampling ratio下采样，下采样2的时候，BatchSize=256
CNNModel = 'vgg16_' + str(DSR) + '_' + TRY
ModelPath = './model/' + CNNModel + '/'
# 保存的模型位置和名称，名称根据epoch和精度变化
ModelFile = ModelPath + 'coco2017-{epoch:03d}-{val_cls_out_categorical_accuracy:.5f}-{val_loss:.5f}.hdf5'
# 设置多少次不提升，就停止训练
Patience = 300
# 显示方式
Verbose = 1
ResizeWidth = 576
ResizeHeight = 480
CHANNEL = 3
CLASS = 80
ABS_LOCATION = False  # Todo: 每次运行要改的地方(2/3)
CrossRatio = 1.0


# 导入事先存好的样本
def loadSample(samples):
    with open(samples, 'rb') as f:
        all_images = pickle.load(f)
        # print(type(all_images))
        # print(len(all_images))
        class_mapping = pickle.load(f)
        classes_count = pickle.load(f)
        # print(class_mapping)
    return all_images, class_mapping, classes_count


# 训练样本的框标注信息需要标准化，均值为0，方差为1


# def trainStandardScaler(label_r):
#     label = label_r.reshape((label_r.shape[0] * label_r.shape[1] * label_r.shape[2], label_r.shape[3]))
#     scale = StandardScaler().fit(label)
#     label = scale.transform(label)
#     label = label.reshape((label_r.shape[0], label_r.shape[1], label_r.shape[2], label_r.shape[3]))
#     return label, scale


# 验证和测试样本的框标注信息需要标准化，标准化参数要使用训练样本的，以保持一致
def vtStandardScaler(label_r, scale):
    label = label_r.reshape((label_r.shape[0] * label_r.shape[1] * label_r.shape[2], label_r.shape[3]))
    label = scale.transform(label)
    label = label.reshape((label_r.shape[0], label_r.shape[1], label_r.shape[2], label_r.shape[3]))
    return label


def generateGridLabel(y, img_size):
    # Todo: 每次运行要改的地方(3/3)
    scale_file = open('./sample/train_scale_try_5_offset.pkl', 'rb')
    train_scale = pickle.load(scale_file)
    scale_file.close()
    if len(img_size) != len(y):
        print('The length of size_info and y is not same.')
        os._exit(0)

    for m in range(len(y)):
        height_ratio = ResizeHeight / img_size[m][0]
        width_ratio = ResizeWidth / img_size[m][1]
        for n in range(len(y[m])):
            y[m][n]['x1'] = round(float(y[m][n]['x1']) * width_ratio, 2)
            y[m][n]['x2'] = round(float(y[m][n]['x2']) * width_ratio, 2)
            y[m][n]['y1'] = round(float(y[m][n]['y1']) * height_ratio, 2)
            y[m][n]['y2'] = round(float(y[m][n]['y2']) * height_ratio, 2)

            if y[m][n]['x2'] < y[m][n]['x1'] or y[m][n]['y2'] < y[m][n]['y1']:
                print('The box is wrong.')
                os._exit(0)
            if y[m][n]['x1'] < 0 or y[m][n]['y1'] < 0 or y[m][n]['x2'] > ResizeWidth or y[m][n]['y2'] > ResizeHeight:
                print('The box is out of image.')
                os._exit(0)

    label_c, label_bbox = hs.relabel_2D_try_5(y, ResizeHeight, ResizeWidth, DSR, CLASS, ABS_LOCATION, CrossRatio)
    label_c = np_utils.to_categorical(label_c, num_classes=CLASS + 1)
    label_bbox = vtStandardScaler(label_bbox, train_scale)

    return label_c, label_bbox


# def shuffleTogether(a, b, c):
#     d = list(zip(a, b, c))
#     random.shuffle(d)
#     a[:], b[:], c[:] = zip(*d)
#     return a, b, c


def getBatch(X_path, label_c, label_bbox, batch_size):
    '''
    参数：
        X_train：所有图片路径列表
        y_train: 所有图片对应的标签列表
        batch_size:批次
        img_w:图片宽
        img_h:图片高
        color_type:图片类型
    返回:
        一个generator，x: 获取的批次图片 y: 获取的图片对应的标签
    '''
    X_path = np.array(X_path)
    while 1:
        # 打乱数据非常重要
        permutation = np.random.permutation(X_path.shape[0])
        X_path = X_path[permutation]
        label_c = label_c[permutation, :, :, :]
        label_bbox = label_bbox[permutation, :, :, :]
        # X, label_c, label_bbox = shuffleTogether(X_path, label_c, label_bbox)
        print(X_path[0])  # 用于确认是否打乱
        for i in range(0, X_path.shape[0], batch_size):
            x = getIMG.get_im_cv2(X_path[i:i + batch_size], ResizeWidth, ResizeHeight, CHANNEL, normalize=True)
            cls = label_c[i:i + batch_size, :, :, :]
            bbox = label_bbox[i:i + batch_size, :, :, :]
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。
            # 就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield ({'input': x}, {'cls_out': cls, 'reg_out': bbox})


# 从CNN到RNN需要张量的轴进行变换
# def tfTranspose(inputs):
#     x = tensorflow.transpose(inputs, [0, 2, 1])
#     return x

# 获取训练所用的模型，从整个输入到输出，会调用network下的模型


def getModel():
    inputs = Input(shape=(ResizeHeight, ResizeWidth, CHANNEL), name='input')
    x = cnn.network_cnn(inputs)
    print(x.shape)
    # x = core.Reshape((x_train.shape[1] // DSR, x_train.shape[2] // DSR))(x)
    # print(x.shape)
    # x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)), name='transpose_1')(x)
    # print(x.shape)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(x)
    print(x.shape)
    x = rnn.network_lstm(x, name='vertical_lstm')
    print(x.shape)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    print(x.shape)
    x = rnn.network_lstm(x, name='horizontal_lstm')
    print(x.shape)
    cls = out.network_classification(x)
    print(cls.shape)
    bbox = out.network_regression(x)
    print(bbox.shape)
    model = Model(inputs=inputs, outputs=[cls, bbox])
    return model





# 训练模型
def train(model, x_train, label_c_train, label_bbox_train, x_val, label_c_val, label_bbox_val):
    # 设置模型的参数
    model.compile(loss={'cls_out': 'categorical_crossentropy', 'reg_out': 'mean_squared_error'},
                  loss_weights={'cls_out': 1.0, 'reg_out': 1.0}, optimizer='adam',
                  metrics={'cls_out': [metrics.categorical_accuracy], 'reg_out': [metrics.msle]})
    # set callbacks
    # 设置模型按什么标准进行保存。比如：acc,loss
    CP = ModelCheckpoint(ModelFile, monitor='val_cls_out_categorical_accuracy',
                         verbose=1, save_best_only=False, mode='auto')
    # 设置如果性能不上升，停止学习
    ES = EarlyStopping(monitor='val_cls_out_categorical_accuracy', patience=Patience)
    callbacks_list = [CP, ES]

    # 训练开始
    # model.fit(x_train, {'cls_out': cls_train, 'reg_out': bbox_train}, shuffle=True,
    #           batch_size=BatchSize, epochs=Epochs,
    #           verbose=Verbose, callbacks=callbacks_list,
    #           validation_data=(x_val, {'cls_out': cls_val, 'reg_out': bbox_val}))
    num_train_sample = len(x_train)
    num_val_sample = len(x_val)

    model.fit_generator(generator=getBatch(x_train, label_c_train, label_bbox_train, TrainBatchSize),
                        steps_per_epoch=num_train_sample / TrainBatchSize, shuffle=True,
                        epochs=Epochs, verbose=Verbose,
                        validation_data=getBatch(x_val, label_c_val, label_bbox_val, ValBatchSize),
                        validation_steps=num_val_sample / ValBatchSize,
                        callbacks=callbacks_list)


def getInformation(data):
    X_path = []
    y = []
    size = []
    for i in range(len(data)):
        X_path.append(data[i]['filepath'])
        y.append(data[i]['bboxes'])
        size.append([data[i]['height'], data[i]['width'], data[i]['channel']])
    return X_path, y, size


def main():
    # 准备样本
    if FromPickle:
        print('All samples are from pickle files...')
        train_pickle = './sample/coco2017_train_sample.pkl'
        train_images, class_mapping, train_classes_count = loadSample(train_pickle)
        print(len(train_images), 'training samples are loaded.')
        # print(class_mapping)
        # input('Stop!')
        if ImageAugmented:
            for i in range(AugSetNumber):
                aug_pickle = './sample/coco2017_train_aug_set' + str(i).zfill(2) + '.pkl'
                aug_images, class_mapping, aug_classes_count = loadSample(aug_pickle)
                print(len(aug_images), 'augmented training samples are loaded.')
                train_images.extend(aug_images)
        val_pickle = './sample/coco2017_val_sample.pkl'
        val_images, class_mapping, val_classes_count = loadSample(val_pickle)
    else:
        print('Load training samples...')
        TRAIN_ANNOTATIONS_FILE = './sample/coco2017_annotations_train_set.txt'
        # 返回样本的所有信息，各类别数目，以及类别映射
        train_images, class_mapping, train_classes_count = getDATA.get_data(TRAIN_ANNOTATIONS_FILE)
        print(len(train_images), 'training samples are loaded.')
        if ImageAugmented:
            for i in range(AugSetNumber):
                print('Load augmented training samples...')
                AUG_ANNOTATIONS_FILE = './sample/coco2017_annotations_train_aug_set' + str(i).zfill(2) + '.txt'
                aug_images, class_mapping, aug_classes_count = getDATA.get_data(AUG_ANNOTATIONS_FILE)
                print(len(aug_images), 'augmented training samples are loaded.')
                train_images.extend(aug_images)
        print('Load validation samples...')
        VAL_ANNOTATIONS_FILE = './sample/coco2017_annotations_val_set.txt'
        val_images, class_mapping, val_classes_count = getDATA.get_data(VAL_ANNOTATIONS_FILE)

    print(len(val_images), 'validation samples are loaded.')
    random.shuffle(val_images)
    random.shuffle(train_images)  # 洗牌
    print('The total training samples are {}.'.format(len(train_images)))
    X_train_path, Y_train, train_size = getInformation(train_images)
    X_val_path, Y_val, val_size = getInformation(val_images)

    # 在指定路径下检查模型是否存在，如果存在，导入。如果不存在，创建。
    FLAG, MN = hm.checkModel(ModelPath)
    if FLAG:
        print('Load model named by {}...'.format(MN))
        model = load_model(MN)
    else:
        print('Build a new model...')
        model = getModel()
    # 打印模型
    print(model.summary())
    # 绘制模型图
    # plot_model(model, to_file='./model/model.png')

    print('Generating grid samples...')
    train_label_c, train_label_bbox = generateGridLabel(Y_train, train_size)
    val_label_c, val_label_bbox = generateGridLabel(Y_val, val_size)
    # input('Stop!')
    # 训练模型
    print('Training start...')
    train(model, X_train_path, train_label_c, train_label_bbox,
          X_val_path, val_label_c, val_label_bbox)
    # 保留最好的几个模型，删除其它的。
    hm.clearModel(ModelPath)
    print('Congratulation! It finished.')


if __name__ == '__main__':
    main()
