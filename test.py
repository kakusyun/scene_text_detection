import pickle
import numpy as np
from help import handle_model as hm
from keras.models import load_model
import time
from help import NMS, AccuracyMAP, Accuracy
from help import visualize_bbox as VBB
import cv2, math
import os, tarfile
import shutil
import random
# from progressbar import *
from tqdm import tqdm
import copy

# Is the time of reading image tested?
TestTime_ReadImg = False
TRY = '01'  #Todo: 每次运行要改的地方(1/5)
DSR = 16  # down sampling ratio下采样，下采样2的时候，BatchSize=256
CNNModel = 'vgg16_' + str(DSR) + '_' + TRY
ModelPath = './model/' + CNNModel + '/'
# ModelPath = './tmp/'
ResultPath = './result/' + CNNModel
Result = ResultPath + '/predicted_'
ResultImage = ResultPath + '/predicted_images_'
mAP_first = True  #Todo: 每次运行要改的地方(2/5)
CLASS = 80
NMS_SAME_CLASS = True
ABS_LOCATION = False #Todo: 每次运行要改的地方(3/5)
GridImage = True #Todo: 每次运行要改的地方(4/5)

# input the saved samples
def loadSample():
    print('Load samples...')
    input_sample = open('./sample/coco2017_valtest_sample.pkl', 'rb')
    all_images = pickle.load(input_sample)
    # classes_count = pickle.load(input_sample)
    class_mapping = pickle.load(input_sample)
    input_sample.close()
    # print(all_images[0])
    return all_images, class_mapping


# (x,y,w,h)->(x1,y1,x2,y2,score,cls)
# handle a single bounding box
def handleSingleBBOX(bbox, score):
    bbox_score_cls = []
    for i in range(bbox.shape[0]):  # 8
        for j in range(bbox.shape[1]):  # 40
            tmp_cls = score[i, j, :].tolist()
            tmp_cls = tmp_cls.index(max(tmp_cls))
            if tmp_cls != CLASS:
                if ABS_LOCATION:
                    x1 = bbox[i][j][0]
                    y1 = bbox[i][j][1]
                    x2 = bbox[i][j][0] + bbox[i][j][2]
                    y2 = bbox[i][j][1] + bbox[i][j][3]
                else:
                    x1 = bbox[i][j][0] + j * DSR
                    y1 = bbox[i][j][1] + i * DSR
                    x2 = bbox[i][j][0] + j * DSR + bbox[i][j][2]
                    y2 = bbox[i][j][1] + i * DSR + bbox[i][j][3]
                if x1 <= x2 and y1 <= y2:
                    bbox_score_cls.append([x1, y1, x2, y2,
                                           np.max(score[i][j]),
                                           tmp_cls])
    # print(bbox_score_cls)
    return bbox_score_cls


# (x,y,w,h)->(x1,y1,x2,y2,score,cls)
# handle the whole bounding boxes
def handleBBOX(bbox, score):
    bbox_score_cls = []
    for num in range(bbox.shape[0]):  # number
        tmp_list = []
        for i in range(bbox.shape[1]):  # 24
            for j in range(bbox.shape[2]):  # 30
                tmp_cls = score[num, i, j, :].tolist()
                tmp_cls = tmp_cls.index(max(tmp_cls))
                if tmp_cls != CLASS:
                    if ABS_LOCATION:
                        x1 = bbox[num][i][j][0]
                        y1 = bbox[num][i][j][1]
                        x2 = bbox[num][i][j][0] + bbox[num][i][j][2]
                        y2 = bbox[num][i][j][1] + bbox[num][i][j][3]
                    else:
                        x1 = bbox[num][i][j][0] + j * DSR
                        y1 = bbox[num][i][j][1] + i * DSR
                        x2 = bbox[num][i][j][0] + j * DSR + bbox[num][i][j][2]
                        y2 = bbox[num][i][j][1] + i * DSR + bbox[num][i][j][3]
                    if x1 <= x2 and y1 <= y2:
                        # print('Congratulation! A box is found.')
                        tmp_list.append([x1, y1, x2, y2,
                                         np.max(score[num][i][j]),
                                         tmp_cls])
        bbox_score_cls.append(tmp_list)
    # print(bbox_score_cls[0])
    return bbox_score_cls


def testImageReadingTime(test_img):
    start_time = time.time()
    for i in range(len(test_img)):
        img = cv2.imread(test_img[i]['filepath'][1:])
    print('The average time of reading an image is {} seconds.'
          .format((time.time() - start_time) / len(test_img)))


# testing with image reading
def testTime(test_img, test_num, md, ts):
    for i in range(test_num):
        img = cv2.imread(test_img[i]['filepath'][1:])
        # print(img.shape)
        # test_x = img.transpose((1, 0))
        test_x = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        # print(img.shape)
        predict_cls, predict_bbox = md.predict(test_x)
        # print(type(predict_bbox[0]))
        predict_bbox = ts.inverse_transform(predict_bbox)
        bbox = handleSingleBBOX(predict_bbox[0], predict_cls[0])
        NMS.non_max_suppression_fast(bbox, overlap_thresh=0.5, max_boxes=20)


# fast test, draw predicted bounding box, and save results to files
def testFast(test_img, md, ts, cm, name_model):
    (height, width, channel) = test_img[0]['pixel'].shape
    test_x = np.zeros((len(test_img), height, width, channel))

    for i in range(len(test_img)):
        test_x[i] = test_img[i]['pixel'] / 127.5 - 1.
    #     for t1 in range(test_x[i].shape[0]):
    #         for t2 in range(test_x[i].shape[1]):
    #             print(test_x[i][t1][t2])

    # axises should be exchanged
    # test_x = test_x.transpose((0, 2, 1, 3))  # height <-> width
    # md.evaluate(test_x,)

    start = time.time()
    # predicting...
    predict_cls, predict_bbox = md.predict(test_x)
    # inverse data standardizing
    predict_bbox = ts.inverse_transform(predict_bbox)  # 有点疑问，是否要reshape
    # handle bounding box
    bbox = handleBBOX(predict_bbox, predict_cls)

    current_model = ModelPath + name_model + '.hdf5'

    # NMS
    if NMS_SAME_CLASS:
        for i in range(len(bbox)):
            if len(bbox[i]) != 0:
                bbox_nms = []
                NO_BLANK = True
                while NO_BLANK:
                    tmp_bbox_1 = []
                    tmp_bbox_1.append(bbox[i][0])
                    tmp_bbox_2 = copy.deepcopy(bbox[i])
                    bbox[i].remove(bbox[i][0])
                    for j in range(1, len(tmp_bbox_2)):
                        if tmp_bbox_2[j][5] == tmp_bbox_2[0][5]:
                            tmp_bbox_1.append(tmp_bbox_2[j])
                            bbox[i].remove(tmp_bbox_2[j])
                    tmp_bbox_1 = NMS.non_max_suppression_fast(tmp_bbox_1, overlap_thresh=0.5, max_boxes=20,
                                                              current_model=current_model)
                    for k in range(len(tmp_bbox_1)):
                        bbox_nms.append(tmp_bbox_1[k])
                    if len(bbox[i]) == 0:
                        NO_BLANK = False
                        bbox[i] = copy.deepcopy(bbox_nms)
    else:
        for i in range(len(bbox)):
            bbox[i] = NMS.non_max_suppression_fast(bbox[i], overlap_thresh=0.5, max_boxes=20,
                                                   current_model=current_model)

    cost_time = time.time() - start
    print('The cost time for an image is', str(cost_time / test_x.shape[0]) + '.', 'Not include reading images.')

    print('Writing the results to files...')

    ResultPath_md = Result + name_model + '/'
    ResultImage_md = ResultImage + name_model + '/'
    ResultImage2_md = ResultImage + name_model + '_grid/'

    if not os.path.exists(ResultPath_md):
        os.makedirs(ResultPath_md)
    if not os.path.exists(ResultImage_md):
        os.mkdir(ResultImage_md)
    if not os.path.exists(ResultImage2_md):
        os.mkdir(ResultImage2_md)
        # progress = ProgressBar()
    for i in tqdm(range(len(test_img))):
        img_path = test_img[i]['filepath']

        # the test files come from different folders,
        # names of files may be same,
        # so put pre-names in the front of them.
        # tmp_list = img_path.strip().split('/')
        # if 'R_HDS5_test_set' in tmp_list:
        #     mark = 'R_HDS5_'
        # elif 'S_HDS5_test_set' in tmp_list:
        #     mark = 'S_HDS5_'
        # elif 'S_MNIST_test_set' in tmp_list:
        #     mark = 'S_MNIST_'
        # else:
        #     print('Can not find the test sample.')
        #     exit(0)

        base_name = str(os.path.basename(img_path).split('.')[0])
        # result_txt = ResultPath_md + mark + base_name + '.txt'
        result_txt = ResultPath_md + base_name + '.txt'

        bboxes = {}
        # write the predicted results to files
        with open(result_txt, 'w') as f:
            for b in range(len(bbox[i])):
                class_name = int(bbox[i][b][5])
                class_prob = round(bbox[i][b][4], 2)
                x1 = round(bbox[i][b][0], 2)
                y1 = round(bbox[i][b][1], 2)
                x2 = round(bbox[i][b][2], 2)
                y2 = round(bbox[i][b][3], 2)
                f.write('{} {} {} {} {} {} \n'.format(class_name, class_prob, x1, y1, x2, y2))

                if class_name not in bboxes:
                    bboxes[class_name] = []
                bboxes[class_name].append([x1, y1, x2, y2, class_prob, class_name])

        # draw predicted bounding box and save
        height_ratio = test_img[i]['height_ratio']
        width_ratio = test_img[i]['width_ratio']
        img = cv2.imread(img_path[1:])
        for cls in bboxes:
            for bn in range(len(bboxes[cls])):
                bboxes[cls][bn][0] = math.floor(bboxes[cls][bn][0]/ width_ratio)
                bboxes[cls][bn][2] = math.floor(bboxes[cls][bn][2] / width_ratio)
                bboxes[cls][bn][1] = math.floor(bboxes[cls][bn][1] / height_ratio)
                bboxes[cls][bn][3] = math.floor(bboxes[cls][bn][3] / height_ratio)

        img = VBB.draw_boxes_and_label_on_image_cv2(img, cm, bboxes)
        # result_image_path = ResultImage_md + mark + base_name + '.png'
        result_image_path = ResultImage_md + base_name + '.png'
        cv2.imwrite(result_image_path, img)


        # print(predict_cls[i].shape)
        # for t1 in range(predict_cls[i].shape[0]):
        #     for t2 in range(predict_cls[i].shape[1]):
        #         print(predict_cls[i][t1][t2])

        # 画小切块的类别
        if GridImage:
            img2 = VBB.draw_boxes_and_label_cv2(test_img[i]['pixel'], cm, predict_cls[i], CLASS)
            result_image2_path = ResultImage2_md + base_name + '.png'
            cv2.imwrite(result_image2_path, img2)

        # input("STOP")
    # test accuracy
    # print('Calculating accuracy and mAP...')
    # os.chdir('./result/' + CNNModel + '/')
    # if os.path.exists('./predicted'):
    #     shutil.rmtree('./predicted')
    # os.mkdir('./predicted')
    # source_dir = './predicted_' + name_model
    # os.rename(source_dir, './predicted')
    # os.chdir('../../')
    if mAP_first:
        print('Calculating accuracy and mAP...')
        eval = AccuracyMAP.startCalACC(ResultPath, name_model)
        good = 0.2
        normal = 0.1
    else:
        print('Calculating accuracy...')
        eval = Accuracy.startCalACC(ResultPath, name_model)
        good = 0.2
        normal = 0.1

    good_model = ModelPath + 'good_model/'
    normal_model = ModelPath + 'normal_model'
    src_model = ModelPath + name_model + '.hdf5'
    source_dir = ResultPath + '/predicted_' + name_model
    if eval >= good:
        if not os.path.exists(good_model):
            os.mkdir(good_model)
        shutil.move(src_model, good_model)
        # pack the results into tar file
        # os.chdir('./result/' + CNNModel + '/')
        if os.path.exists(ResultPath + '/predicted.tar'):
            os.remove(ResultPath + '/predicted.tar')
        # os.rename('./predicted', source_dir)
        os.chdir(source_dir)
        tar_file = './predicted.tar'
        with tarfile.open(tar_file, "w:") as tar:
            for file in tqdm(os.listdir('./')):
                tar.add(file)
        shutil.move('./predicted.tar', '../')
    elif eval >= normal:
        if not os.path.exists(normal_model):
            os.mkdir(normal_model)
        shutil.move(src_model, normal_model)
    else:
        os.remove(src_model)


def main():
    # prepare samples
    test_images, class_label_mapping, = loadSample()
    # print(class_label_mapping)
    # input('Stop!')
    # if you want to test a small set of samples, use it
    # TestNumber = 1000
    # test_images = random.sample(test_images, TestNumber)
    TestNumber = len(test_images)
    print(len(test_images), 'test samples are loaded.')

    # testImageReadingTime(test_images)
    # input('STOP!')

    # input the saved standardizing parameters of the training samples
    # Todo: 每次运行要改的地方(5/5)
    scale_file = open('./sample/train_scale_try_5_offset.pkl', 'rb')
    train_scale = pickle.load(scale_file)
    scale_file.close()

    # check if the models exist
    # if yes, input
    # if no, end
    FLAG, MN = hm.checkModel(ModelPath)
    if FLAG:
        print('Load model...')
        model = load_model(MN)
        model_name = os.path.basename(MN).split('/')[-1]
        print('The model named by', model_name, 'is loaded.')
        model_name = model_name[:-5]

        # execute it if you want to know the time of reading images
        if TestTime_ReadImg:
            print('Testing for time starts...')
            start = time.time()
            testTime(test_images, TestNumber, model, train_scale)
            cost_time = time.time() - start
            print('The cost time for an image is', str(cost_time / TestNumber) + '.')

        # fast test, not reading images
        # read the data from saved pickle file
        print('Testing for saving files starts...')
        testFast(test_images, model, train_scale, class_label_mapping, model_name)

        # input("Stop!")

        print('Congratulation! It finished.')
    else:
        print('There is no trained model. Please check.')


if __name__ == '__main__':
    main()
