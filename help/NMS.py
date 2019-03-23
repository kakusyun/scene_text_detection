import numpy as np
import os
import time


def non_max_suppression_fast(boxes, overlap_thresh=0.9, max_boxes=300, current_model=None):
    """
    从所给定的所有框中选择指定个数最合理的边框
    :param boxes: 框
    :param overlap_thresh:
    :param max_boxes:
    :return: 框（x1,y1,x2,y2）的形式
    """
    # I changed this method with boxes already contains probabilities, so don't need prob send in this method
    # TODO: Caution!!! now the boxes actually is [x1, y1, x2, y2, prob, Cls] format!!!! with prob built in
    if len(boxes) == 0:
        # 没有框
        return []
    # normalize to np.array
    boxes = np.array(boxes)
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]



    # 对输入数据进行确认
    try:
        np.testing.assert_array_less(x1, x2)
        np.testing.assert_array_less(y1, y2)
    except:
        os.remove(current_model)
        print('A bad model is removed.')
        count = 0
        b = 10
        while (count < b):
            print(b-count)
            time.sleep(1)
            count += 1
        os._exit(0)

    # 转换数据类型
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexed
    pick = []
    # calculate the areas
    area = (x2 - x1) * (y2 - y1)
    # sorted by boxes last element which is prob
    # indexes是数值所在的位置。
    # list = [1,3,2], after argsort
    # idxs = [0,2,3]
    indexes = np.argsort([i[-2] for i in boxes])

    # 按照概率从大到小取出框，且框的重合度不可以高于阈值：
    # 思路：
    # 1、每一次取概率最大的框（即indexes最后一个）
    # 2、删除掉剩下的框中重合度高于阈值的框
    # 3、直到取满max_boxes为止
    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        # 取出idexes队列中最大概率框的序号，将其添加到pick中
        pick.append(i)

        # find the intersection
        xx1_int = np.maximum(x1[i], x1[indexes[:last]])
        yy1_int = np.maximum(y1[i], y1[indexes[:last]])
        xx2_int = np.minimum(x2[i], x2[indexes[:last]])
        yy2_int = np.minimum(y2[i], y2[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        # find the union
        area_union = area[i] + area[indexes[:last]] - area_int

        # compute the ratio of overlap
        # 交并比
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        # 删除重叠率较高的位置
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break
    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick]
    # 返回相应的框，pick中存的是位置
    return boxes