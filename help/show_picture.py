import os
import cv2


def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(0, img.shape[0], 16):
        cv2.line(img, (0, i), (img.shape[1] - 1, i), 127, 1)
    cv2.line(img, (0, img.shape[0] - 1), (img.shape[1] - 1, img.shape[0] - 1), 27, 1)

    for i in range(0, img.shape[1], 16):
        cv2.line(img, (i, 0), (i, img.shape[0] - 1), 127, 1)
    cv2.line(img, (img.shape[1] - 1, 0), (img.shape[1] - 1, img.shape[0] - 1), 27, 1)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    cv2.namedWindow('pic', 1)  # 1表示原图
    # cv2.moveWindow('pic', 0, 0)
    # cv2.resizeWindow('pic')  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove('./1.jpg')
