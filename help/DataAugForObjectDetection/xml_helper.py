# -*- coding=utf-8 -*-
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
import xml.dom.minidom as DOC
import os


# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    '''
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        difficult = obj.find('difficult').text
        BndBox = obj.find('bndbox')
        # x_min = int(box[0].text)
        # y_min = int(box[1].text)
        # x_max = int(box[2].text)
        # y_max = int(box[3].text)
        x_min = int(round(float(BndBox.find('xmin').text), 0))  # -1是因为程序是按0作为起始位置的
        y_min = int(round(float(BndBox.find('ymin').text), 0))
        x_max = int(round(float(BndBox.find('xmax').text), 0))
        y_max = int(round(float(BndBox.find('ymax').text), 0))
        coords.append([x_min, y_min, x_max, y_max, name, difficult])
    return coords


# 将bounding box信息写入xml文件中, bouding box格式为[[x_min, y_min, x_max, y_max, name]]
def generate_xml(img_name, coords, img_size, out_root_path):
    '''
    输入：
        img_name：图片名称，如a.jpg
        coords:坐标list，格式为[[x_min, y_min, x_max, y_max, name]]，name为概况的标注
        img_size：图像的大小,格式为[h,w,c]
        out_root_path: xml文件输出的根路径
    '''
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2012_AUG'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img_size[1])

    node_height = SubElement(node_size, 'height')
    node_height.text = str(img_size[0])

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(img_size[2])

    for coord in coords:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = coord[4]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = coord[5]
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '1'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(float(coord[0])))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(float(coord[1])))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(float(coord[2])))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(float(coord[3])))

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    out_xml_file = os.path.join(out_root_path, img_name[:-4] + '.xml')
    with open(out_xml_file, 'wb') as f:
        f.write(xml)
