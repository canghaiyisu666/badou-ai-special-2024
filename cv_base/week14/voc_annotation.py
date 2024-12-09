import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

wd = getcwd()
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(year, image_id, list_file):
    """
    将图像的标注信息转换为所需格式并写入到 list_file 中。

    该函数从 Pascal VOC 数据集中的 XML 文件读取图像的标注信息，提取图像中每个对象的信息，
    并以指定格式将这些信息写入到 list_file 中。写入的信息包括图像路径、对象的边界框坐标以及对象类别。

    参数:
    - year (str): 数据集年份，例如 '2007' 或 '2012'。
    - image_id (str): 图像的唯一标识符。
    - list_file (file object): 用于写入转换后标注信息的文件对象。

    返回:
    无
    """
    # 打开 XML 标注文件
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))

    # 解析 XML 文件
    tree = ET.parse(in_file)
    root = tree.getroot()

    # 检查是否存在对象标签
    if root.find('object') == None:
        return

    # 写入图像路径
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))

    # 对有对象的图片记录标签转换为classes的索引后也写入文件中
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        # 跳过不在类别列表中的对象或困难对象
        if cls not in classes or int(difficult) == 1:
            continue

        # 获取类别的索引
        cls_id = classes.index(cls)

        # 获取边界框信息
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))

        # 写入边界框坐标和类别索引
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    # 写入换行符
    list_file.write('\n')



for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()
