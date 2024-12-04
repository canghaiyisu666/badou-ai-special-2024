import os
import random

# 定义XML文件路径和保存基础路径
xmlfilepath = r'./VOCdevkit/VOC2007/Annotations'
saveBasePath = r"./VOCdevkit/VOC2007/ImageSets/Main/"

# 定义训练验证集和训练集的比例
trainval_percent = 1
train_percent = 1

# 获取XML文件夹中的所有文件，并筛选出.xml文件
temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

# 计算总文件数，并生成索引列表
num = len(total_xml)
list = range(num)
# 根据比例计算训练验证集和训练集的大小
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
# 随机选择训练验证集和训练集的索引
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

# 打印训练验证集和训练集的大小
print("train and val size", tv)
print("train size", tr)
# 打开文件以写入相应的索引
ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

# 遍历索引列表，根据索引写入相应的文件名到对应的文件中
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

# 关闭所有打开的文件
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
