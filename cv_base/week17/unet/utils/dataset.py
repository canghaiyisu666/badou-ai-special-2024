import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    def __init__(self, train_path):
        # 初始化函数，读取所有data_path下的图片
        self.train_path = train_path
        self.imgs_path = glob.glob(os.path.join(train_path, 'images/*.tif'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]

        # 根据image_path生成label_path
        label_path = image_path.replace('images', 'label')
        # print(image_path)
        # print(label_path)

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])  # 加入通道数1
        label = label.reshape(1, label.shape[0], label.shape[1])  # 加入通道数1
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理，四分之一概率为2
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    # 测试数据集
    isbi_dataset = ISBI_Loader("../dataset/train/")  # 和train.py的工作路径不同， 不能传入一样的路径
    print(len(isbi_dataset))

    train_data = torch.utils.data.DataLoader(dataset=isbi_dataset, batch_size=1, shuffle=True)

    for image, label in train_data:
        print(image.shape)
        print(label.shape)
