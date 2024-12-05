
from resnet50 import ResNet50
import numpy as np
from keras.utils import np_utils
import cv2
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

def find_line_index(file_path, target_words):
    indexs = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for word in target_words:
        found = False
        for index, line in enumerate(lines):
            if line.strip().startswith(word):
                indexs.append(index)
                found = True
                break
        if not found:
            indexs.append(-1)
    return indexs

def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X = []
        Y = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(lines[i].split(';')[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            X.append(img)
            # tag = lines[i].split(';')[1][:-1]
            Y.append(lines[i].split(';')[1][:-1])
            # print(np.array(Y))

            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像
        X = resize_image(X, (224, 224))
        X = X.reshape(-1, 224, 224, 3)
        Y=np.array(find_line_index(r".\synset.txt",Y))
        Y = np_utils.to_categorical(Y, num_classes=1000)
        yield (X, Y)



# 打开数据集的txt
with open(r".\imagenet_dataset_val.txt", "r") as f:
    lines = f.readlines()

model = ResNet50()
model.summary()

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')

val_loss, val_accuracy = model.evaluate_generator(generate_arrays_from_file(lines[:100], 128), steps=1)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')


