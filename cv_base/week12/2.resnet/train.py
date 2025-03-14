from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from resnet50 import ResNet50
import numpy as np
import cv2
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.layers import Input
from tensorflow.keras.models import Model

from keras_applications import imagenet_utils
from keras import backend as K


# K.set_image_dim_ordering('tf')
# K.image_data_format() == 'channels_first'


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


def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

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
        # yield使得生成器能够在需要时产生下一个值，而不需要一次性将所有数据加载到内存中。与普通函数使用return语句不同，
        # yield可以暂停函数的执行并保存当前状态，然后在下一次调用时从上次暂停的地方继续执行


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\imagenet_dataset_val.txt", "r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于评估。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 一次的训练集大小
    batch_size = 32

    # 保存的方式，3代保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='accuracy',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    # 学习率下降的方式，acc三次不下降就自动下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='accuracy',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    model = ResNet50()
    # 交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    # Train on 22500 samples, val on 2500 samples, with batch size 128.

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),  # 每个epoch（训练周期）内模型遍历训练数据集的次数
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),  # 地板除7 // 2 =3
                        epochs=10,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir + 'last1.h5')
