from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np


class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(1e-3, 0.5)

        self.discriminator = self.build_discriminator()  # 创建 ， 编译 discriminator判别器
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()     # Build the generator生成器

        z = Input(shape=(self.latent_dim,))         # 生成器使用噪声来生成图片
        img = self.generator(z)

        self.discriminator.trainable = False        # combined模型只更新生成器的参数
        validity = self.discriminator(img)          # 判别器接收生成的图像作为输入，并判断其有效性

        self.combined = Model(z, validity)    # 编译组合模型  (生成器+判别器)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        # 共有两个compile的模型，combined后的模型只更新生成器的参数

    def build_generator(self):

        model = Sequential(name="GENERATOR")

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))    # np.prod计算数组中所有元素乘积
        model.add(Reshape(self.img_shape))                              # img_shape == 28*28*1

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential(name="DISCRIMINATOR")

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)  # img_shape==28*28*1
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=200):

        # 正样本选择mnist数据集，共6000条
        (X_train, _), (_, _) = mnist.load_data()

        # 标准化到 [-1 , 1]
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # 构建ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator训练判别器
            # ---------------------
            idx = np.random.randint(0, X_train.shape[0], batch_size)  # 通过随机输入得到一张正样本图片
            # 生成一个包含batch_size个随机整数的数组，这些整数的范围在 0 到X_train.shape[0]-1 之间，用于从训练数据集中选择一个批次的数据。

            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))  # 基于噪音生成图片

            gen_imgs = self.generator.predict(noise)            # 生成器生成一批图片,gen_imgs.shape==(batchsize,h,w,c)

            # 训练判别器，同时记录loss
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)     # 总loss取平均值

            # ---------------------
            #  Train Generator训练生成器
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))  # 这里的noise要保证和上面的不同

            # 训练生成器，并记录loss (生成器生成的图片都标记为正样本***，这样训练过程中，生成器参数会趋于正样本)
            g_loss = self.combined.train_on_batch(noise, valid)

            # 打印训练进度
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if (epoch + 1) % sample_interval == 0:      # 当epoch计数达到预定的采样间隔时，生成一组图像进行保存
                self.sample_images(epoch + 1)

    def sample_images(self, epoch):
        """          生成并保存指定epoch的图像样本。
        参数:     - epoch: 训练的当前epoch数，用于保存图像的文件名。
        此函数不会返回任何值，但会生成一组图像并保存到本地目录。
        """

        r, c = 5, 5                                     # 定义行数和列数，用于展示图像网格
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))  # 生成随机噪声作为生成器的输入
                                                                                #  random.normal为高斯噪声，均值0，标准差1
                                                                                #  r * c：表示生成样本的数量
                                                                                #  latent_dim表示生成器的输入向量维度
        gen_imgs = self.generator.predict(noise)        # 使用生成器预测产生图像
        gen_imgs = 0.5 * gen_imgs + 0.5                 # 将生成的图像像素值重新缩放至0-1范围，因为生成器通过tanh激活输出为[-1,1]
        fig, axs = plt.subplots(r, c)                   # 创建一个子图网格用于展示生成的图像
        cnt = 0                                         # 初始化计数器以遍历所有生成的图像
        for i in range(r):
            for j in range(c):                          # 显示图像并关闭坐标轴
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./images/mnist_%d.png" % epoch)    # 保存图像到本地

        plt.close()                                     # 关闭图像以释放内存 不close会内存泄露


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2000, batch_size=32, sample_interval=200)
