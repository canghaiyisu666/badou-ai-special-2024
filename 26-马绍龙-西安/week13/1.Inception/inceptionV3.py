# -------------------------------------------------------------#
#   InceptionV3的网络部分
# -------------------------------------------------------------#
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              strides=(1, 1),
              padding='same',
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    # 64+64+96+32 = 256  nhwc-0123
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3, name='mixed0')

    # Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    # 64+64+96+64 = 288 
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3, name='mixed1')

    # Block1 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    # 64+64+96+64 = 288 
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3, name='mixed2')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3, name='mixed4')

    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3, name='mixed' + str(5 + i))

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3,
            name='mixed' + str(9 + i))

    # 平均池化后全连接。
    x = GlobalAveragePooling2D(name='avg_pool')(x)  # 可以代替flatten拍扁，但GlobalAveragePooling还进一步进行了压缩，减少了计算量
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input

    model = Model(inputs, x, name='inception_v3')

    return model


def preprocess_input(x):  # 把输入数据标准化为[-1, 1]的范围
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = InceptionV3()

    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    for i in {"airp", "bike", "boat", "car", "cat", "dog", "horse", "sheep", "table", "train"}:  # 对十张照片进行准预测，并且计算准确率
        path = "./test_data/" + i + ".jpg"
        img = image.load_img(path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 打印预测结果
        print("result: ")
        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds))
        print("===========================" + i)
        print('--------------------------------------------------------------------------')




"""
result: 
Predicted: [[('n02412080', 'ram', 0.9691943), ('n02415577', 'bighorn', 0.004932375), ('n02105412', 'kelpie', 0.0009911921), ('n02797295', 'barrow', 0.0004641722), ('n02328150', 'Angora', 0.0003913182)]]
===========================sheep
--------------------------------------------------------------------------
result: 
Predicted: [[('n02123045', 'tabby', 0.35685024), ('n02123159', 'tiger_cat', 0.20545918), ('n02124075', 'Egyptian_cat', 0.14499731), ('n02127052', 'lynx', 0.049526762), ('n04040759', 'radiator', 0.020444896)]]
===========================cat
--------------------------------------------------------------------------
result: 
Predicted: [[('n03376595', 'folding_chair', 0.5106684), ('n03201208', 'dining_table', 0.45491225), ('n04099969', 'rocking_chair', 0.0012571805), ('n03761084', 'microwave', 0.0005414652), ('n03018349', 'china_cabinet', 0.00042493845)]]
===========================table
--------------------------------------------------------------------------
result: 
Predicted: [[('n04285008', 'sports_car', 0.5983392), ('n03100240', 'convertible', 0.14720407), ('n02974003', 'car_wheel', 0.032744437), ('n02704792', 'amphibian', 0.017280106), ('n03459775', 'grille', 0.0031893714)]]
===========================car
--------------------------------------------------------------------------
result: 
Predicted: [[('n04310018', 'steam_locomotive', 0.34040344), ('n03272562', 'electric_locomotive', 0.32673994), ('n03393912', 'freight_car', 0.22898766), ('n03895866', 'passenger_car', 0.022015784), ('n04532670', 'viaduct', 0.0009797245)]]
===========================train
--------------------------------------------------------------------------
result: 
Predicted: [[('n04147183', 'schooner', 0.780304), ('n03947888', 'pirate', 0.05778463), ('n04612504', 'yawl', 0.028385997), ('n04606251', 'wreck', 0.001354308), ('n03216828', 'dock', 0.0010730714)]]
===========================boat
--------------------------------------------------------------------------
result: 
Predicted: [[('n03792782', 'mountain_bike', 0.4925117), ('n02835271', 'bicycle-built-for-two', 0.10839923), ('n04482393', 'tricycle', 0.031536303), ('n03208938', 'disk_brake', 0.012938443), ('n03649909', 'lawn_mower', 0.0039693187)]]
===========================bike
--------------------------------------------------------------------------
result: 
Predicted: [[('n04552348', 'warplane', 0.8546734), ('n04592741', 'wing', 0.05919604), ('n04008634', 'projectile', 0.010891675), ('n03773504', 'missile', 0.0073517933), ('n02690373', 'airliner', 0.0041570365)]]
===========================airp
--------------------------------------------------------------------------
result: 
Predicted: [[('n02099601', 'golden_retriever', 0.86163354), ('n02099712', 'Labrador_retriever', 0.026784329), ('n02115641', 'dingo', 0.0037285928), ('n02091831', 'Saluki', 0.0033842374), ('n02101388', 'Brittany_spaniel', 0.0022100655)]]
===========================dog
--------------------------------------------------------------------------
result: 
Predicted: [[('n02389026', 'sorrel', 0.9545797), ('n04604644', 'worm_fence', 0.0016825886), ('n03124170', 'cowboy_hat', 0.0009222231), ('n03538406', 'horse_cart', 0.0009009582), ('n02795169', 'barrel', 0.0008929525)]]
===========================horse
--------------------------------------------------------------------------
"""
