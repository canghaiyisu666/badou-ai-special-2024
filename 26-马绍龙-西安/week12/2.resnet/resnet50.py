# -------------------------------------------------------------#
#   ResNet50的网络部分
# -------------------------------------------------------------#
from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])  # 加法尺寸相同。 concat为级联  3*3 concat 3*3 = 3*6  concat要确保其他维度尺寸相等
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)  # 单独加padding，而不是在conv2d中做

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


if __name__ == '__main__':
    model = ResNet50()
    model.summary()  # 自动打印模型结构，包括每一层的名字，参数个数、输入输出形状（relu和池化中存在不可训练参数）
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # 调用接口，做了归一化预处理

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))  # keras自带解码，输出标签结果





'''
result: 
Predicted: [[('n02389026', 'sorrel', 0.9987846), ('n02437616', 'llama', 0.0010307925), ('n02397096', 'warthog', 3.9470662e-05), ('n02403003', 'ox', 2.7034836e-05), ('n04604644', 'worm_fence', 2.5609914e-05)]]
===========================horse
--------------------------------------------------------------------------
result: 
Predicted: [[('n03947888', 'pirate', 0.5491915), ('n04147183', 'schooner', 0.42673454), ('n04612504', 'yawl', 0.023428513), ('n04606251', 'wreck', 0.00041119888), ('n03216828', 'dock', 9.443945e-05)]]
===========================boat
--------------------------------------------------------------------------
result: 
Predicted: [[('n02123045', 'tabby', 0.3186851), ('n02123159', 'tiger_cat', 0.1831804), ('n04589890', 'window_screen', 0.1713991), ('n02124075', 'Egyptian_cat', 0.09318353), ('n02123394', 'Persian_cat', 0.0498708)]]
===========================cat
--------------------------------------------------------------------------
result: 
Predicted: [[('n02412080', 'ram', 0.9686623), ('n01877812', 'wallaby', 0.025353875), ('n02415577', 'bighorn', 0.0021512802), ('n02437616', 'llama', 0.00036255876), ('n02395406', 'hog', 0.00035574022)]]
===========================sheep
--------------------------------------------------------------------------
result: 
Predicted: [[('n04592741', 'wing', 0.7701196), ('n04552348', 'warplane', 0.13590305), ('n02690373', 'airliner', 0.05750114), ('n04266014', 'space_shuttle', 0.011926264), ('n02692877', 'airship', 0.0037955604)]]
===========================airp
--------------------------------------------------------------------------
result: 
Predicted: [[('n03201208', 'dining_table', 0.54302233), ('n03376595', 'folding_chair', 0.43352717), ('n04201297', 'shoji', 0.0028409695), ('n04553703', 'washbasin', 0.0023521287), ('n03018349', 'china_cabinet', 0.0021711374)]]
===========================table
--------------------------------------------------------------------------
result: 
Predicted: [[('n04310018', 'steam_locomotive', 0.85631955), ('n03393912', 'freight_car', 0.06789272), ('n03272562', 'electric_locomotive', 0.044307705), ('n03895866', 'passenger_car', 0.021852477), ('n04335435', 'streetcar', 0.0031961205)]]
===========================train
--------------------------------------------------------------------------
result: 
Predicted: [[('n02099601', 'golden_retriever', 0.548536), ('n02105162', 'malinois', 0.069410354), ('n02115641', 'dingo', 0.055435784), ('n02099712', 'Labrador_retriever', 0.02928076), ('n02088466', 'bloodhound', 0.027839318)]]
===========================dog
--------------------------------------------------------------------------
result: 
Predicted: [[('n04285008', 'sports_car', 0.35216752), ('n03100240', 'convertible', 0.30046308), ('n02974003', 'car_wheel', 0.18292405), ('n04037443', 'racer', 0.06407548), ('n03680355', 'Loafer', 0.014549531)]]
===========================car
--------------------------------------------------------------------------
result: 
Predicted: [[('n03792782', 'mountain_bike', 0.42900684), ('n02835271', 'bicycle-built-for-two', 0.15556844), ('n04485082', 'tripod', 0.027216112), ('n04557648', 'water_bottle', 0.025366185), ('n04482393', 'tricycle', 0.018176634)]]
===========================bike
--------------------------------------------------------------------------
'''
