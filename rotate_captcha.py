# -*- coding: utf-8 -*-
"""
图片旋转验证码识别
训练集图片命名格式: index_angle.jpeg/index_angle_1.jpeg/index.jpeg
"""
import os
import os.path
import time

import cv2

import numpy as np
import keras_preprocessing.image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications
from tensorflow.keras import models
from tensorflow.keras import metrics


def rotate_image(src, dst, angle):
    """
    旋转图片
    :param src: 原始图片路径
    :param dst: 目标存储路径
    :param angle: 旋转角度
    """
    src_image = cv2.imread(src)
    # 获取图像尺寸
    (h, w) = src_image.shape[:2]
    center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(src_image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    cv2.imwrite(dst, rotated)


def load_image_data(image_dir_path, image_height, image_width, image_channel=3):
    """
    加载图片数据
    图片标签从图片文件名中读取 图片文件名应该符合 label_xxxx.jpg(png)格式
    RGB图片将会转换成灰度图片
    :param image_dir_path: 图片路径
    :param image_height: 图片高度
    :param image_width: 图片宽度
    :param image_channel 图片通道
    :return: image_data, data_label
    """
    image_name_list = os.listdir(image_dir_path)
    image_data = np.zeros(shape=(len(image_name_list), image_height, image_width, image_channel))
    label_data = np.zeros(shape=(len(image_name_list), 1))
    color_mode = 'rgb' if image_channel == 3 else 'grayscale'
    for index, image_name in enumerate(image_name_list):
        img = keras_preprocessing.image.utils.load_img(os.path.join(image_dir_path, image_name), color_mode=color_mode)
        x = keras_preprocessing.image.utils.img_to_array(img)
        if hasattr(img, 'close'):
            img.close()
        image_data[index] = x
        image_name_with_suffix = image_name[0:image_name.rfind('.')]
        fields = image_name_with_suffix.split('_')
        if len(fields) == 1 or len(fields) == 2:
            label_data[index] = np.zeros(1)
        else:
            label_data[index] = np.ones(1)
    return image_data, label_data


class DirectoryImageGenerator(keras_preprocessing.image.Iterator):

    def _get_batches_of_transformed_samples(self, index_array):
        image_data = np.zeros(shape=(len(index_array), self.image_height, self.image_width, self.image_channel))
        label_data = np.zeros(shape=(len(index_array), 1))
        color_mode = 'rgb' if self.image_channel == 3 else 'grayscale'
        for index, image_index in enumerate(index_array):
            image_path = self.filter_images[image_index]
            image = keras_preprocessing.image.utils.load_img(image_path, color_mode=color_mode)
            x = keras_preprocessing.image.utils.img_to_array(image)
            if hasattr(image, 'close'):
                image.close()
            image_data[index] = x
            image_name = image_path.split(os.sep)[-1]
            image_name = image_name[0: image_name.rfind('.')]
            name_fields = image_name.split('_')
            if len(name_fields) in (1, 2):
                label_data[index] = np.zeros(1)
            elif len(name_fields) == 3:
                label_data[index] = np.ones(1)
            else:
                raise RuntimeError('image name must in formats (base_angle/base_angle_1/base)')
        return image_data, label_data

    def __init__(self, directory_path, image_height, image_width, image_channel, batch_size, shuffle=False, seed=0, image_suffix=None):
        if not image_suffix:
            image_suffix = ['.png', '.jpg', '.jpeg', '.bmp', '.ppm', '.tif', '.tiff']
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise RuntimeError('directory must exist')
        if image_channel not in (1, 3):
            raise RuntimeError('image channel must be 1 or 3')
        names = os.listdir(directory_path)
        self.filter_images = []
        for name in names:
            name_lower = name.lower()
            for suffix in image_suffix:
                if name_lower.endswith(suffix):
                    self.filter_images.append(os.path.join(directory_path, name))
                    break
        self.size = len(self.filter_images)
        if self.size == 0:
            raise RuntimeError('there is no image in %s' % directory_path)
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        super(DirectoryImageGenerator, self).__init__(self.size, batch_size, shuffle, seed)


class RotateImageCaptcha(object):

    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = 3
        self.learning_rate = 0.001

    def model(self):
        """
        ResNet50 + Flatten + 1 FC
        """
        input_tensor = keras.Input(shape=(self.image_height, self.image_width, self.image_channel), batch_size=None)
        input_tensor = applications.resnet50.preprocess_input(input_tensor, data_format='channels_last')
        net = applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=input_tensor, pooling='max')
        x = net.output
        # flatten
        x = layers.Flatten()(x)
        """
        # 效果不明显
        x = layers.Dense(units=128, activation='relu')(x)
        x = layers.Dropout(rate=0.25)(x)
        """
        # full connection
        x = layers.Dense(units=1, activation='sigmoid')(x)
        model = models.Model(inputs=net.inputs, outputs=x, name="rotateresnet50")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss="binary_crossentropy",
                      metrics=[metrics.BinaryAccuracy(threshold=0.9)])
        return model


def test(image_path, model_path='rotatemodel/model.h5'):
    image_height = 350
    image_width = 350
    image_channel = 3
    color_mode = 'rgb' if image_channel == 3 else 'grayscale'
    model = RotateImageCaptcha(image_height, image_width).model()
    model.load_weights(model_path)
    data = np.full(shape=(1, image_height, image_width, image_channel), fill_value=-1, dtype='float32')
    img = keras_preprocessing.image.utils.load_img(image_path, color_mode=color_mode)
    image_array = keras_preprocessing.image.utils.img_to_array(img)
    data[0] = image_array
    if hasattr(img, 'close'):
        img.close()
    data = applications.resnet50.preprocess_input(data, data_format='channels_last')
    result = model.predict(data)
    result = tf.reshape(result, [-1])
    np_result = keras.backend.eval(result)
    print(np_result)


def train(train_data_dir, epochs=10, model_path='rotatemodel/model.h5'):
    # load data
    image_height = 350
    image_width = 350
    image_channel = 3
    # 根据图片大小 可用显存调整
    batch_size = 128
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=model_path)
    ]
    captcha = RotateImageCaptcha(image_height, image_width)
    captcha.image_channel = image_channel
    model = captcha.model()
    model.summary()
    # load weights
    if os.path.exists(model_path):
        model.load_weights(model_path)
    # generator
    generator = DirectoryImageGenerator(train_data_dir, image_height, image_width, image_channel,
                                        batch_size=batch_size,
                                        seed=int(time.monotonic()))
    # train
    model.fit(x=generator, validation_data=None, epochs=epochs, callbacks=callbacks, steps_per_epoch=generator.size//batch_size)


if __name__ == '__main__':
    #test('rotate_origin/0_340_1.jpeg')
    train()
