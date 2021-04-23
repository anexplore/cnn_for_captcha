# -*- coding: utf-8 -*-
"""
CNN训练定长验证码识别模型
"""
import json
import io
import os
import os.path


import keras_preprocessing.image
import numpy as np
from PIL import Image as pil_image
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def label_to_array(text, labels):
    """
    转换成向量
    :param text: 验证码
    :param labels: 验证码所有可能字符集合
    :return: numpy array
    """
    hots = np.zeros(shape=(len(labels) * len(text)))
    for i, char in enumerate(text):
        index = i * len(labels) + labels.index(char)
        hots[index] = 1
    return hots


def array_to_label(array, labels):
    """
    向量转换成label
    :param array: numpy array
    :param labels: label
    :return: label string
    """
    text = []
    for index in array:
        text.append(labels[index])
    return ''.join(text)


def load_image_data(image_dir_path, image_height, image_width, labels, target_label_length):
    """
    加载图片数据
    图片标签从图片文件名中读取 图片文件名应该符合 label_xxxx.jpg(png)格式
    RGB图片将会转换成灰度图片
    :param image_dir_path: 图片路径
    :param image_height: 图片高度
    :param image_width: 图片宽度
    :param labels: 所有标签
    :param target_label_length: 图片标签固定长度
    :return: image_data, data_label
    """
    image_name_list = os.listdir(image_dir_path)
    image_data = np.zeros(shape=(len(image_name_list), image_height, image_width, 1))
    label_data = np.zeros(shape=(len(image_name_list), len(labels) * target_label_length))

    for index, image_name in enumerate(image_name_list):
        img = keras_preprocessing.image.utils.load_img(os.path.join(image_dir_path, image_name), color_mode='grayscale')
        x = keras_preprocessing.image.utils.img_to_array(img)
        y = label_to_array(image_name.split('_')[0], labels)
        if hasattr(img, 'close'):
            img.close()
        image_data[index] = x
        label_data[index] = y
    return image_data, label_data


class FixCaptchaLengthModel(object):
    """
    定长验证码模型
    Attributes:
        image_height: 高度
        image_width: 宽度
        learning_rate: 学习率
        dropout: dropout比例
        label_number: 所有可能字符的种类数量
        fixed_length: 验证码的固定长度
    """

    def __init__(self, image_height, image_width, label_number, fixed_length,
                 learning_rate=0.0001, dropout=0.25):
        self.image_height = image_height
        self.image_width = image_width
        # 这里固定转化为灰度图像
        self.image_channel = 1
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.label_number = label_number
        self.fixed_length = fixed_length

    def model(self):
        """
        :return: keras.Sequential instance
        """
        model = keras.Sequential()
        # 输入层
        input = keras.Input(shape=(self.image_height, self.image_width, self.image_channel), batch_size=None)
        model.add(input)
        # 第一层 卷积
        model.add(layers.Convolution2D(filters=32, kernel_size=(3, 3), strides=1, padding="valid", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Dropout(rate=self.dropout))
        # 第二层 卷积
        model.add(layers.Convolution2D(filters=64, kernel_size=(3, 3), strides=1, padding="valid", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Dropout(rate=self.dropout))
        # 第三层 卷积
        model.add(layers.Convolution2D(filters=128, kernel_size=(3, 3), strides=1, padding="valid", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Dropout(rate=self.dropout))
        model.add(layers.Flatten())
        # 第三层 全连接
        model.add(layers.Dense(units=1024, activation='relu'))
        model.add(layers.Dropout(rate=self.dropout))
        # 第四层 全连接
        model.add(layers.Dense(units=self.fixed_length * self.label_number, activation="sigmoid"))
        model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss="binary_crossentropy",
                      metrics=["binary_accuracy"])
        return model

    def load_from_disk(self, model_file_path):
        """
        从文本磁盘加载已经训练好的模型
        :param model_file_path: 模型文件路径
        :return: keras.Sequential
        """
        if not os.path.exists(model_file_path):
            raise Exception('%s do not exists' % model_file_path)
        model = self.model()
        model.load_weights(model_file_path)
        return model


class CheckAccuracyCallback(keras.callbacks.Callback):
    """
    检查上一轮的训练准确率
    """
    def __init__(self, input_x, input_y, test_x, test_y, label_number, fixed_label_length, batch_size=128):
        super(CheckAccuracyCallback, self).__init__()
        self.input_x = input_x
        self.input_y = input_y
        self.test_x = test_x
        self.test_y = test_y
        self.label_number = label_number
        self.fixed_label_length = fixed_label_length
        self.batch_size = batch_size

    def _compare_accuracy(self, data_x, data_y):
        predict_y = self.model.predict(data_x)
        predict_y = keras.backend.reshape(predict_y, [len(data_x), self.fixed_label_length, self.label_number])
        data_y = keras.backend.reshape(data_y, [len(data_y), self.fixed_label_length, self.label_number])
        equal_result = keras.backend.equal(keras.backend.argmax(predict_y, axis=2),
                                           keras.backend.argmax(data_y, axis=2))
        return keras.backend.mean(keras.backend.min(keras.backend.cast(equal_result, tf.float32), axis=1))

    def on_epoch_end(self, epoch, logs=None):
        print('\nEpoch %s with logs: %s' % (epoch, logs))
        # 选择一个batch并计算准确率
        batches = (len(self.input_x) + self.batch_size - 1) / self.batch_size
        target_batch = (epoch + 1) % batches
        batch_start = int((target_batch - 1) * self.batch_size)
        batch_x = self.input_x[batch_start: batch_start + self.batch_size]
        batch_y = self.input_y[batch_start: batch_start + self.batch_size]
        on_train_batch_acc = self._compare_accuracy(batch_x, batch_y)
        print('Epoch %s with image accuracy on train batch: %s' % (epoch, keras.backend.eval(on_train_batch_acc)))
        on_test_batch_acc = self._compare_accuracy(self.test_x, self.test_y)
        print('Epoch %s with image accuracy on test: %s\n' % (epoch, keras.backend.eval(on_test_batch_acc)))


class Config(object):

    def __init__(self, **kwargs):
        self.image_height = kwargs['image_height']
        self.image_width = kwargs['image_width']
        self.fixed_length = kwargs['fixed_length']
        self.train_batch_size = kwargs['batch_size']
        self.model_save_path = kwargs['save_path']
        self.labels = kwargs['labels']
        self.train_image_dir = kwargs['train_image_dir']
        self.validation_image_dir = kwargs['validation_image_dir']
        self.learning_rate = kwargs['learning_rate']
        self.dropout_rate = kwargs['dropout_rate']
        self.epochs = kwargs['epochs']

    @staticmethod
    def load_configs_from_json_file(file_path='fixed_length_captcha.json'):
        """
        {
          ""
        }
        :param file_path: file path
        :return: dict instance
        """
        with open(file_path, 'r') as fd:
            config_content = fd.read()
        return Config(**json.loads(config_content))


class Predictor(object):
    """
    预测器
    """
    def __init__(self, config_file_path='fixed_length_captcha.json'):
        self.config = Config.load_configs_from_json_file(config_file_path)
        self.model = FixCaptchaLengthModel(self.config.image_height, self.config.image_width, len(self.config.labels),
                                           self.config.fixed_length, learning_rate=self.config.learning_rate,
                                           dropout=self.config.dropout_rate).load_from_disk(self.config.model_save_path)
        self.label_number = len(self.config.labels)

    def predict(self, image_file_path):
        """
        预测单张图片
        :param image_file_path: 单张图片文件路径
        :return: predict text
        """
        with open(image_file_path, 'rb') as f:
            return self.predict_single_image_content(f.read())

    def predict_remote_image(self, remote_image_url, headers=None, timeout=30, save_image_to_file=None):
        """
        预测远程图片
        :param remote_image_url: 远程图片URL
        :param headers: 请求头
        :param timeout: 超时时间
        :param save_image_to_file: 是否保存图片到文件
        :return: predict text
        """
        response = requests.get(remote_image_url, headers=headers, timeout=timeout, stream=True)
        content = response.content
        if save_image_to_file is not None:
            with open(save_image_to_file, 'wb') as fd:
                fd.write(content)
        return self.predict_single_image_content(content)


    def predict_single_image_content(self, image_content):
        """
        预测单张图片
        :param image_content: byte content
        :return: predict text
        """
        p_image = pil_image.open(io.BytesIO(image_content))
        if p_image.mode not in ('L', 'I;16', 'I'):
            p_image = p_image.convert('L')
        image_data = np.zeros(shape=(1, self.config.image_height, self.config.image_width, 1))
        image_data[0] = keras_preprocessing.image.img_to_array(p_image)
        if hasattr(p_image, 'close'):
            p_image.close()
        result = self.model.predict(image_data)
        result = keras.backend.reshape(result, [1, self.config.fixed_length, self.label_number])
        result = keras.backend.argmax(result, axis=2)
        return array_to_label(keras.backend.eval(result)[0], self.config.labels)


def train():
    """
    训练
    """
    config = Config.load_configs_from_json_file()
    train_x, train_y = load_image_data(config.train_image_dir, config.image_height, config.image_width,
                                       config.labels, config.fixed_length)
    test_x, test_y = load_image_data(config.validation_image_dir, config.image_height, config.image_width,
                                     config.labels, config.fixed_length)
    print('total train image number: %s' % len(train_x))
    print('total validation image number: %s' % len(train_y))
    model = FixCaptchaLengthModel(config.image_height, config.image_width, len(config.labels), config.fixed_length,
                                  learning_rate=config.learning_rate, dropout=config.dropout_rate)
    if os.path.exists(config.model_save_path):
        model = model.load_from_disk(config.model_save_path)
    else:
        model = model.model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=config.model_save_path),
        CheckAccuracyCallback(train_x, train_y, test_x, test_y, len(config.labels), config.fixed_length,
                              batch_size=config.train_batch_size)
    ]
    model.fit(train_x, train_y, batch_size=config.train_batch_size, epochs=config.epochs,
              validation_data=(test_x, test_y), callbacks=callbacks)


if __name__ == '__main__':
    train()
    """
    pre = Predictor()
    print(pre.predict_remote_image('remote url', save_image_to_file='test.jpg'))
    """