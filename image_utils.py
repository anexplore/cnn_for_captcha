# -*- coding: utf-8 -*-
"""
图片处理相关函数
"""
import cv2


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


def image2jpeg(src, dst):
    """
    将图片转换成jpeg图片
    :param src: png图片路径
    :param dst: dst图片路径
    """
    image = cv2.imread(src)
    cv2.imwrite(dst, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def image2gray(src, dst):
    """
    将图片转换成灰度图
    """
    image = cv2.imread(src)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(dst, image)


def wait_for_destroy_windows():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image(name, image):
    """
    展示图片
    :param name: window name
    :param image:  image mat
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow(name, image)
