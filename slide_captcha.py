# -*- coding: utf-8 -*-
"""
滑动验证码相关
"""
import cv2
import numpy as np


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


def _process_image(image, blur=False):
    """
    预处理; 部分网站的图片先模糊再求边缘的匹配效果 不如直接求边缘的匹配效果好
    <pre>
        模糊
        求边界
    <pre>
    :param image: image mat
    :param blur 是否模糊
    :return: handle image mat
    """
    if blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.Canny(image, 50, 150)


def _read_image_from_local_file(image_path, image_scale=cv2.IMREAD_GRAYSCALE):
    with open(image_path, 'rb') as fd:
        content = fd.read()
    return cv2.imdecode(np.frombuffer(content, dtype=np.uint8), image_scale)


def _read_image_from_bytes(image_bytes, image_scale=cv2.IMREAD_GRAYSCALE):
    if not isinstance(image_bytes, bytes):
        raise RuntimeError('image bytes must be bytes type')
    return cv2.imdecode(image_bytes, image_scale)


def detect_displacement(image_slider, image_background, blur=False, display_image=True):
    """
    探测缺口偏移量
    :param image_slider: 缺口图 numpy.ndarray or image file path
    :param image_background: 底图 numpy.ndarray or image file path
    :param blur: 预处理时是否模糊图片
    :param display_image: 展示图片
    :return: top_left_x, top_left_y
    """
    if isinstance(image_slider, str):
        image_slider = _read_image_from_local_file(image_slider)
    if isinstance(image_background, str):
        image_background = _read_image_from_local_file(image_background)
    processed_image_slider = _process_image(image_slider, blur=blur)
    processed_image_background = _process_image(image_background, blur=blur)
    # match
    res = cv2.matchTemplate(processed_image_slider, processed_image_background, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_location = cv2.minMaxLoc(res)
    # pos
    x, y = max_location
    # height width
    h, w = image_slider.shape
    # draw match
    cv2.rectangle(image_background, (x, y), (x + w, y + h), (255, 255, 255), 2)
    if display_image:
        show_image("processed_image_slider", processed_image_slider)
        show_image("processed_image_background", processed_image_background)
        show_image("match", image_background)
    return x, y

