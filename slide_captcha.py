# -*- coding: utf-8 -*-
"""
滑动验证码相关
"""
import cv2
import numpy as np

# yolov5 project
try:
    import torch
    from yolov5.models import experimental
    from yolov5.utils import torch_utils
    from yolov5.utils import datasets
    from yolov5.utils import general
    from yolov5.utils import plots
except:
    pass


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
    :return: top_left_x, top_left_y, width, height
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
    return x, y, w, h

"""
基于YOLO的方法定位缺口需要标注数据且进行训练 时间成本相对较高
一般情况下使用cv2.matchTemplate基本就能框定出缺口的位置
使用 [yolo v5](https://github.com/ultralytics/yolov5) 进行训练
基于yolo v5中的detect.py改造自己的检查函数即可
"""


class DisplacementFinderByYolo(object):

    def __init__(self):
        self.model = None
        self.device = None

    def load_models(self, weights_file_path, **kwargs):
        """
        加载模型
        :param weights_file_path: weights文件路径
        :param kwargs: 其它控制参数
        """
        # 选择设备
        device = torch_utils.select_device()
        # 加载模型
        model = experimental.attempt_load([weights_file_path,], map_location=device)
        model.float().eval()
        self.model = model
        self.device = device

    def detect_displacement(self, img_path, img_size=None):
        """
        基于yolo v5中detect.py的run函数改造此函数即可
        方法一: 直接调用detect.run()方法并设置结果写出到txt文件 通过读取txt文件解析结果(此方法每次调用都需要重新加载模型 适合一次性大批量处理)
        方法二: 复用detect.run中代码，将模型加载放到 self._load_models中 将探测代码放到 detect_displacement中
        :param img_path: 图片路径
        :param img_size: 图片(高 宽) 这里需要和训练模型时传递的图片size参数一致
        :return: 类别，置信度，边框(x, y, w, h) x,y是左上角坐标
        """
        stride = max(int(self.model.stride.max()), 32)
        imgsz = general.check_img_size(img_size, s=stride)
        dataset = datasets.LoadImages(img_path, img_size=imgsz, stride=stride, auto=True)

        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            # uint8 to fp16/32
            im = im.float()
            # 0 - 255 to 0.0 - 1.0
            im /= 255
            if len(im.shape) == 3:
                # expand for batch 4-dim
                im = im[None]
            pred = self.model(im, augment=False, visualize=False)
            # 非极大值抑制
            pred = general.non_max_suppression(pred[0], 0.25, 0.4, None, False, max_det=1000)
            # Process predictions
            # per image
            for _, det in enumerate(pred):
                if len(det):
                    # process result
                    # 转换回原始图片尺度
                    det[:, :4] = general.scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        box = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                        box[2] = (box[2] - box[0])
                        box[3] = (box[3] - box[1])
                        confidence_value = conf.item()
                        class_index = cls.item()
                        return int(class_index), confidence_value, box

                    #print('conf %s, class %s, box: %s' % (confidence_value, class_index, box))
                    """
                    ann = plots.Annotator(im0s.copy())
                    ann.box_label(xyxy, 'dis')
                    im0 = ann.result()
                    cv2.imshow('dis', im0)
                    cv2.waitKey(5000)
                    """
        return None, None, None
