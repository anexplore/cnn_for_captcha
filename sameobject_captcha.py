# -*- coding: utf-8 -*-
"""
<p>点击图片相似物体类验证码</p>
<pre>
* 基于YoloV5 object detection实现
* 需要标注训练数据
</pre>
"""
import torch
from yolov5.models import experimental
from yolov5.utils import torch_utils
from yolov5.utils import datasets
from yolov5.utils import general


class ObjectFinderByYolo(object):

    def __init__(self, confidence_threshold=0.25, iou_threshold=0.4):
        """
        :param confidence_threshold: 用于非极大值抑制时的置信度阈值
        :param iou_threshold: iou 阈值
        """
        self.model = None
        self.device = None
        self.labels = None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

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
        self.labels = model.names
        model.float().eval()
        self.model = model
        self.device = device

    def detect_object(self, img_path, img_size=None):
        """
        找到提供的所有图片中的符合一定阈值的Object

        :param img_path: 图片路径 可以为单张图片 也可以图片目录
        :param img_size: 图片(高 宽) 这里需要和训练模型时传递的图片size参数一致, 可以为单个int 或者 tuple int, 比如 344, (344, 344)
        :return: dict()  path -> [类别索引，类别，置信度，边框(x, y, w, h)] xy为左上角坐标
        """
        stride = max(int(self.model.stride.max()), 32)
        imgsz = general.check_img_size(img_size, s=stride)
        dataset = datasets.LoadImages(img_path, img_size=imgsz, stride=stride, auto=True)
        result = dict()
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
            pred = general.non_max_suppression(pred[0], self.confidence_threshold, self.iou_threshold, None, False, max_det=1000)
            # Process predictions
            for _, det in enumerate(pred):
                if len(det):
                    # 转换回原始图片尺度
                    det[:, :4] = general.scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        box = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                        box[2] = (box[2] - box[0])
                        box[3] = (box[3] - box[1])
                        confidence_value = conf.item()
                        class_index = int(cls.item())
                        result[path] = [class_index, self.labels[class_index], confidence_value, box]
                        """
                        print('conf %s, class %s, box: %s' % (confidence_value, class_index, box))
                        ann = plots.Annotator(im0s.copy())
                        ann.box_label(xyxy, label)
                        im0 = ann.result()
                        cv2.imshow('label', im0)
                        cv2.waitKey(5000)
                        """
        return result
