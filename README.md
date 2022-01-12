# cnn_for_captcha

**基于深度学习的图片验证码识别**

## 1. 固定长度的文字验证码识别 fixed_length_captcha.py

### 依赖
_**tensorflow 2.X**_

### 1.1 输入要求
* 将训练集和验证集分别放到配置文件指定的目录中
* 目录中所有图片尺寸相同
* 图片命名规则 验证码_编号.图片格式, 举例 **abce_012312.jpg**

### 1.2 配置文件
* 默认文件 fixed_length_captcha.json
* 字段见文知义

### 1.3 训练
~~~ python
python fixed_length_captcha.py
~~~

### 1.4 预测
~~~ python
predictor = Predictor()
# 预测本地磁盘文件
predictor.predict('xxx.jpg')
# 直接二进制内容预测
predictor.predict_single_image_content(b'PNGxxxxx')
# 预测远程图片
predictor.predict_remote_image('http://xxxxxx/xx.jpg', save_image_to_file='remote.jpg')
~~~

### 1.5 效果
* 根训练集样本大小有关
* ![image](images/4x4e_11039.png) 这种图片2w张左右的训练集训练后实际能达到90%以上的准确率


## 2. 滑动验证码 slide_captcha.py
提供滑动验证码相关解决方法与思路
### 2.1 基于opencv2的match template
此方法简单易于验证, 配合一些规则即可达到满意效果

以下是match template的效果

![match template](images/matchtemplate.png)

### 2.2 基于YOLOv5的检测
此方法需要标注数据以及进行训练但相对于模板探测是更稳定的通用方案

基于 [yolo v5](https://github.com/ultralytics/yolov5) 进行模型的训练与缺口探测

训练方法参考 [train custom data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
~~~text
本项目中yolov5中提供了100张标注好的图片;
参数:
batch-size 根据可用内存或者显存以及训练效果;
epochs 根据训练效果来定
yolov5s.yaml在yolov5项目的models下面
img 图片缩放基准 建议用图片的宽或者高即可(需要考虑图片大小 如果图片过大建议调低此值)
weights 设置预训练模型 没有为空即可

python train.py --batch-size 4 --epochs 200 --img 344 --data displacement.yaml --weights '' --cfg yolov5s.yaml
~~~

探测方法参考 [detect.py](https://github.com/ultralytics/yolov5/blob/master/detect.py) 中run方法


下面是通过标注100张图片并经过训练得到的模型的探测效果
![yolov5](images/yolodetect.jpeg)

## X.其它
### X.1 图片数据切分
将准备好的图片按照比例切分成训练集和验证集
~~~ python
python split_data.py all_image_dir train_image_dir validation_image_dir 0.9
~~~
参数:
* all_image_dir 准备好的图片目录
* train_image_dir 训练集图片目录
* validation_image_dir 验证集图片目录
* 0.9 训练集比例

以上目录需要提前创建

### X.2 泄漏问题
* keras model.predict 或者 keras model predict_on_batch都存在内存泄漏问题，目前尚未找到解决方案
