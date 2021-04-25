# cnn_for_captcha

**基于深度学习的图片验证码识别**

## 1. 固定长度的文字验证码识别 fixed_length_captcha.py

### 依赖
_**tensorflow-gpu==1.15.5**_

### 1.1 输入要求
* 将训练集和验证集分别放到配置文件指定的目录中
* 图片文件保持宽高一致
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
