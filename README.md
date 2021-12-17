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
