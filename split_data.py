# -*- coding: utf-8 -*-
"""
将图片分隔成训练集和验证集
"""
import os.path
import random
import shutil
import sys


all_image_dir = sys.argv[1]
train_image_dir = sys.argv[2]
validation_image_dir = sys.argv[3]

if len(sys.argv) > 4:
    train_image_ratio = float(sys.argv[4])
else:
    train_image_ratio = 0.95

all_images = os.listdir(all_image_dir)
all_image_count = len(all_images)
train_image_count = int(all_image_count * train_image_ratio)
validation_image_count = all_image_count - train_image_count
print('total has %s images, %s train images, %s validation images' % (all_image_count, train_image_count, validation_image_count))

random = random.Random()

target_list = all_images[0: validation_image_count]
index = validation_image_count
while index < all_image_count:
    r = random.randint(0, validation_image_count - 1)
    if r < validation_image_count:
        target_list[r] = all_images[index]
    index += 1

for image_name in all_images:
    if image_name in target_list:
        shutil.copy(os.path.join(all_image_dir, image_name), os.path.join(validation_image_dir, image_name))
    else:
        shutil.copy(os.path.join(all_image_dir, image_name), os.path.join(train_image_dir, image_name))

print('all done')