# -*- coding: utf-8 -*-
"""
将labelme的标注数据转换成yolov5格式
"""
import json
import sys
import os.path


def build_class_map(classes_file_path, dest_classes_file_path):
    class_index_map = dict()
    dest = open(dest_classes_file_path, 'w', encoding='utf8')
    with open(classes_file_path, 'r', encoding='utf8') as fd:
        index = 0
        for line in fd:
            line = line.strip()
            if line in class_index_map:
                continue
            class_index_map[line] = index
            index += 1
            dest.write('%s\n' % line)
    dest.close()
    return class_index_map


labelme_annotation_dir = sys.argv[1]
yolov5_annotation_dir = sys.argv[2]

if not os.path.exists(yolov5_annotation_dir):
    os.mkdir(yolov5_annotation_dir)


class_index_map = build_class_map(os.path.join(labelme_annotation_dir, 'classes.txt'),
                                  os.path.join(yolov5_annotation_dir, 'classes.txt'))
file_number = 0
print('...start...')
for filename in os.listdir(labelme_annotation_dir):
    if not filename.endswith('.json'):
        print('ignore file: %s' % filename)
        continue
    file_number += 1
    fullpath = os.path.join(labelme_annotation_dir, filename)
    destpath = os.path.join(yolov5_annotation_dir, '%s.txt' % filename[0:filename.rindex('.')])
    with open(fullpath, 'r', encoding='utf8') as fd:
        content = fd.read()
    destfile = open(destpath, 'w', encoding='utf8')
    data_dict = json.loads(content, encoding='utf8')
    shapes = data_dict['shapes']
    image_height = data_dict['imageHeight']
    image_width = data_dict['imageWidth']
    for shape in shapes:
        label = shape['label']
        label_index = class_index_map.get(label)
        if label_index is None:
            raise Exception('%s cannot find index, please check classes.txt' % label)
        points = shape['points']
        topx, topy = points[0]
        bottomx, bottomy = points[1]
        # 归一化到 0-1 xy为边框中心点 wh为边框宽高
        x = (topx + bottomx) / 2 / image_width
        y = (topy + bottomy) / 2 / image_height
        w = (bottomx - topx) / image_width
        h = (bottomy - topy) / image_height
        destfile.write('%s\t%s\t%s\t%s\t%s\n' % (label_index, x, y, w, h))
    destfile.close()

print('class number: %s, file number: %s' % (len(class_index_map), file_number))
print('...end...')