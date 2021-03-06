#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:07:27
#   Description :
#
#================================================================

import time
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image

input_size   = 416
image_path   = "./tests/test9.jpg"
CLASSES      = ["r2d2","c3po","luke-skywalker","obi-wan-kinobi","sturmtruppler"]
NUM_CLASS    = len(CLASSES)

input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)

original_image      = cv2.imread(image_path)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights("./starwars_yolov3")
model.summary()

inference_start_time = time.time()
pred_bbox = model.predict(image_data)
print("tf.keras Erkennungszeit: {} ms".format(int(round((time.time() - inference_start_time) * 1000))))

pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
pred_bbox = tf.concat(pred_bbox, axis=0)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')

for bbox in bboxes:
    print(bbox)
    coor = np.array(bbox[:4], dtype=np.int32)
    score = bbox[4]
    class_ind = int(bbox[5])
    class_name = CLASSES[class_ind]
    score = '%.4f' % score
    xmin, ymin, xmax, ymax = list(map(str, coor))
    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
    print('\t' + str(bbox_mess).strip())

image = utils.draw_bbox(original_image, bboxes, classes=CLASSES)
image = Image.fromarray(image)
image.save("./test9-box.jpg")
#image.show()


