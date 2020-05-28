import cv2
import numpy as np
import argparse
import os
import time
from PIL import Image
import tensorrt as trt
import img_utils
import common

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def postprocess_boxes(pred_bbox, org_img_shape, input_shape, score_threshold):
    valid_scale=[0, np.inf]

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    input_h, input_w = input_shape
    resize_ratio = min(input_w / org_w, input_h / org_h)

    dw = (input_w - resize_ratio * org_w) / 2
    dh = (input_h - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def reshape_output(output, num_classes):
    output = np.transpose(output, [1, 2, 3, 0])
    _, height, width, _ = output.shape
    dim1, dim2 = height, width
    dim3 = 3
    dim4 = (4 + 1 + num_classes)

    pred_bbox = np.reshape(output, (dim1*dim2*dim3, dim4))
    return pred_bbox

def process_outputs(trt_outputs, num_classes, resolution_raw, input_shape, conf_th):
    outputs_reshaped = list()
    for trt_output in trt_outputs:
        reshaped = reshape_output(trt_output, num_classes)
        outputs_reshaped.append(reshaped)

    pred_bbox = np.concatenate(outputs_reshaped, axis=0)
    bboxes = postprocess_boxes(pred_bbox, resolution_raw, input_shape, conf_th)
    bboxes = nms(bboxes, 0.6, method='nms')

    return bboxes

def main():
    # model
    label_file_path = 'models/starwars.names'
    engine_file_path = "models/starwars_yolov3_fp16.trt"

    # label list
    all_classes = img_utils.load_label_classes(label_file_path)
    num_classes = len(all_classes)

    trt_runtime = trt.Runtime(TRT_LOGGER)

    print(f"Lade TensorRT Engine {engine_file_path}")
    trt_engine = common.load_engine(trt_runtime, engine_file_path)

    inputs, outputs, bindings, streams = common.allocate_buffers(trt_engine)
    context = trt_engine.create_execution_context()

    new_width = 416
    new_height = 416
    input_shape = (new_width, new_height)
    output_shapes = [(1, -1, new_height // 32, new_width // 32),
                     (1, -1, new_height // 16, new_width // 16),
                     (1, -1, new_height //  8, new_width //  8)]

    # open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kann die Webcam, nicht öffnen")
        exit()

    while cap.isOpened():
        
        # read frame from webcam 
        status, frame = cap.read()

        if not status:
            print("Kann kein Bild laden")
            exit()

        #fps = cap.get(cv2.CAP_PROP_FPS)
        #print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")


        image_resized = cv2.resize(frame, (new_width, new_height), interpolation = cv2.INTER_AREA)
        image_resized = np.array(image_resized, dtype=np.float32, order='C')
        image_resized /= 255.0
        image_processed = np.transpose(image_resized, [2, 0, 1])
        image_processed = np.expand_dims(image_processed, axis=0)
        image_processed = np.array(image_processed, dtype=np.float32, order='C')


        inputs[0].host = image_processed

        trt_outputs = common.do_inference_v2(
           context,
           bindings=bindings,
           inputs=inputs,
           outputs=outputs,
           stream=streams
        )

        trt_outputs  = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

        resolution_raw = (int(frame.shape[1]), int(frame.shape[0]))

        bboxes = process_outputs(trt_outputs, num_classes, resolution_raw, input_shape, 0.6)
        

        # loop through detected bounding boxes
        for bbox in bboxes:
            # get corner points of face rectangle        
            coor = np.array(bbox[:4], dtype=np.int32)
            (startX, startY) = coor[0], coor[1]
            (endX, endY) = coor[2], coor[3]

            # draw rectangle over the detected object
            cv2.rectangle(frame, (int(startX+20),int(startY+20)), (int(endX-20),int(endY-20)), (0,255,0), 2)

            # get label with max accuracy
            score = bbox[4]
            score = '%.2f' % score

            class_ind = int(bbox[5])
            class_name = all_classes[class_ind]

            print(f"{class_name} LEGO Figur erkannt zu {score}%")

            # write label and confidence above face rectangle
            cv2.putText(frame, class_name, (startX, startY),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

        # display output
        cv2.imshow("LEGO Star Wars Object Detection", frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
