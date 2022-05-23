"""yolo_model.py"""
from typing import Tuple, List, Union, Optional
import numpy as np
from numpy import ndarray as ARRAY
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
WEIGHTPATH = "model/yolo.weights"
CONFIGPATH = "model/yolo.cfg"
CLASSESPATH = "model/yolo_classes.txt"

net = cv2.dnn.readNet(WEIGHTPATH, CONFIGPATH)

CLASSES = [line.strip() for line in open(CLASSESPATH, "r").readlines()]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

layer_names = net.getLayerNames()
output_layers = []

if cv2.__version__ != '4.5.3':
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
else:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(
        img: ARRAY,
        size: Optional[Tuple[int, int]] = (416, 416),
        scale: Optional[float] = 0.00392,
        conf_threshold: Optional[float] = 0.4,
        nms_threshold: Optional[float] = 0.5,
        )->\
            Tuple[
                List[Union[int,int,int,int]],
                    List[str]]:
    """
    Detect objects in an image. and draw it's boxes

    Parameters
    ----------
    img : ARRAY
        The image to detect objects in.
    size : Optional[Tuple[int,int]]
        The size of the image to detect objects in.
    scale : Optional[float]
        The scale of the image to detect objects in.
    conf_threshold : Optional[float]
        The confidence threshold to use.
    nms_threshold : Optional[float]
        The NMS threshold to use.

    Returns
    -------
    boxes : List[Union[int,int,int,int]]
        The bounding boxes of the objects.
    labels : List[str]
        The labels of the objects.
    """

    blob = cv2.dnn.blobFromImage(
        img, scale, size, (0, 0, 0), False, crop=False)
    net.setInput(blob)
    # predict
    outs = net.forward(output_layers)

    width, height = img.shape[1], img.shape[0]
    class_ids, confidences, boxes, bbox, labels = [], [], [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                rel_width = int(detection[2] * width)
                rel_height = int(detection[3] * height)

                # Rectangle coordinates
                new_x = int(center_x - rel_width / 2)
                new_y = int(center_y - rel_height / 2)

                boxes.append([new_x, new_y, rel_width, rel_height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if cv2.__version__ != '4.5.3':
        for ind in indices:
            x, y, w, h = boxes[ind]
            bbox.append([int(x), int(y), int(x+w), int(y+h)])
            labels.append(str(CLASSES[class_ids[ind]]))
    else:
        for ind in indices:
            ind = ind[0]
            x, y, w, h = boxes[ind]
            bbox.append([int(x), int(y), int(x+w), int(y+h)])
            labels.append(str(CLASSES[class_ids[ind]]))

    return bbox, labels


def draw_boxes(
    img: ARRAY,
    bbox: List[Union[int, int, int, int]],
    labels: List[str],
) -> ARRAY:
    """
    Draw boxes on an image.

    Parameters
    ----------
    img : ARRAY
        The image to draw boxes on.
    bbox : List[Union[int,int,int,int]]
        The bounding boxes to draw.
    labels : List[str]
        The labels of the bounding boxes.

    Returns
    -------
    img : ARRAY
        The image with drawn boxes.
    """

    for ind, label in enumerate(labels):
        color = COLORS[ind]
        cv2.rectangle(img, (bbox[ind][0], bbox[ind][1]),
                      (bbox[ind][2], bbox[ind][3]), color, 2)
        cv2.putText(img, str(label + str(ind+1)), (bbox[ind][0],
                    bbox[ind][1]-10), FONT, 0.5, color, 2)

    return img
