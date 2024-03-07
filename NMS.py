import cv2
import numpy as np
import copy


seed = 10001
np.random.seed(seed)

bounding_boxes = [
        [545, 125, 765, 440],
        [890, 100, 1115, 430],
        [1275, 170, 1490, 490]
    ]

confidence_score = [0.95, 0.98, 0.96]

num_anchor = 10
anchors = copy.deepcopy(bounding_boxes)
scores = copy.deepcopy(confidence_score)


def nms(bboxes, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    scores= y2-y1
    # 从大到小对应的的索引
    order = scores.argsort()[::-1]

    # 记录输出的bbox
    keep = []
    while order.size > 0:
        i = order[0]
        # 记录本轮最大的score对应的index
        keep.append(i)

        if order.size == 1:
            break
        # 计算当前bbox与剩余的bbox之间的IoU
        # 计算IoU需要两个bbox中最大左上角的坐标点和最小右下角的坐标点
        # 即重合区域的左上角坐标点和右下角坐标点
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 如果两个bbox之间没有重合, 那么有可能出现负值
        w = np.maximum(0.0, (xx2 - xx1))
        h = np.maximum(0.0, (yy2 - yy1))
        inter = w * h

        # iou = inter / (areas[i] + areas[order[1:]] - inter)

        #跟最小的那个比
        iou = inter / np.minimum(areas[i],areas[order[1:]])
        # iou = inter /[areas[i],areas[order[1:]]].min()

        # 删除IoU大于指定阈值的bbox(重合度高), 保留小于指定阈值的bbox
        ids = np.where(iou <= threshold)[0]
        # 因为ids表示剩余的bbox的索引长度
        # +1恢复到order的长度
        order = order[ids + 1]

    return keep

#去除包含关系的框
def remove_little(bboxes):
    for box in bboxes:
        x1, y1, x2, y2 = box
        for box2 in bboxes:
            x3, y3, x4, y4 = box2
            if x1 < x3 and y1 < y3 and x2 > x4 and y2 > y4:
                bboxes.remove(box2)
    return bboxes