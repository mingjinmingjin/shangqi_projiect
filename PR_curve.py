import numpy as np
from sklearn.metrics import auc

def compute_iou(box1, box2):
    """计算两个框的IoU值"""
    left = max(box1[0], box2[0])
    right = min(box1[1], box2[1])
    inter = max(0, right - left)
    union = box1[1] - box1[0] + box2[1] - box2[0] - inter
    iou = inter / union
    return iou

def compute_pr_curve(scores, labels, num_pos):
    """计算精度-召回率曲线"""
    sorted_inds = np.argsort(scores)[::-1]
    tp = labels[sorted_inds]
    fp = 1 - tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / num_pos
    precision = tp / (tp + fp)
    return recall, precision

def compute_map(test_data, test_annotation, test_detection, iou_thresh=0.5):
    """计算测试集MAP"""
    # num_classes = len(test_annotation)
    aps = []
    for c in range(36):
        num_pos = len(test_annotation[c])
        scores = []
        labels = []
        for i in range(len(test_data)):
            det = test_detection[i][c]
            if det is None:
                continue
            scores.append(det[0])
            box_det = det[1]
            ious = [compute_iou(box_det, box_gt) for box_gt in test_annotation[c][i]]
            if max(ious) >= iou_thresh:
                labels.append(1)
            else:
                labels.append(0)
        if sum(labels) == 0:
            ap = 0
        else:
            recall, precision = compute_pr_curve(scores, labels, num_pos)
            ap = auc(recall, precision)
        aps.append(ap)
    map = np.mean(aps)
    return map




















