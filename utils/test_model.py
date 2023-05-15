import os
from typing import Iterable

import cv2 as cv
import numpy as np
import torch

from MyYOLOX.boxes import IOULoss
from .dataset import validimg_transform
from .others import Timer

import torchvision


def nms_torch(pred_boxes, pred_confid, iou_thres):
    min_x = torch.min(pred_boxes[..., [0, 2, 4, 6]], dim=-1).values
    min_y = torch.min(pred_boxes[..., [1, 3, 5, 7]], dim=-1).values
    max_x = torch.max(pred_boxes[..., [0, 2, 4, 6]], dim=-1).values
    max_y = torch.max(pred_boxes[..., [1, 3, 5, 7]], dim=-1).values
    boxes = torch.stack((min_x, min_y, max_x, max_y), dim=-1)
    return torchvision.ops.nms(boxes, pred_confid, iou_thres)


def nms(pred_y, confidence, iou_thres):
    mask = torch.zeros(pred_y.shape[0], dtype=torch.bool, device=pred_y.device)
    for i in range(pred_y.shape[0]):
        if pred_y[i, 8] < confidence:
            continue
        for j in range(pred_y.shape[0]):
            if pred_y[j, 8] > confidence and i != j and IOULoss.get_IOU(
                    pred_y[None, i, :8], pred_y[None, j, :8]
                    ).item() > iou_thres:
                if pred_y[i, 8] > pred_y[j, 8]:
                    mask[i] = True
                    mask[j] = False
                    pred_y[j, 8] = 0
                else:
                    mask[j] = False
                    mask[j] = True
                    pred_y[i, 8] = 0
    return mask


def matching_pred_gt(pred, gt, iou_thresh):
    matching_matrix = torch.zeros(gt.shape[0], pred.shape[0], dtype=torch.long)
    gt = gt.unsqueeze(1).repeat(1, pred.shape[0], 1)
    pred = pred.unsqueeze(0).repeat(gt.shape[0], 1, 1)
    # ! shape is [gt, pred, 8]
    pair_wise_ious = IOULoss.get_IOU(pred, gt, train_mode=False)

    matching_matrix[pair_wise_ious > iou_thresh] = 1

    # !sum anchor matching gts num anchor-wise
    anchor_matching_gt = matching_matrix.sum(0)
    # !if exist anchor matching more than one gt
    if (anchor_matching_gt > 1).sum() > 0:
        # !pick the most matching gt anchor-wise for those have more than one matching
        # !anchors.
        iou_max_indices = pair_wise_ious[:, anchor_matching_gt > 1].argmax(dim=0)
        matching_matrix[:, anchor_matching_gt > 1] = 0
        matching_matrix[iou_max_indices, anchor_matching_gt > 1] = 1
    # !pick those anchors that have matching gt.
    anchor_matching_mask = matching_matrix.sum(0) > 0
    gt_matching_mask = matching_matrix.sum(1) > 0
    matching_matrix = matching_matrix[gt_matching_mask, :]
    matched_anchor_inds = matching_matrix.argmax(1) \
        if matching_matrix.size(1) > 0 else \
        torch.zeros_like(matching_matrix, dtype=torch.long)
    return anchor_matching_mask, gt_matching_mask, matched_anchor_inds


@torch.no_grad()
def test(epoch_, loader, model, confidence, device, iou_thresh, precision=3):
    TP, FP, FN, TN = 0, 0, 0, 0
    correct, total = 0, 0
    model.eval()
    model.to(device)
    timer = Timer(precision=precision)
    timer.start()
    for X, Y, N in loader:
        # ! pred_Y shape [batch, n_anchors_all, reg:8+obj:1+cls:24]
        # ! Y shape [batch, max_label_per_image, cls:1+reg:8]
        X, Y = X.to(device), Y.to(device)
        pred_Y = model(X)
        # ! pic-wise
        for x, y, n, pred_y in zip(X, Y, N, pred_Y):
            # ! confid_y shape is [confidence_pred, reg:8+obj:1+cls:24]
            pred_confid = pred_y[..., 8]
            confid_mask = pred_confid > confidence
            confid_y = pred_y[confid_mask]
            # nms_mask = nms(confid_y, confidence, iou_thresh)
            pred_boxes = confid_y[..., :8]
            nms_mask = nms_torch(pred_boxes, pred_confid[confid_mask], iou_thresh)
            confid_boxes = pred_boxes[nms_mask]
            confid_cls = confid_y[nms_mask, 9:]
            anchor_mask, gt_mask, anchor_inds = matching_pred_gt(
                    confid_boxes, y[:n, 1:], iou_thresh
                    )
            TP += anchor_mask.sum().item()
            FP += (~anchor_mask).sum().item()
            FN += (~gt_mask).sum().item()
            TN += pred_y.size(0) - anchor_mask.size(0)
            if confid_cls.numel() > 0:
                cls_pred = confid_cls[anchor_inds].argmax(dim=1)
                cls_gt = y[:n][gt_mask][:, 0]
                correct += (cls_pred == cls_gt).sum().item()
                total += cls_gt.size(0)
    try:
        res_dict = {
                'cls acc':   round(correct / total, precision),
                'Accuracy':  round((TP + TN) / (TP + FN + FP + TN), precision),
                'Precision': round(TP / (TP + FP), precision),
                'Recall':    round(TP / (TP + FN), precision),
                'F1-score':  round(2 / (1 / (TP / (TP + FP)) + 1 / (TP / (TP + FN))), precision)
                }
        res = (
                f'Test {epoch_}: total = {total}, cls acc = '
                f'{res_dict["cls acc"]}, '
                f'using time = {timer.stop(False)}s\n'
                f'Accuracy = {res_dict["Accuracy"]} '
                f'Precision = {res_dict["Precision"]} '
                f'Recall = {res_dict["Recall"]} '
                f'F1-score = {res_dict["F1-score"]}\n'
                f'TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}')
    except ZeroDivisionError:
        res = f'Test {epoch_}: ZeroDivisionError When Testing.'
        res_dict = None
    print(res)
    model.train()
    return res, res_dict


@torch.no_grad()
def infer(model, img, confidence, draw=True, device=torch.device("cpu")):
    model.eval()
    img = cv.resize(img, (640, 480))
    pic = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pic = validimg_transform(pic)
    anchors = model(pic[None, ...].to(device)).cpu()
    confid_y = anchors[anchors[..., 8] > confidence]
    confid_box = confid_y[:, :8]
    confid_cls = confid_y[:, 9:].argmax(dim=-1)
    if draw:
        for anchor, cls in zip(confid_box, confid_cls):
            print("cls =", cls)
            for i in range(0, 8, 2):
                cv.line(
                        img, (int(anchor[i - 2]), int(anchor[i - 1])),
                        (int(anchor[i]), int(anchor[i + 1])), (0, 255, 0), 1
                        )
    return confid_box, confid_cls, img


@torch.no_grad()
def test_model(model, folder, confidence, device):
    model.load_state_dict(torch.load('pickup_model/v1.2.pt', map_location=device))
    files = os.listdir(folder)
    np.random.shuffle(files)
    for file in files:
        if os.path.splitext(file)[1] not in ".jpg .png":
            continue
        img = cv.imread(os.path.join(folder, file))
        box, cls, img = infer(model, img, confidence)
        print("total box num: ", len(box))
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyWindow('img')


@torch.no_grad()
def calc_mAP(model, loader, device, iou_thresh, confid_range=None, precision=3):
    if confid_range is None:
        confid_range = (x / 10 for x in range(3, 10))
    model.eval()
    model.to(device)
    curve_x, curve_y = [], []
    timer = Timer(precision=precision)
    timer.start()
    for confidence in confid_range:
        TP, FP, FN, TN = 0, 0, 0, 0
        correct, total = 0, 0
        for X, Y, N in loader:
            # ! pred_Y shape [batch, n_anchors_all, reg:8+obj:1+cls:24]
            # ! Y shape [batch, max_label_per_image, cls:1+reg:8]
            X, Y = X.to(device), Y.to(device)
            pred_Y = model(X)
            # ! pic-wise
            for x, y, n, pred_y in zip(X, Y, N, pred_Y):
                # ! confid_y shape is [confidence_pred, reg:8+obj:1+cls:24]
                pred_confid = pred_y[..., 8]
                confid_mask = pred_confid > confidence
                confid_y = pred_y[confid_mask]
                # nms_mask = nms(confid_y, confidence, iou_thresh)
                pred_boxes = confid_y[..., :8]
                nms_mask = nms_torch(pred_boxes, pred_confid[confid_mask], iou_thresh)
                confid_boxes = pred_boxes[nms_mask]
                confid_cls = confid_y[nms_mask, 9:]
                anchor_mask, gt_mask, anchor_inds = matching_pred_gt(
                        confid_boxes, y[:n, 1:], iou_thresh
                        )
                TP += anchor_mask.sum().item()
                FP += (~anchor_mask).sum().item()
                FN += (~gt_mask).sum().item()
                TN += pred_y.size(0) - anchor_mask.size(0)
                if confid_cls.numel() > 0:
                    cls_pred = confid_cls[anchor_inds].argmax(dim=1)
                    cls_gt = y[:n][gt_mask][:, 0]
                    correct += (cls_pred == cls_gt).sum().item()
                    total += cls_gt.size(0)
        curve_y.append(round(TP / (TP + FP), precision))
        curve_x.append(round(TP / (TP + FN), precision))
    mAP = 0
    for idx in range(1, len(curve_x)):
        mAP += (curve_x[idx - 1] - curve_x[idx]) * max(curve_y[:idx])
    return mAP
