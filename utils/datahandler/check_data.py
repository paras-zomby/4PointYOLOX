import json
import os
from itertools import chain

import cv2 as cv
import numpy as np
import torch


def get_id(file_name):
    return int(file_name.split(sep='_')[0][3:])


def check_each_data(dataset_path, img_format='.jpg', shuffle=False):
    labels = {}
    pics = []

    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    for file in chain(os.listdir(train_path), os.listdir(valid_path)):
        if os.path.splitext(file)[1] == ".label":
            try:
                with open(os.path.join(train_path, file), "r") as f:
                    labels.setdefault(get_id(file), json.load(f))
            except FileNotFoundError:
                with open(os.path.join(valid_path, file), "r") as f:
                    labels.setdefault(get_id(file), json.load(f))
        elif os.path.splitext(file)[1] in img_format:
            pics.append(file)

    if shuffle:
        np.random.shuffle(pics)

    for pic in pics:
        img = cv.imread(os.path.join(train_path, pic))
        if img is None:
            img = cv.imread(os.path.join(valid_path, pic))
        pic_id = get_id(pic)
        label_n = labels[pic_id]
        cv.putText(
                img, f"ID {pic_id}", (0, 20), cv.FONT_HERSHEY_DUPLEX, 0.8,
                (255, 255, 0), 1
                )
        for idx, label in enumerate(label_n):
            points = label[1:9]
            for i in range(0, 8, 2):
                cv.putText(
                        img, str(i // 2 + 1), (int(points[i]), int(points[i + 1])),
                        cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, int(255 - idx * 80), idx * 80),
                        1
                        )
        cv.imshow("img", img)
        cv.waitKey(0)


def check_order(points, num=4):
    points = points.view(*points.shape[:-1], 4, 2)
    area = torch.zeros(*points.shape[:-2], dtype=torch.float, device=points.device)
    for i in range(num):
        area += torch.stack(
                (points[..., i, :],
                 points[..., i - 1, :]), dim=-2
                ).det()
    return area < 0


def correct_single_label(path):
    with open(path, "r") as f:
        labels = json.load(f)
    points = torch.tensor(labels)
    mask = check_order(points[:, 1:])
    if mask.sum() > 0:
        points[mask] = points[mask][:, [0, 7, 8, 5, 6, 3, 4, 1, 2]]
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for idx, label in enumerate(points[mask]):
            point = label[1:9]
            for i in range(0, 8, 2):
                cv.putText(
                        img, str(i // 2 + 1), (int(point[i]), int(point[i + 1])),
                        cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, int(255 - idx * 80), idx * 80),
                        1
                        )
        cv.imshow("img", img)
        cv.waitKey(0)
    with open(path, "w") as f:
        json.dump(points.tolist(), f)


def correct_label(dataset_path):
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    for file in chain(os.listdir(train_path), os.listdir(valid_path)):
        if os.path.splitext(file)[1] == ".label":
            try:
                correct_single_label(os.path.join(train_path, file))
            except FileNotFoundError:
                correct_single_label(os.path.join(valid_path, file))


if __name__ == '__main__':
    correct_label("../../dataset/data")
    check_each_data("../../dataset/data", shuffle=True)
