import sys
from typing import Union

import torch
import torchvision.transforms as transforms

import os
import cv2 as cv
import numpy as np
from math import floor
import json
from enum import Enum, auto

from MyYOLOX.boxes import IOULoss


class RandomErasing(torch.nn.Module):
    """
    Args:
        value_mode: the value that used to fill earsed region. it can be a float number or str "random", "mix".
                    In "random", the erasing region would be filled with 0~1 from 'standard normal' distribution.
                    In "mix", it will crop other img's non-label region to fill it.
    """

    def __init__(
            self, p=0.5, scale=(0.07, 0.1), ratio=(0.33, 3),
            value_mode: Union[int, str] = 0
            ):
        super(RandomErasing, self).__init__()
        self.p = p
        self.scale = (scale[1] - scale[0], scale[0])
        self.ratio = (ratio[1] - ratio[0], ratio[0])
        self.value_mode = value_mode

    def forward(self, x, label, nlabel):
        for lab in label[:nlabel]:
            position = lab[[1, 2, 5, 6]].tolist()
            if torch.rand(1).item() > self.p or (position is not None and (
                    position[2] - position[0] - 2 <= 0 or position[3] - position[1] - 2 <= 0)):
                continue
            ratio = np.random.random() * self.ratio[0] + self.ratio[1]
            scale = np.random.random() * self.scale[0] + self.ratio[1]
            # x shape is [H, W, C]
            area = (position[2] - position[0]) * (position[3] - position[1]) * scale
            w = min((area / ratio) ** 0.5, position[2] - position[0] - 2)
            h = min(w * ratio, position[3] - position[1] - 2)
            px = position[2] - position[0] - w
            py = position[3] - position[1] - h
            sx = np.random.random() * px + position[0]
            sy = np.random.random() * py + position[1]
            sx = floor(min(max(sx, 0), x.shape[1] - 1))
            sy = floor(min(max(sy, 0), x.shape[0] - 1))
            h = floor(min(max(h, 1), x.shape[0] - sy))
            w = floor(min(max(w, 1), x.shape[1] - sx))
            if self.value_mode == 'mix':
                import sys
                print("mix mode is under devel.", file=sys.stderr)
                pass
            elif self.value_mode == 'random':
                x[sy:(sy + h), sx:(sx + w), :] = np.random.randint(0, 255, (h, w, x.shape[2]), dtype=np.uint8)
            elif isinstance(self.value_mode, float):
                value = np.empty((h, w, x.shape[2]), dtype=np.uint8)
                value.fill(self.value_mode)
                x[sy:(sy + h), sx:(sx + w), :] = value
        return x, label, nlabel


def random_func_warpper(func, *args):

    def random_():
        return func(*args).item()

    return random_


class AffineTransform(torch.nn.Module):
    class RandomMode(Enum):
        GAUSS = auto
        UNIFORM = auto

    def __init__(self, angle=(-40, 40), p=(0, 0.3), random_mode=RandomMode.GAUSS):
        super(AffineTransform, self).__init__()
        assert (random_mode == self.RandomMode.GAUSS and isinstance(p, tuple) and len(p) == 2) \
               or (random_mode == self.RandomMode.UNIFORM and p is None), "invaild random mode and p-value"
        self.avg_angle = sum(angle) / len(angle)
        self.r_angle = angle[-1] - self.avg_angle
        if random_mode == self.RandomMode.GAUSS:
            self.random_generator = random_func_warpper(np.random.normal, p[0], p[1], 1)
        elif random_mode == self.RandomMode.UNIFORM:
            self.random_generator = random_func_warpper(np.random.uniform, -1, 1, 1)

    def forward(self, x, label, nlabel):
        x, label[:nlabel, 1:] = self.rotate_img(
                np.array(x), label[:nlabel, 1:].view(-1, 4, 2).double(), round(self.random_generator() * self.r_angle + self.avg_angle)
                )
        return x, label, nlabel

    @staticmethod
    def rotate_img(img, points, angle):

        width = img.shape[1]
        height = img.shape[0]

        rotated_matrix = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

        cos_ = abs(rotated_matrix[0, 0])
        sin_ = abs(rotated_matrix[0, 1])
        new_width = int(cos_ * width + sin_ * height)
        new_height = int(sin_ * width + cos_ * height)

        rotated_matrix[0, 2] += (new_width / 2 - width / 2)
        rotated_matrix[1, 2] += (new_height / 2 - height / 2)
        rotated_img = cv.warpAffine(img, rotated_matrix, (new_width, new_height), cv.INTER_LINEAR, 0)
        resize_img = cv.resize(rotated_img, (640, 480), None, None, None, cv.INTER_AREA)

        rotated_matrix = torch.from_numpy(rotated_matrix).double()
        # 对标签点进行透射变换
        for num, point in enumerate(points):
            points[num] = torch.mm(rotated_matrix[:2, :2], points[num].T).T + rotated_matrix[:, -1]
            points[num, :, 0] *= img.shape[1] / rotated_img.shape[1]  # width
            points[num, :, 1] *= img.shape[0] / rotated_img.shape[0]  # height
        return resize_img, points.view(-1, 8).float()


class Merge(torch.nn.Module):
    def __init__(self, dataset, img_origin_size, minum_label_size, p=0.35):
        super(Merge, self).__init__()
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.img_cut_size = tuple(x // 2 for x in img_origin_size)
        self.p = p
        self.threshold = minum_label_size

    def forward(self, x, label, nlabel):
        if np.random.random() < self.p and self.check_label(label, nlabel):
            others = []
            imgs = [cv.resize(x, self.img_cut_size, None, None, None, cv.INTER_AREA)]
            labels = [self.transform_label(label[:nlabel])]
            while imgs.__len__() < 4:
                item = np.random.randint(0, self.dataset_len - 1, 1).item()
                if item in others:
                    continue
                img_, label_, nlabel_ = self.dataset.pullitem(item)
                if not self.check_label(label_, nlabel_):
                    continue
                for transform in self.dataset.before_totensor:
                    if not isinstance(transform, Merge):
                        img_, label_, nlabel_ = transform(img_, label_, nlabel_)
                imgs.append(cv.resize(img_, self.img_cut_size, None, None, None, cv.INTER_AREA))
                labels.append(self.transform_label(label_[:nlabel_]))
                nlabel += nlabel_
            labels[1] = self.add_orrd(labels[1], "x")
            labels[2] = self.add_orrd(labels[2], "y")
            labels[3] = self.add_orrd(labels[3], "xy")
            # img size: [H, W, C]
            x = np.concatenate((np.concatenate(imgs[:2], axis=1), np.concatenate(imgs[2:], axis=1)), axis=0)
            label[:nlabel] = torch.concat(labels, dim=0)
        return x, label, nlabel

    @staticmethod
    def transform_label(label):
        label[:, 1:] /= 2
        return label

    def add_orrd(self, label, oper):
        if "x" in oper:
            label[:, [1, 3, 5, 7]] += self.img_cut_size[0]
        if "y" in oper:
            label[:, [2, 4, 6, 8]] += self.img_cut_size[1]
        return label

    def check_label(self, label, nlabel) -> bool:
        for lab in label[:nlabel, 1:]:
            if IOULoss._calc_area_fixnum(lab[None, ...].view(-1, 4, 2)).item() < self.threshold:
                return False
        return True


class MixUp(torch.nn.Module):
    def __init__(self):
        super(MixUp, self).__init__()

    def forward(self, x):
        pass


# 数据增强的类的创建
class ImgStrengthen(object):
    def __init__(self, folder_path, max_label, img_format='.jpg'):
        self.folder_path = folder_path
        self.img_format = img_format
        self.max_label = max_label
        self.data = []
        labels = {}
        pics = []
        for file in os.listdir(self.folder_path):
            if os.path.splitext(file)[1] == ".label":
                labels.setdefault(self.get_id(file), file)
            elif os.path.splitext(file)[1] in img_format:
                pics.append(file)

        np.random.shuffle(pics)
        for path in pics:
            with open(os.path.join(folder_path, labels[self.get_id(path)]), "r") as f:
                label = json.load(f)
            self.data.append((path, label, len(label)))
        self.before_totensor = [
                RandomErasing(scale=(0.06, 0.09), p=0.2, value_mode='random'),
                AffineTransform((-20, 20), (0, 0.2)),
                Merge(self, (640, 480), 140, 1),
                ]

        self.topil = transforms.ToPILImage()
        self.totensor = transforms.ToTensor()
        self.after_totensor = transforms.Compose(
                [
                        transforms.ColorJitter(
                                brightness=[0.3, 1.7], hue=[-0.035, 0.045],
                                contrast=[0.5, 1.7], saturation=[0.6, 1.9]
                                ),
                        # transforms.RandomApply([transforms.GaussianBlur(7, (0.1, 2.0))], p=0.4),
                        # 标准化(每个通道的均值,每个通道的标准差)
                        # transforms.Normalize(
                        #         mean=[0.110972, 0.133437, 0.136208],
                        #         std=[0.057981, 0.063314, 0.070612]
                        #         )
                        ]
                )

    def __len__(self):
        return len(self.data)

    def pullitem(self, index):
        img_path = self.data[index][0]
        nlabel = self.data[index][2]
        label = torch.zeros(self.max_label, 9)
        label[:nlabel] = torch.tensor(self.data[index][1])
        img = cv.imread(os.path.join(self.folder_path, img_path))
        return img, label, nlabel

    @staticmethod
    def get_id(file_name):
        return int(file_name.split(sep='_')[0][3:])

    def read_transform_show(self, draw_label=True):
        for data in self.data:
            img_path = data[0]
            raw_img = cv.imread(os.path.join(self.folder_path, img_path))
            img = raw_img.copy()
            label = torch.zeros(self.max_label, 9)
            nlabel = data[2]
            label[:nlabel] = torch.tensor(data[1])

            for transform in self.before_totensor:
                img, label, nlabel = transform(img, label, nlabel)
            img = self.totensor(img)
            img = self.after_totensor(img)

            img = np.array(self.topil(img))
            cv.imshow("raw img", raw_img)
            for point in label[:nlabel, 1:].view(-1, 4, 2):
                cv.polylines(img, [np.array(point, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
            cv.imshow("transformed img", img)
            cv.waitKey(0)

    def save_strengthen_imgs(self, save_folder, rounds, save_tensor=True):
        for idx in range(rounds):
            for data in self.data:
                img_path = data[0]
                img_info = os.path.splitext(img_path)
                img = cv.cvtColor(cv.imread(os.path.join(self.folder_path, img_path)), cv.COLOR_RGB2BGR)
                img = self.totensor_normalize(img)
                label = torch.zeros(self.max_label, 9)
                nlabel = data[2]
                label[:nlabel] = torch.tensor(data[1])

                for i in range(nlabel):
                    img = self.random_erasing(img, label[i][[1, 2, 5, 6]].tolist())

                if not save_tensor:
                    img = self.topil(img)
                    save_path = os.path.join(save_folder, img_info[0]) + '_' + str(idx) + img_info[1]
                    img.save(save_path)
                else:
                    save_path = os.path.join(save_folder, img_info[0]) + '_' + str(idx) + '.tsr'
                    with open(save_path, "wb") as f:
                        torch.save(img, f)


if __name__ == '__main__':
    ImgStrengthen(
            r"../../dataset/data/train", max_label=32
            ).read_transform_show()
