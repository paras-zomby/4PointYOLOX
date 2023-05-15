# 数据集的读取与处理
# 相关模块的导入

import json
import math
import os

import numpy as np
import torch
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from .datahandler.data_augs import RandomErasing, AffineTransform, Merge
from .datahandler.change_armors import PerspectiveTransform

# 测试数据集原图片的转换
validimg_transform = transforms.Compose(
        [
                transforms.ToTensor(),
                transforms.Normalize(
                        mean=[0.110972, 0.133437, 0.136208],
                        std=[0.057981, 0.063314, 0.070612]
                        ),
                ]
        )


# 数据导入的类的创建
class ObjDetDataset(Dataset):
    def __init__(
            self, dataset_folder_path, max_label,
            img_format=".jpg",
            data_ratio=1,
            need_nlabel=False,
            valid_dataset=False
            ):
        self.dataset_folder_path = dataset_folder_path
        self.max_label = max_label
        self.need_nlabel = need_nlabel
        self.is_tsr = img_format == ".tsr"
        self.data_augs = not valid_dataset
        self.data = []
        labels = {}
        pics = []
        if not self.is_tsr:
            for file in os.listdir(self.dataset_folder_path):
                if os.path.splitext(file)[1] == ".label":
                    labels.setdefault(self.get_id(file), file)
                elif os.path.splitext(file)[1] in img_format:
                    pics.append(file)
            pics = np.random.choice(pics, round(len(pics) * data_ratio), False).tolist()
            for path in pics:
                with open(os.path.join(dataset_folder_path, labels[self.get_id(path)]), "r") as f:
                    label = json.load(f)
                self.data.append((path, label, len(label)))
        else:
            for file in os.listdir(self.dataset_folder_path):
                if os.path.splitext(file)[1] == ".tsr":
                    pics.append(file)
            self.data = np.random.choice(pics, round(len(pics) * data_ratio), False).tolist()

        self.before_totensor = [
                PerspectiveTransform(p=0.25),
                RandomErasing(scale=(0.06, 0.09), p=0.25, value_mode='random'),
                AffineTransform((-20, 20), (0, 0.2)),
                Merge(self, (640, 480), 120, 0.35),
                ]
        self.totensor = transforms.ToTensor()
        self.after_totensor = transforms.Compose(
                [
                        transforms.ColorJitter(
                                brightness=[0.15, 2], hue=[-0.035, 0.045],
                                contrast=[0.37, 1.6], saturation=[0.6, 1.9]
                                ),
                        # transforms.RandomApply([transforms.GaussianBlur(7, (0.1, 2.0))], p=0.4),
                        # 标准化(每个通道的均值,每个通道的标准差)
                        ]
                )
        self.normalize = transforms.Normalize(
                                mean=[0.110972, 0.133437, 0.136208],
                                std=[0.057981, 0.063314, 0.070612]
                                )

    @staticmethod
    def get_id(file_name):
        return int(file_name.split(sep='_')[0][3:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # 样本文件夹路径
        if self.is_tsr:
            with open(os.path.join(self.dataset_folder_path, self.data[item]), "rb") as f:
                data = torch.load(f)
            return (data["img"], data["label"]) if not self.need_nlabel else (data["img"], data["label"], data["nlabel"])
        else:
            img, label, nlabel = self.pullitem(item)

            if self.data_augs:
                for transform in self.before_totensor:
                    img, label, nlabel = transform(img, label, nlabel)
            img = self.totensor(img)
            if self.data_augs:
                img = self.after_totensor(img)
            img = self.normalize(img)
            return (img, label) if not self.need_nlabel else (img, label, nlabel)

    def pullitem(self, index):
        img_path = self.data[index][0]
        nlabel = self.data[index][2]
        label = torch.zeros(self.max_label, 9)
        label[:nlabel] = torch.tensor(self.data[index][1])
        img = cv.imread(os.path.join(self.dataset_folder_path, img_path))
        return img, label, nlabel


# 数据集的导入以及处理的函数
def get_dataset_loader(
        folder_path, max_label_num, batch_size=1, data_loder_workers=0, need_label_num=True,
        img_format=".jpg",
        data_ratio=1,
        ):
    # 训练集文件路径的合成
    train_folder_path = os.path.join(folder_path, 'train')
    # 获得经过转换后的训练数据集
    train_dataset = ObjDetDataset(
            train_folder_path, max_label_num, need_nlabel=need_label_num,
            img_format=img_format, data_ratio=data_ratio
            )
    # 将得到的训练数据集进行数据导入(数据集,每一次batch的数量,是否打乱顺序,当数据不够最后一次batch是否变小,线程数,设置为锁页内存可以让速度更快)
    train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=data_loder_workers, pin_memory=True
            )
    # 对标签集进行处理,与训练集处理相同
    val_folder_path = os.path.join(folder_path, 'valid')
    val_dataset = ObjDetDataset(
            val_folder_path, max_label_num, need_nlabel=need_label_num,
            img_format=img_format, data_ratio=data_ratio, valid_dataset=True
            )
    val_data_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, drop_last=False,
            num_workers=0,
            pin_memory=True
            )
    print(
            f"batch size = {batch_size}, total train sample = {len(train_dataset)}, "
            f"train batches = {math.ceil(len(train_dataset) / batch_size)}, "
            f"total validation sample = {len(val_dataset)}"
            )
    return train_dataset, train_data_loader, val_dataset, val_data_loader


# 数组归一化到0~1
def normalize(array):
    min_elem = np.min(array)
    max_elem = np.max(array)
    if max_elem == min_elem:
        return array
    else:
        return (array - min_elem) / (max_elem - min_elem)


# 主函数
if __name__ == '__main__':

    FolderPath = '../dataset'
    TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = get_dataset_loader(
            FolderPath, 4, batch_size=40
            )
    for epoch in range(1):
        for i, (Img, Label, SampleName) in enumerate(TrainDataLoader):
            print(Img.shape)
            print(Label)
