# 数据集的生成
import json
import os
import random
import sys

import cv2 as cv
import imageio
import numpy as np
import pandas as pd


def traverse_folder(folders, init_path, target_suffix):
    # print(Father_Folder)
    lists = []
    for child in folders:
        if child[0] == ".":
            continue
        child_path = os.path.join(init_path, child)
        if os.path.isdir(child_path):
            child_path_folders = os.listdir(child_path)
            lists.extend(
                    traverse_folder(child_path_folders, child_path, target_suffix)
                    )
        elif os.path.splitext(child_path)[1] in target_suffix:
            lists.append(child_path)
    return lists


def calculate_means_std(filepath, w, h, img_format=".jpg"):
    pathDir = [x for x in os.listdir(filepath) if os.path.splitext(x)[1] in img_format]
    total = len(pathDir)
    R_channel = 0
    G_channel = 0
    B_channel = 0
    for filename in pathDir:
        img = imageio.imread(os.path.join(filepath, filename)) / 255.0
        R_channel += np.sum(img[:, :, 0])
        G_channel += np.sum(img[:, :, 1])
        B_channel += np.sum(img[:, :, 2])

    num = total * w * h
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for filename in pathDir:
        img = imageio.imread(os.path.join(filepath, filename)) / 255.0
        R_channel += np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel += np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel += np.sum((img[:, :, 2] - B_mean) ** 2)

    R_std = np.sqrt(R_channel / num)
    G_std = np.sqrt(G_channel / num)
    B_std = np.sqrt(B_channel / num)
    print(f'calc finish. total pic num = {total}')
    print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    print("R_std is %f, G_std is %f, B_std is %f" % (R_std, G_std, B_std))
    return (R_mean, G_mean, B_mean), (R_std, G_std, B_std)


# 读取图片和标签
id_cnt_train = {i: 0 for i in range(24)}
id_cnt_val = {j: 0 for j in range(24)}

''' 
红4:     13   0
蓝4:     4    1 
灰4:     22   2 

红1:     10   3
蓝1:     1    4
灰1:     19   5

红哨兵:   9    6   
蓝哨兵:   0    7
灰哨兵:   18   8

红前哨:   15   9
蓝前哨:   6    10
灰前哨:   24   11

# 老数据集中，小基地装甲板被标成了大装甲板的上交ID，
# 对于新数据集大装甲板的上交ID却用于标记真实的大装甲板。
红大基地:   17    12
蓝大基地:   8     13
灰大基地:   26    14
  
蓝2:       2     15
红2:       11    16
灰2:       20    17
  
蓝3:       3     18
红3:       12    19
灰3:       21    20
  
蓝5:       5     21
红5:       14    22
灰5:       23    23
'''

class_corrcet = [7, 4, 15, 18, 1, 21, 10, None, 13, 6, 3, 16, 19, 0, 22, 9,
                 None, 12, 8, 5, 17, 20, 2, 23, 11, None, 14]


def collect_data(img_path, label_path, save_folder, our_data=False):
    os.system("rm -rf " + os.path.join(save_folder, "train/*"))
    os.system("rm -rf " + os.path.join(save_folder, "valid/*"))
    if isinstance(img_path, list) and isinstance(label_path, list) and len(img_path) == len(
            label_path
            ):
        imgs = []
        labels = []
        for idx in range(len(img_path)):
            father_img_folder = os.listdir(img_path[idx])
            imgs.extend(
                    traverse_folder(
                            father_img_folder, img_path[idx], ".jpg .png"
                            )
                    )
            father_label_folder = os.listdir(label_path[idx])
            labels.extend(traverse_folder(father_label_folder, label_path[idx], ".txt"))
    else:
        father_img_folder = os.listdir(img_path)
        imgs = traverse_folder(
                father_img_folder, img_path, ".jpg .png"
                )

        father_label_folder = os.listdir(label_path)
        labels = traverse_folder(father_label_folder, label_path, ".txt")
    imgs.sort()
    labels.sort()

    # 确定图像的大小
    img_size = (640, 480)

    # #列出所有的图片和标签文件
    # imgs = os.listdir(Img_Folder)
    # Labeltxts = os.listdir(Label_Folder)

    print("total images:", len(imgs))
    print("total labels:", len(labels))
    # time.sleep(10)

    # 定义存储的标签
    img_index = 0
    correct_index = 0  # 用于确保图片和标签可以一一对应
    id_cnt = None

    max_label_len = 0

    for Num, Labeltxt in enumerate(labels):
        # print(Labeltxt)
        # print(imgs[Num])
        # print("_____________________")

        while (Labeltxt.split("/")[-1].split('.')[0] !=
               imgs[correct_index].split("/")[-1].split(".")[0]):
            correct_index += 1

        # Full_LabelPath = os.path.join(label_path,Labeltxt)
        txt = pd.read_table(Labeltxt, sep=' ', header=None)

        # to see max label num per img
        if max_label_len < txt.shape[0]:
            max_label_len = txt.shape[0]
            print(f"max label num = {max_label_len}")

        # 读取图片
        # Full_ImgPath = os.path.join(img_path,imgs[Num])
        Img = cv.imread(imgs[correct_index])
        # print(imgs[Correct_Index].split(".")[0].split("-")[3:6])
        Img = cv.resize(Img, img_size)
        Img_Labels = []
        for i in range(txt.shape[0]):
            Label = []
            try:
                # 获取装甲板类型(需要自己重制)
                id_cnt_raw = txt[0][i]
                # ID纠错
                id_cnt = class_corrcet[id_cnt_raw]
                if id_cnt is None:  # 如果id_cnt为-1，说明装甲板不能用。
                    raise IndexError
                if our_data and 12 <= id_cnt <= 14:  # 如果是自己拍得数据集的基地装甲板弃用
                    continue
                Label.append(int(id_cnt))

            except IndexError:
                print(
                        f"not in expect Label, label is {Labeltxt}, raw label = {id_cnt_raw}, "
                        f"label = {id_cnt}",
                        file=sys.stderr
                        )
                continue

            for j in range(1, txt.shape[1]):
                # 读取四个角点，左上，左下，右下，右上
                Label.append(float(txt[j][i] * img_size[not (j % 2)]))
            Img_Labels.append(Label)
        if not Img_Labels:
            continue
        file_name = "Img{}_{}".format(img_index, id_cnt)
        # 按序号保存
        if random.random() < 0.8:
            path = os.path.join(save_folder, "train", file_name)
            for label in Img_Labels:
                id_cnt_train[label[0]] += 1
        else:
            path = os.path.join(save_folder, "valid", file_name)
            for label in Img_Labels:
                id_cnt_val[label[0]] += 1
        cv.imwrite(path + ".jpg", Img)
        with open(path + ".label", "w", encoding="utf-8") as f:
            json.dump(Img_Labels, f, ensure_ascii=True, allow_nan=False)

        img_index += 1

    print("train ID nums: ", id_cnt_train, "\nvalid ID nums: ", id_cnt_val)


if __name__ == '__main__':
    collect_data(
            ["../../dataset/sj_data", "../../dataset/self_data/Img", "../../dataset/fenqu_data", "../../dataset/heu_data/Img"],
            ["../../dataset/sj_data", "../../dataset/self_data/Label", "../../dataset/fenqu_data", "../../dataset/heu_data/Label"],
            "../../dataset/data/",
            our_data=False
            )
    print("remeber to run 'check data' later!!!", file=sys.stderr)
