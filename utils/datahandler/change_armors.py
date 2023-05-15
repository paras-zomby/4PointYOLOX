import random
import cv2 as cv
import numpy as np
import torch.nn

# from utils.dataset import ObjDetDataset

labels = {
        "R1": 3,
        "B1": 4,
        "R3": 19,
        "B3": 18,
        "R4": 0,
        "B4": 1,
        "R5": 22,
        "B5": 21,
        }

coordinate = {
        3:  [[0, 45], [287, 33], [0, 115], [287, 114]],
        4:  [[0, 45], [346, 36], [0, 139], [346, 139]],
        19: [[0, 29], [90, 27], [0, 61], [90, 60]],
        18: [[0, 24], [81, 24], [0, 58], [81, 54]],
        0:  [[0, 22], [106, 21], [0, 63], [106, 62]],
        1:  [[0, 26], [116, 25], [0, 74], [116, 75]],
        22: [[0, 60], [289, 70], [2, 178], [291, 173]],
        21: [[0, 25], [123, 27], [0, 79], [123, 79]],
        }


class PerspectiveTransform(torch.nn.Module):
    def __init__(self, p=0.5, targets=("R3", "B3", "R4", "B4", "R5", "B5"), weight=None):
        super(PerspectiveTransform, self).__init__()
        self.p = p
        self.weight = weight
        self.red_armors = []
        self.blue_armors = []
        for target in targets:
            img = cv.imread("utils/datahandler/armors/" + target + ".png")
            if target[0] == "R":
                self.red_armors.append((img, labels[target]))
            elif target[0] == "B":
                self.blue_armors.append((img, labels[target]))

    def choice_armor(self, cls):
        if cls in (0, 16, 19, 22) and self.red_armors:  # red armors
            return random.choices(self.red_armors, self.weight)[0]
        elif cls in (1, 15, 18, 21) and self.blue_armors:  # blue armors
            return random.choices(self.blue_armors, self.weight)[0]
        else:  # not impl how to change, including `hero`, `outpost`, `sentry` and `base`
            return None, None

    def forward(self, x: np.ndarray, label, nlabel):
        for i in range(nlabel):
            if self.p > random.random():
                armor, label_cls = self.choice_armor(label[i][0].item())
                if armor is None or label_cls is None:
                    continue
                armor = self.combine_pics(armor, self.bounding_region(x, label[i][1:]))
                label[i][0] = label_cls
                matrix = cv.getPerspectiveTransform(
                        np.array(coordinate[label_cls], dtype=np.float32),
                        np.array(
                                ((label[i][1], label[i][2]), (label[i][7], label[i][8]), (label[i][3], label[i][4]), (label[i][5], label[i][6])),
                                dtype=np.float32
                                ),
                        )
                out = cv.warpPerspective(armor, matrix, (x.shape[1], x.shape[0]))
                # 去除仿射变换产生的奇怪黑边
                mask = cv.erode((out > 0).astype(np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))).astype(np.bool)
                x[mask] = out[mask]
                # cv.imshow("img", x)
                # cv.imshow("out", out)
                # cv.waitKey(0)
        return x, label, nlabel

    @staticmethod
    def combine_pics(src, style):
        src = src.astype(np.float32)
        src += (style.mean() - src.mean()) * 0.2
        for ch in range(style.shape[2]):
            coff = np.clip(style[..., ch].mean() / src[..., ch].mean(), 0.7, 1.3)
            offset = (style[..., ch].mean() - src[..., ch].mean()) * 0.35  # TODO: some times it maybe empty
            src[..., ch] = np.clip(255 * coff * (np.clip(src[..., ch] + offset, 0, 255) / 255) ** 1.5, 0, 255).astype(np.uint8)
        return src.astype(np.uint8)

    @staticmethod
    def bounding_region(pic, points):
        x = [int(points[i].item()) for i in range(len(points)) if i % 2 == 0]
        y = [int(points[i].item()) for i in range(len(points)) if i % 2 == 1]
        miny = min(y)
        maxy = max(y)
        adj = (maxy - miny) // 2
        return pic[miny - adj:maxy + adj, min(x):max(x), :]


if __name__ == '__main__':
    train_dataset = ObjDetDataset(
            "../../dataset/data/train", 32, need_nlabel=True,
            img_format=".jpg", data_ratio=1
            )
    pres = PerspectiveTransform(p=1)
    for i in range(len(train_dataset)):
        pres(*train_dataset.pullitem(i))
