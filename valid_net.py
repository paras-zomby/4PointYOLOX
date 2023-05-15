import torch
from MyYOLOX import YOLOX
from utils import *

if __name__ == "__main__":
    device = torch.device("cuda")

    net = YOLOX(num_class=24)
    net.to(device)

    train_dataset, train_loader, test_dataset, test_loader = get_dataset_loader(
            "dataset/data", max_label_num=8, batch_size=12,
            data_ratio=0.005, data_loder_workers=0
            )

    net.load_state_dict(torch.load("pick_up_models/new_v2.3_210.pt", map_location=device))

    # test(0, test_loader, net, 0.75, device, 0.5)
    mAP = calc_mAP(net, test_loader, device, 0.7)
    print(f"mAP = {mAP}")
