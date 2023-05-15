import os

import torch
from MyYOLOX import YOLOX
from utils import *


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":
    root_path = ''
    dataset_root_path = ''
    last_epoch = 0

    epoches = 2000
    device = torch.device("cuda")

    net = YOLOX(num_class=24).to(device)

    optim = torch.optim.Adam(net.parameters(), 0.001, weight_decay=0.001)
    # optim = torch.optim.SGD(net.parameters(), lr=0.008, momentum=0.9)
    train_dataset, train_loader, test_dataset, test_loader = get_dataset_loader(
            os.path.join(dataset_root_path, "dataset/data"), max_label_num=32, batch_size=12,
            data_ratio=0.001, data_loder_workers=2, img_format=".jpg"
            )

    # lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, 5, min_lr=1e-8)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 80, 2)
    # lr_sch = torch.optim.lr_scheduler.StepLR(optim, 75, 0.5)
    # with torch.autograd.set_detect_anomaly(True):
    fit(
            epoches, train_loader, test_loader, net, optim, device, lr_sch,
            grad_max_norm=10.0, grad_accumulate_batches=1,
            save_path_folder=os.path.join(root_path, "fited_models"),
            save_model_epoches=5, test_model_batch=5,
            print_info_batches=20, detailed_loss_info=True, last_epoch=last_epoch
            )
