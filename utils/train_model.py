import os
import sys

import torch

import torch.utils.tensorboard as tensorboard

from .others import Timer
from .test_model import test
from .checkpoint import save_ckpt, load_ckpt


def fit(
        epoches, train_data_loader, test_data_loader,
        model, optimizer=None, device=None,
        lr_scheduler=None,
        grad_max_norm=None,
        grad_accumulate_batches=1,
        save_path_folder=None,
        save_model_epoches=None,
        print_info_batches=100,
        test_model_batch=None,
        test_pred_confidence=0.6,
        detailed_loss_info=False,
        use_tensorboard=True,
        last_epoch=0
        ):
    if not device:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters())

    if last_epoch != 0:
        print(f"last epoch is {last_epoch}, using check point file...")
        model, optimizer, last_epoch_, (
                fit_loss, test_cls_acc, test_prec, test_recall) = load_ckpt(
                model, optimizer, os.path.join(
                        save_path_folder, "checkpoints", f'epoch_{last_epoch:04d}.ckpt'
                        ), device
                )
        assert last_epoch_ == last_epoch, "last epoch must be same as its saved in ckpt file."
    else:
        fit_loss, test_cls_acc, test_prec, test_recall = [], [], [], []

    save_flag = save_path_folder and save_model_epoches
    if not save_flag and (save_path_folder or save_model_epoches or use_tensorboard):
        print(
                'you should give both "save_path_folder" and "save_model_epoches"'
                'if you want to save the training model or use tensorboard for visualization.',
                file=sys.stderr
                )
    elif save_flag:
        f = open(os.path.join(save_path_folder, "train.log"), "a+", encoding='utf-8')
        if use_tensorboard:
            writer = tensorboard.SummaryWriter(
                    os.path.join(save_path_folder, "tensorboard_logs"), flush_secs=300
                    )

    timer = Timer()
    model.train()

    with Timer("fit the model"):
        for epoch in range(1 + last_epoch, epoches + 1):
            loss = train(
                    epoch, train_data_loader, model, optimizer, device, grad_max_norm,
                    grad_accumulate_batches, print_info_batches, detailed_loss_info, timer
                    )
            fit_loss.append(loss)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if epoch % test_model_batch == 0:
                res, res_dict = test(
                        epoch, test_data_loader, model, test_pred_confidence, device, 0.65
                        )
                if save_flag:
                    f.write(res)
                if res_dict is not None:
                    test_cls_acc.append(res_dict["cls acc"])
                    test_prec.append(res_dict['Precision'])
                    test_recall.append(res_dict['Recall'])
                    if use_tensorboard:
                        writer.add_scalars("test result", res_dict, epoch)

            if save_flag:
                f.write(
                        (
                                f'epoch = {epoch} detailed losses: '
                                f'iou = {loss["iou_loss"]} obj = {loss["obj_loss"]} '
                                f'cls = {loss["cls_loss"]} reg = {loss["reg_loss"]} \n'
                        ) if detailed_loss_info else f"epoch = {epoch} loss = {loss}"
                        )
                save_ckpt(
                        epoch, model, optimizer,
                        (fit_loss, test_cls_acc, test_prec, test_recall),
                        os.path.join(save_path_folder, "checkpoints")
                        )
                if use_tensorboard:
                    writer.add_scalars("train losses", loss, epoch)

            if save_flag and epoch % save_model_epoches == 0:
                torch.save(
                        model.state_dict(),
                        os.path.join(save_path_folder, 'model_{:04d}.pt'.format(epoch))
                        )
                f.flush()
    f.close()
    writer.close()
    return fit_loss, test_cls_acc, test_prec, test_recall


def train(
        epoch, loader, model, optimzer, device,
        grad_max_num=None,
        grad_accumulate_batches=1,
        print_info_batches=100,
        detailed_loss_info=False,
        timer=None
        ):
    run_loss = {
            "total_loss": 0,
            "iou_loss":   0,
            "obj_loss":   0,
            "cls_loss":   0,
            "reg_loss":   0,
            "num_fg":     0
            }
    last_loss = 0
    total_pic = 0
    total_gt = 0
    if not timer:
        timer = Timer()
    optimzer.zero_grad()
    timer.start()
    for batch_idx, (X, Y, N) in enumerate(loader):
        batch_idx += 1
        X, Y = X.to(device), Y.to(device)

        loss = model(X, (Y, N))

        loss['total_loss'].backward()
        if grad_max_num is not None:
            torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_max_num
                    )

        run_loss['total_loss'] += loss['total_loss'].item()
        run_loss['iou_loss'] += loss['iou_loss']
        run_loss['obj_loss'] += loss['obj_loss']
        run_loss['cls_loss'] += loss['cls_loss']
        run_loss['reg_loss'] += loss['reg_loss']
        run_loss['num_fg'] += loss['num_fg']
        total_gt += N.sum().item()
        total_pic += N.shape[0]

        # 使用max函数获取指定维度上的最大值的值与索引（下标），返回值第一个是值，第二个是索引。
        # pred_y = Y_pred.detach().argmax(dim=1)
        # correct += (pred_y.detach() == Y.detach()).sum().item()
        if not batch_idx % grad_accumulate_batches:
            optimzer.step()
            optimzer.zero_grad(set_to_none=True)
        if not batch_idx % print_info_batches:
            print(
                    f'this batch = {batch_idx}, sum loss = {round(run_loss["total_loss"], 4)},'
                    f' these {print_info_batches} batches sum loss ='
                    f' {round(run_loss["total_loss"] - last_loss, 4)}',
                    )
            last_loss = run_loss["total_loss"]
    print(
            f'Train {epoch}:total pic = {total_pic} total gt = {total_gt}, '
            f'loss = {round(run_loss["total_loss"], 3)}, '
            f'avg fg/gt = {round(run_loss["num_fg"] / total_pic, 3)} '
            f'using time = {timer.stop(False)}s'
            )
    if detailed_loss_info:
        print(
                f'detailed losses: '
                f'iou = {run_loss["iou_loss"]} obj = {run_loss["obj_loss"]} '
                f'cls = {run_loss["cls_loss"]} reg = {run_loss["reg_loss"]} '
                )
        return run_loss
    else:
        return run_loss['total_loss']
