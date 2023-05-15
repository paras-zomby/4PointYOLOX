#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boxes import IOULoss
from .netblocks import BaseConv, DWConv
from .losses import BinaryFocalLoss, FocalLossWithSoftmax


class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes=24,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            reg_weight=0.1,
            iou_weight=5.0,
            cls_weight=3.0,
            act="silu",
            depthwise=False,
            ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): <MY>whether apply 'Depthwise Separable Convolution' in conv
            branch.
            Defalut value: False.
        """
        super().__init__()

        self.iou_weight = iou_weight
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight

        self.hw = None
        self.n_anchors = 1
        self.num_classes = num_classes
        # *把5个预测改成了9个预测目标，包括obj和4个坐标点，并把其转化为了成员变量
        self.n_channel = 1 + 8 + self.num_classes

        # !if using decode, the output will be real position in img but not
        # !the relatively position in the grids.
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                    BaseConv(
                            in_channels=int(in_channels[i] * width),
                            out_channels=int(256 * width),
                            ksize=1,
                            stride=1,
                            act=act,
                            )
                    )
            self.cls_convs.append(
                    nn.Sequential(
                            *[
                                    Conv(
                                            in_channels=int(256 * width),
                                            out_channels=int(256 * width),
                                            ksize=3,
                                            stride=1,
                                            act=act,
                                            ),
                                    Conv(
                                            in_channels=int(256 * width),
                                            out_channels=int(256 * width),
                                            ksize=3,
                                            stride=1,
                                            act=act,
                                            ),
                                    ]
                            )
                    )
            self.reg_convs.append(
                    nn.Sequential(
                            *[
                                    Conv(
                                            in_channels=int(256 * width),
                                            out_channels=int(256 * width),
                                            ksize=3,
                                            stride=1,
                                            act=act,
                                            ),
                                    Conv(
                                            in_channels=int(256 * width),
                                            out_channels=int(256 * width),
                                            ksize=3,
                                            stride=1,
                                            act=act,
                                            ),
                                    ]
                            )
                    )
            # !下面都是预测用的卷积核了
            self.cls_preds.append(
                    nn.Conv2d(
                            in_channels=int(256 * width),
                            out_channels=self.n_anchors * self.num_classes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            )
                    )
            self.reg_preds.append(
                    nn.Conv2d(
                            in_channels=int(256 * width),
                            out_channels=8,  # * <MY>改成了四个点的xy坐标
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            )
                    )
            self.obj_preds.append(
                    nn.Conv2d(
                            in_channels=int(256 * width),
                            out_channels=self.n_anchors * 1,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            )
                    )

        self.use_iou_loss = False
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.focalloss = BinaryFocalLoss(0.25, 2, "none", with_logist=True)
        self.iou_loss = IOULoss(reduction="none")
        self.reg_loss = nn.L1Loss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)  # ![0.,0.,0.]

    def initialize_biases(self, prior_prob=1e-2):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, nlabel=None):
        outputs = []
        x_shifts = []  # !four points have the same shift
        y_shifts = []
        expanded_strides = []
        xin_type = xin[0].type()

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)
                ):
            # !xin shape are 74/8  144/16  288/32
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                # !output[batch, reg:8+obj:1+cls:24, H, W]
                output = torch.cat([reg_output, obj_output, cls_output], dim=1)
                # !output[batch, anchors*H*W, n_channels], grid[1, anchors*height*width, 8]
                output, grid = self.get_output_and_grid(
                        output, k, stride_this_level, xin_type
                        )
                # TODO: consider move this to __init__() func in final edition.
                # !grid's last dimension have 8 elem include four points.
                # !but four points have the same shift.
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                # !expanded_stride shape[1, anchors*H*W], vals = stride
                expanded_strides.append(
                        torch.zeros(1, grid.shape[1])  # !grid.shape[1] = H*W*anchors
                            .fill_(stride_this_level)  # !make each elem equals stride
                            .type(xin_type)
                        )
            else:
                # !output[batch, reg:8+obj:1+cls:24, H, W]
                output = torch.cat(
                        [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], dim=1
                        )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                    x_shifts,
                    y_shifts,
                    expanded_strides,
                    labels,
                    nlabel,
                    torch.cat(outputs, 1),
                    dtype=xin[0].dtype,
                    )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # ! [batch, n_anchors_all, reg:8+obj:1+cls:24]
            outputs = torch.cat(
                    [x.flatten(start_dim=2) for x in outputs], dim=2
                    ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin_type)
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        """change the pred_reg in output to the whole img scale, and get the grid"""
        # !grid default shape = [0.]
        grid = self.grids[k]
        batch_size = output.shape[0]
        hsize, wsize = output.shape[-2:]

        # TODO: consider move it to __init__() func.
        # !if grid's height and width is not equals to output's
        if grid.shape[2:4] != output.shape[2:4]:
            # !xv and yv both have two dimensions, size is (h, w).
            # !xv's elems only increase horizontally but yv only vertically.
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # !grid's elem is a (x, y)*4 point. the other dimensions' shape is the same as
            # xv(or
            # yv).
            grid = torch.stack((xv, yv) * 4, dim=2).view(
                    1, self.n_anchors, hsize, wsize, 8
                    ).type(dtype)
            # !make grids[k] shape [1, anchors, height, width, 8]
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, self.n_channel, hsize, wsize)
        # !make output shape[batch, anchors*H*W, n_channels](channels = reg:8+obj:1+cls:24)
        output = output.permute(0, 1, 3, 4, 2).reshape(
                batch_size, self.n_anchors * hsize * wsize, self.n_channel
                )
        # !make grid shape [1, anchors*height*width, 2*4]
        grid = grid.view(1, -1, 8)
        # !make x,y value transformed from each grid to the whole img
        output[..., :8] = (output[..., :8] + grid) * stride
        # output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        # *让xy变成了8个数值对应四个点的坐标而不是原文的一个左上角点，并且相应的改了grid和output的计算公式。
        # *把原文中的w，h计算公式删除了，因为新的模型不需要计算bbox的宽和高。
        return output, grid

    def decode_outputs(self, outputs, dtype):
        # !this func transforms output reg_points into the whole img range.
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            # !xv and yv both have two dimentions, size is (h, w).
            # !xv's elems only increase horizontally but yv only vertically.
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # !make grid shape [1, anchor*height*width, 2*4]
            grid = torch.stack((xv, yv) * 4, 2).view(1, -1, 8)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        # !make strides shape [len(self.strides), anchor*height*width, 1] and value = stride
        strides = torch.cat(strides, dim=1).type(dtype)

        # *just same as above to change output.
        outputs[..., :8] = (outputs[..., :8] + grids) * strides
        return outputs

    def get_loss_single(
            self, outputs, labels, num_gt, bbox_preds, cls_preds, obj_preds,
            constants
            ):
        (total_num_anchors,
         expanded_strides,
         x_shifts,
         y_shifts,
         dtype) = constants
        num_gt = int(num_gt)
        if num_gt == 0:
            # !new_zeros func: return a tensor with the given size and zero-filled
            # !which have the same type&device as the old one.
            cls_target = outputs.new_zeros((0, self.num_classes))
            reg_target = outputs.new_zeros((0, 8))  # *改成了8个点的reg回归
            obj_target = outputs.new_zeros((total_num_anchors, 1))
            fg_mask = outputs.new_zeros(total_num_anchors).bool()
            num_fg_img = 0
        else:
            # !labels format: [batch, max_label_num, single_label]
            # !each single label should be ?absolute value?
            # !max_label_num = num_gt + fill_val
            # !single_label = cls:1 + bbox:8
            # !bbox points: [lt, lb, rb, rt]
            gt_bboxes_per_image = labels[:num_gt, 1:9]  # *把bbox的范围改成了1到9
            gt_classes = labels[:num_gt, 0]  # !gtclass shape [num_gt]
            # !bboxes_preds_per_image shape is [total_num_anchors, 8]
            bboxes_preds_per_image = bbox_preds

            # try:
            (gt_matched_classes,
             # !matched gt_classes based on anchor order, shape[num_fg]
             fg_mask,  # !shape is [total_num_anchors]
             pred_ious_this_matching,  # !anchors iou with gt, shape [num_fg]
             matched_gt_inds,  # !anchors' matched gts' indces, shape [num_fg]
             num_fg_img,  # !num of anchors that have gt to matched per img.
             ) = self.get_assignments_single(
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_preds,
                    obj_preds,
                    )

            # !cls targrt mulitply obj because it should be zero when obj is zero.
            # !shape is [num_fg, num_classes]
            cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                    ) * pred_ious_this_matching.unsqueeze(-1)
            # !shape is [total_num_anchors, 1]
            obj_target = fg_mask.unsqueeze(-1)
            # !shape is [num_fg, 8]
            reg_target = gt_bboxes_per_image[matched_gt_inds]
        return num_gt, num_fg_img, cls_target, reg_target.to(dtype), obj_target.to(dtype), fg_mask

    def get_losses(
            self,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            nlabel,
            outputs,
            dtype
            ):

        # *change the bbox_preds and realtive obj&cls preds.
        bbox_preds = outputs[:, :, :8]  # ![batch, total_num_anchors, 8]
        # ![batch, total_num_anchors, 1]->[batch, total_num_anchors]
        obj_preds = outputs[:, :, 8].unsqueeze(-1)
        cls_preds = outputs[:, :, 9:]  # ![batch, total_num_anchors, n_cls]

        # calculate targets
        # nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]  # !total = all anchors in three feature maps
        x_shifts = torch.cat(x_shifts, 1)  # [1, total_num_anchors]
        y_shifts = torch.cat(y_shifts, 1)  # [1, total_num_anchors]
        expanded_strides = torch.cat(expanded_strides, 1)  # [1, total_num_anchors]

        batch_size = outputs.shape[0]
        results = map(
                self.get_loss_single, outputs, labels, nlabel, bbox_preds, cls_preds, obj_preds,
                ((total_num_anchors, expanded_strides, x_shifts, y_shifts, dtype),) * batch_size,
                )
        num_gt_imgs, num_fg_imgs, cls_targets, reg_targets, obj_targets, fg_masks = tuple(zip(*results))

        num_gts = sum(num_gt_imgs)
        num_fg = sum(num_fg_imgs)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        # cls_targets = []
        # reg_targets = []
        # obj_targets = []
        # fg_masks = []
        #
        # # !total fg and gt used to calc avg loss.
        # num_fg = 0.0  # !this batch's total matched anchors num.
        # num_gts = 0.0  # !this batch's total gts num.
        #
        # for batch_idx in range(outputs.shape[0]):
        #     num_gt = int(nlabel[batch_idx])
        #     num_gts += num_gt
        #     if num_gt == 0:
        #         # !new_zeros func: return a tensor with the given size and zero-filled
        #         # !which have the same type&device as the old one.
        #         cls_target = outputs.new_zeros((0, self.num_classes))
        #         reg_target = outputs.new_zeros((0, 8))  # *改成了8个点的reg回归
        #         obj_target = outputs.new_zeros((total_num_anchors, 1))
        #         fg_mask = outputs.new_zeros(total_num_anchors).bool()
        #     else:
        #         # !labels format: [batch, max_label_num, single_label]
        #         # !each single label should be ?absolute value?
        #         # !max_label_num = num_gt + fill_val
        #         # !single_label = cls:1 + bbox:8
        #         # !bbox points: [lt, lb, rb, rt]
        #         gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:9]  # *把bbox的范围改成了1到9
        #         gt_classes = labels[batch_idx, :num_gt, 0]  # !gtclass shape [num_gt]
        #         # !bboxes_preds_per_image shape is [total_num_anchors, 8]
        #         bboxes_preds_per_image = bbox_preds[batch_idx]
        #
        #         # try:
        #         (gt_matched_classes,
        #          # !matched gt_classes based on anchor order, shape[num_fg]
        #          fg_mask,  # !shape is [total_num_anchors]
        #          pred_ious_this_matching,  # !anchors iou with gt, shape [num_fg]
        #          matched_gt_inds,  # !anchors' matched gts' indces, shape [num_fg]
        #          num_fg_img,  # !num of anchors that have gt to matched per img.
        #          ) = self.get_assignments(
        #                 batch_idx,
        #                 num_gt,
        #                 total_num_anchors,
        #                 gt_bboxes_per_image,
        #                 gt_classes,
        #                 bboxes_preds_per_image,
        #                 expanded_strides,
        #                 x_shifts,
        #                 y_shifts,
        #                 cls_preds,
        #                 obj_preds,
        #                 )
        #         torch.cuda.empty_cache()
        #         num_fg += num_fg_img
        #
        #         # !cls targrt mulitply obj because it should be zero when obj is zero.
        #         # !shape is [num_fg, num_classes]
        #         cls_target = F.one_hot(
        #                 gt_matched_classes.to(torch.int64), self.num_classes
        #                 ) * pred_ious_this_matching.unsqueeze(-1)
        #         # !shape is [total_num_anchors, 1]
        #         obj_target = fg_mask.unsqueeze(-1)
        #         # !shape is [num_fg, 8]
        #         reg_target = gt_bboxes_per_image[matched_gt_inds]
        #
        #     cls_targets.append(cls_target)
        #     reg_targets.append(reg_target.to(dtype))
        #     obj_targets.append(obj_target.to(dtype))
        #     fg_masks.append(fg_mask)
        #
        # cls_targets = torch.cat(cls_targets, 0)
        # reg_targets = torch.cat(reg_targets, 0)
        # obj_targets = torch.cat(obj_targets, 0)
        # fg_masks = torch.cat(fg_masks, 0)

        # !to avoid divide zero, num_fg must be at least one.
        num_fg = max(num_fg, 1)
        # *把传入4个点改成了传入8个点，这样才能计算IOU。
        if self.use_iou_loss:
            loss_iou = (self.iou_loss(
                    bbox_preds.view(-1, 8)[fg_masks], reg_targets
                    )).sum() / num_fg
        else:
            loss_iou = torch.tensor(0.0)
        # *增加了regloss， 计算四个点和标注之间的MSEloss。
        loss_reg_ = self.reg_loss(
                bbox_preds.view(-1, 8)[fg_masks], reg_targets
                )
        # TODO: 改成QFocal Loss
        loss_obj_ = self.focalloss(obj_preds.view(-1, 1), obj_targets)
        loss_cls_ = self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                )

        loss_reg = loss_reg_.sum() / num_fg
        loss_obj = loss_obj_.sum() / num_fg
        loss_cls = loss_cls_.sum() / num_fg

        loss = self.iou_weight * loss_iou + loss_obj + self.cls_weight * loss_cls + \
               self.reg_weight * loss_reg

        return (
                loss,
                loss_iou * self.iou_weight,
                loss_obj,
                loss_cls * self.cls_weight,
                loss_reg * self.reg_weight,
                num_fg / max(num_gts, 1),  # !calculate average matched anchor per gt.
                )

    @torch.no_grad()
    def get_assignments_single(
            self,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,  # !shape is [num_gt, 8]
            gt_classes,  # !shape is [num_gt]
            bboxes_preds_per_image,  # !shape is [total_num_anchors, 8]
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            obj_preds,
            mode="gpu", ):
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # !fg_mask: show an anchor if in at least one box or center.
        # !         single dim. shape is [total_num_anchors], dtype is bool.
        # !is_in_boxes_and_center:  show an anchor if both in an ground truth's box and center,
        # !                         but only has anchors which at least in one box or center.
        # !                         shape is [num_gt, (is_in_boxes_anchor > 0).sum()]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                gt_bboxes_per_image,
                expanded_strides,
                x_shifts,
                y_shifts,
                total_num_anchors,
                num_gt,
                )

        # !remove those anchors that not in any gt_bbox and gt_centers
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        # !shape is [num_in_boxes_anchor, 24]
        cls_preds_ = cls_preds[fg_mask]
        # !shape is [num_in_boxes_anchor, 1]
        obj_preds_ = obj_preds[fg_mask]
        # !the number of anchors that we not dropped.
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # !shape is [num_gt, total_num_anchors], dtype = float
        pair_wise_ious = IOULoss.get_IOU(
                bboxes_preds_per_image[None, ...].repeat(gt_bboxes_per_image.shape[0], 1, 1),
                gt_bboxes_per_image[:, None, :].repeat(1, bboxes_preds_per_image.shape[0], 1)
                )

        # !this is used when debugging because there may be negative iou value.
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        pair_wise_regs_loss = YOLOXHead.bbox_l1_loss(
                gt_bboxes_per_image,
                bboxes_preds_per_image
                )

        # !shape is [num_gt, num_in_boxes_anchor, num_classes]
        # !classes is encoded as one-hot vector.
        gt_cls_per_image = (
                F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                    .float()
                    .unsqueeze(1)
                    .repeat(1, num_in_boxes_anchor, 1)
        )

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        # ! autocast-enabled allows tensors transform there dtype automatically. Disable it to
        # ! avoid automatic type transform, which give us explicit control over the
        # ! execution type.
        with torch.cuda.amp.autocast(enabled=False):
            # !after cls_pred_ shape is [num_gt, anchors_in_boxes, num_classes]
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            # !using BCE to calc cls loss between cls_pred and gt_cls for each gt and anchor.
            pair_wise_cls_loss = F.binary_cross_entropy(
                    cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
                    ).sum(-1)
        del cls_preds_

        # !cost is weighted loss, and cls loss weight 1, iou_loss weight 3.
        # !but if anchor is not in both gt's box and center, it will cost 1e5 besides.
        # !shape is [num_gt, num_in_boxes_anchor]
        cost = (pair_wise_cls_loss
                + 0.0001 * pair_wise_regs_loss  # TODO 确定一下权重是否合理
                + 100000.0 * (~is_in_boxes_and_center)
                ) + (pair_wise_ious_loss * 5.0 if self.use_iou_loss else 0)

        (num_fg,
         gt_matched_classes,
         pred_ious_this_matching,
         matched_gt_inds,
         ) = self.dynamic_k_matching(
                cost,
                pair_wise_ious,
                gt_classes, num_gt,
                fg_mask
                # !fg_mask is also changed in this func. fg_mask shape is [total_num_anchors],
                )  # !but means which anchor have matching gt and fg_mask.sum() = num_fg
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (gt_matched_classes,  # !matched gt_classes based on anchor order, shape[num_fg]
                fg_mask,  # !shape is [total_num_anchors]
                pred_ious_this_matching,  # !anchors iou with gt, shape [num_fg]
                matched_gt_inds,  # !anchors' matched gts' indces, shape [num_fg]
                num_fg)  # !num of anchors that have gt to matched.

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,  # !shape is [num_gt, 8]
            gt_classes,  # !shape is [num_gt]
            bboxes_preds_per_image,  # !shape is [total_num_anchors, 8]
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            obj_preds,
            mode="gpu",
            ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # !fg_mask: show an anchor if in at least one box or center.
        # !         single dim. shape is [total_num_anchors], dtype is bool.
        # !is_in_boxes_and_center:  show an anchor if both in an ground truth's box and center,
        # !                         but only has anchors which at least in one box or center.
        # !                         shape is [num_gt, (is_in_boxes_anchor > 0).sum()]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                gt_bboxes_per_image,
                expanded_strides,
                x_shifts,
                y_shifts,
                total_num_anchors,
                num_gt,
                )

        # !remove those anchors that not in any gt_bbox and gt_centers
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        # !shape is [num_in_boxes_anchor, 24]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        # !shape is [num_in_boxes_anchor, 1]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        # !the number of anchors that we not dropped.
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # !shape is [num_gt, total_num_anchors], dtype = float
        pair_wise_ious = IOULoss.get_IOU(
                bboxes_preds_per_image[None, ...].repeat(gt_bboxes_per_image.shape[0], 1, 1),
                gt_bboxes_per_image[:, None, :].repeat(1, bboxes_preds_per_image.shape[0], 1)
                )

        # !this is used when debugging because there may be negative iou value.
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        pair_wise_regs_loss = YOLOXHead.bbox_l1_loss(
                gt_bboxes_per_image,
                bboxes_preds_per_image
                )

        # !shape is [num_gt, num_in_boxes_anchor, num_classes]
        # !classes is encoded as one-hot vector.
        gt_cls_per_image = (
                F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                    .float()
                    .unsqueeze(1)
                    .repeat(1, num_in_boxes_anchor, 1)
        )

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        # ! autocast-enabled allows tensors transform there dtype automatically. Disable it to
        # ! avoid automatic type transform, which give us explicit control over the
        # ! execution type.
        with torch.cuda.amp.autocast(enabled=False):
            # !after cls_pred_ shape is [num_gt, anchors_in_boxes, num_classes]
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            # !using BCE to calc cls loss between cls_pred and gt_cls for each gt and anchor.
            pair_wise_cls_loss = F.binary_cross_entropy(
                    cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
                    ).sum(-1)
        del cls_preds_

        # !cost is weighted loss, and cls loss weight 1, iou_loss weight 3.
        # !but if anchor is not in both gt's box and center, it will cost 1e5 besides.
        # !shape is [num_gt, num_in_boxes_anchor]
        cost = (pair_wise_cls_loss
                + 0.0001 * pair_wise_regs_loss  # TODO 确定一下权重是否合理
                + 100000.0 * (~is_in_boxes_and_center)
                ) + (pair_wise_ious_loss * 5.0 if self.use_iou_loss else 0)

        (num_fg,
         gt_matched_classes,
         pred_ious_this_matching,
         matched_gt_inds,
         ) = self.dynamic_k_matching(
                cost,
                pair_wise_ious,
                gt_classes, num_gt,
                fg_mask
                # !fg_mask is also changed in this func. fg_mask shape is [total_num_anchors],
                )  # !but means which anchor have matching gt and fg_mask.sum() = num_fg
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (gt_matched_classes,  # !matched gt_classes based on anchor order, shape[num_fg]
                fg_mask,  # !shape is [total_num_anchors]
                pred_ious_this_matching,  # !anchors iou with gt, shape [num_fg]
                matched_gt_inds,  # !anchors' matched gts' indces, shape [num_fg]
                num_fg)  # !num of anchors that have gt to matched.

    @staticmethod
    def bbox_l1_loss(bboxes_a, bboxes_b):
        if bboxes_a.shape[1] != 8 or bboxes_b.shape[1] != 8:
            raise IndexError

        # !loss shape are [a.shape[0], b.shape[0]]
        return F.l1_loss(
                bboxes_a[:, None, :].repeat(1, bboxes_b.shape[0], 1).float(),
                bboxes_b[None, :, :].repeat(bboxes_a.shape[0], 1, 1).float(),
                reduction="none"
                ).sum(dim=-1)

    @staticmethod
    def get_in_boxes_info(
            gt_bboxes_per_image,  # !shape[num_gt, 8]
            expanded_strides,  # !shape[1, total_num_anchors]
            x_shifts,  # !shape[1, total_num_anchors]
            y_shifts,  # !shape[1, total_num_anchors]
            total_num_anchors,  # ! = total_num_anchors in this img.
            num_gt,  # !number of ground truth in this img.
            ):
        # !a single dim tensor, all val = stride
        expanded_strides_per_image = expanded_strides[0]
        # !a single dim terson storge all points shift(e.g. 1 or 2)
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        # !these steps get anchors center points(not anchor boxs we predicted!) and copy them
        x_centers_per_image = (
                (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                    .unsqueeze(0)
                    .repeat(num_gt, 1)
        )  # ![total_num_anchors] -> [num_gt, total_num_anchors]
        y_centers_per_image = (
                (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                    .unsqueeze(0)
                    .repeat(num_gt, 1)
        )

        # *改成了判断中心点是否在四条边的直线之内
        # !thses bboxes' l,r,b,t means gtbboxes' left, right, bottom, top coordinates.
        # !gt_bboxes shape[num_gt, total_num_anchors]
        gt_bboxes_per_image_points = gt_bboxes_per_image[:, 0:8].unsqueeze(1).repeat(
                1, total_num_anchors, 1
                )

        # !whether the anchor's center points is in the gt bbox
        # !shape[num_gt, total_num_anchors], value dtype = bool
        is_in_boxes = IOULoss.if_inside_bounding_box(
                torch.stack((x_centers_per_image, y_centers_per_image), dim=-1),
                gt_bboxes_per_image_points
                )
        # is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0

        # !whether at least one gt_bbox encircle the center of an anchor.
        # !shape[total_num_anchors], value dtype = bool
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5  # !radius: 半径，范围

        # *改成了四个点坐标平均来确定中心点坐标
        gt_bboxes_per_image_x_centers = gt_bboxes_per_image[:, 0::2].sum(dim=-1) / 4
        gt_bboxes_per_image_y_centers = gt_bboxes_per_image[:, 1::2].sum(dim=-1) / 4
        # !thses bboxes' l,r,b,t means gtbboxes' left, right, bottom, top coordinates.
        # !but get from center point ± (2.5*anchor's side length)
        # !gt_boxes shape [num_gt, total_num_anchors].
        gt_bboxes_per_image_l = gt_bboxes_per_image_x_centers.unsqueeze(1).repeat(
                1, total_num_anchors
                ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = gt_bboxes_per_image_x_centers.unsqueeze(1).repeat(
                1, total_num_anchors
                ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = gt_bboxes_per_image_y_centers.unsqueeze(1).repeat(
                1, total_num_anchors
                ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = gt_bboxes_per_image_y_centers.unsqueeze(1).repeat(
                1, total_num_anchors
                ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        # !shape[num_gt, total_num_anchors, 4]
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        # !whether the anchor's center points is in center(bbox get from 2.5*anchor)
        # !shape[num_gt, total_num_anchors], value dtype = bool
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        # !whether at least one center encircle the center of an anchor.
        # !shape[total_num_anchors], value dtype = bool
        is_in_centers_all = is_in_centers.sum(dim=0) > 0
        # !show an anchor if in at least one box or center
        # !only single dim. shape is [total_num_anchors]
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        # !show an anchor if both in an ground truth's box and center
        # !it only has anchors that at least in one box or center
        # !shape is [num_gt, (is_in_boxes_anchor > 0).sum()]
        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )

        # !this is used to debug because there may be empty tensor in this mask.
        # if is_in_boxes_anchor.sum() == 0:
        #     raise Exception("fgmask = 0")
        return is_in_boxes_anchor, is_in_boxes_and_center

    @staticmethod
    def dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # !all matrixs' shape are [num_gt, num_in_boxes_anchor]
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        ious_in_boxes_matrix = pair_wise_ious
        # !pick at most 10 anchors to match gt (if num_in_boxes_anchor > 10)
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        # !pick topk matches' values, shape is [num_gt, topk]
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        # !sum every gt's topk anchors' ious, and ious less than 1 are set to 1.
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1, max=cost.size(1))
        dynamic_ks = dynamic_ks.tolist()  # !change sum losses to list
        # !for each gt:
        for gt_idx in range(num_gt):
            # !pick top k smallest cost as the match gt and anchor.
            # !k depends on each gt's total anchors' iou
            _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
                    )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        # !sum anchor matching gts num anchor-wise
        anchor_matching_gt = matching_matrix.sum(0)
        # !if exist anchor matching more than one gt
        if (anchor_matching_gt > 1).sum() > 0:
            # !pick the most matching gt anchor-wise for those have more than one matching
            # anchors.
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        # !pick those anchors that have matching gt.
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        # !get those anchors' num
        num_fg = fg_mask_inboxes.sum().item()

        # !fg_mask shows whether this anchor have matching gt.
        # !it is used outside this func. shape[total_nums_anchor]
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # !get each anchor's matching gt's index. shape is [num_fg]
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # !reorder gt_classes, make it match anchors order and remove those
        # !gts that cannot be matched with any anchor. shape is [num_fg]
        gt_matched_classes = gt_classes[matched_gt_inds]

        # !get those matched gts and anchors iou based on anchor order. shape is [num_fg]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
