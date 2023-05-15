#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch.nn as nn

from .head_acc import YOLOXHead
from .shufflenetv2 import ShuffleNetV2
from .neck import YOLOPAFPN


class YOLOX(nn.Module):
    """
    yolox model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, num_class, backbone=None, head=None):
        super().__init__()

        if backbone is None:
            backbone = YOLOPAFPN(
                    backbone=ShuffleNetV2(model_size="0.5x"), depth=0.33, width=0.25,
                    in_channels=[192, 384, 768], depthwise=True,
                    )

        if head is None:
            head = YOLOXHead(
                    num_classes=num_class, width=0.25, in_channels=[192, 384, 768],
                    strides=[8, 16, 32], obj_weight=10.0, cls_weight=10.0, reg_weight=0.1,
                    depthwise=True
                    )
            head.initialize_biases()

        self.backbone = backbone
        self.head = head
        self.apply(self.__init_yolox)

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            results = self.head(
                    fpn_outs, *targets
                    )
            if isinstance(results, tuple):
                loss, iou_loss, obj_loss, cls_loss, reg_loss, num_fg = results
                outputs = {
                        "total_loss": loss,
                        "iou_loss":   iou_loss,
                        "obj_loss":   obj_loss,
                        "cls_loss":   cls_loss,
                        "reg_loss":   reg_loss,
                        "num_fg":     num_fg,
                        }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    @staticmethod
    def __init_yolox(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                # * try to change momentum to increase BN layer effect.
                m.momentum = 0.03  # pytorch default 0.1, yolo default 0.03
