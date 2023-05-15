import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import weighted_loss


# 针对二分类任务的 Focal Loss
# 从实现原理上类似 BinaryCrossEntropyLossWithLogist
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean", eps=1e-8, with_logist=False):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.need_mean = reduction == "mean"
        self.need_sum = reduction == "sum"
        self.eps = eps
        self.logist = lambda x: (x.sigmoid() if with_logist else x)

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        pred = self.logist(pred)
        target_ = 1 - target

        # if target == 1, pred = pred.
        # else target == 0, pred = 1 - pred.
        probs = (target_ - pred).abs().clamp(min=self.eps, max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = (target_ - self.alpha).abs()

        # 根据 Focal Loss 的公式计算 Loss
        focal_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.need_mean:
            return focal_loss.mean()
        elif self.need_sum:
            return focal_loss.sum()
        else:
            return focal_loss


# 针对 Multi-Label 任务的 Focal Loss
# 从实现原理上类似CrossEntropyLoss
class FocalLossWithSoftmax(nn.Module):
    def __init__(self, gamma=2, weight=1, reduction="mean", eps=1e-8):
        super(FocalLossWithSoftmax, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.need_mean = reduction == "mean"
        self.need_sum = reduction == "sum"
        self.elipson = eps

    def forward(self, pred, labels):
        """
        cal culates loss
        pred: batch_size * labels_length
        labels: batch_size
        """
        label_onehot = F.one_hot(labels, pred.shape[-1])
        pred.softmax_(-1)

        # calculate log
        ce = -1 * torch.log(pred + self.elipson) * label_onehot
        focal_loss = torch.pow((1 - pred), self.gamma) * ce
        focal_loss = torch.mul(focal_loss, self.weight)
        focal_loss = torch.sum(focal_loss, dim=-1)
        if self.need_mean:
            return focal_loss.mean()
        elif self.need_sum:
            return focal_loss.sum()
        else:
            return focal_loss


# 实验表明，gamma=1/2，alpha=0.25/0.5
# import numpy as np
# import matplotlib.pyplot as plt
# label_y = torch.randint(0, 1, (1000,))
# pred_y1 = torch.from_numpy(np.linspace(0, 1, 1000)).float()
# pred_y2 = torch.zeros(1000)
# print()
# # print(pred_y.sigmoid())
# # print(label_y)
# x = (label_y - pred_y).abs().numpy()
# y0 = BinaryFocalLoss(0.5, gamma=0, reduction="none")(pred_y, label_y)
# y1 = BinaryFocalLoss(0.5, gamma=1, reduction="none")(pred_y, label_y)
# y2 = BinaryFocalLoss(0.5, gamma=2, reduction="none")(pred_y, label_y)
# plt.figure(figsize=(10,10))
# plt.xlim([0, 1])
# plt.ylim([0, 3])
# plt.plot(x, y0, color='blue', label='gamma=0')
# plt.plot(x, y1, color='red', label='gamma=1')
# plt.plot(x, y2, color='green', label='gamma=2')
# plt.legend()
# plt.show()

# 用一个简单的线性模型验证Loss函数的正确性。
# net = nn.Sequential(
#         nn.Linear(6, 12), nn.Sigmoid(),
#         nn.Linear(12, 12), nn.Sigmoid(),
#         nn.Linear(12, 2)
#         )
# optim = torch.optim.Adam(net.parameters(), lr=0.1)
# loss_func = BinaryFocalLoss(alpha=0.5, gamma=0, with_logist=True)
# # loss_func = nn.BCEWithLogitsLoss()
# data = torch.rand(100, 6)
# acty = (data.sum(-1) < 3).long()
# target = F.one_hot(acty, 2).float()
# for epoch in range(100):
#     optim.zero_grad()
#     pred = net(data)
#     loss = loss_func(pred, target)
#     loss.backward()
#     optim.step()
#
#     predy = pred.detach().argmax(dim=-1)
#     corret = (predy == acty).sum()
#     print(
#             f"accuracy = {corret / 100}, correct num = {corret}\n"
#             f"loss = {loss.item():.4f}"
#             )


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert (
            len(target) == 2
    ), """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
            pred, zerolabel, reduction="none"
            ) * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = torch.nonzero((label >= 0) & (label < bg_class_ind), as_tuple=False).squeeze(
            1
            )
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pred[pos, pos_label], score[pos], reduction="none"
            ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def distribution_focal_loss(pred, label):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = (
            F.cross_entropy(pred, dis_left, reduction="none") * weight_left
            + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
    )
    return loss


class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, use_sigmoid=True, beta=2.0, reduction="mean", loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid in QFL supported now."
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
            self, pred, target, weight=None, avg_factor=None, reduction_override=None
            ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                    pred,
                    target,
                    weight,
                    beta=self.beta,
                    reduction=reduction,
                    avg_factor=avg_factor,
                    )
        else:
            raise NotImplementedError
        return loss_cls


class DistributionFocalLoss(nn.Module):
    r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
            self, pred, target, weight=None, avg_factor=None, reduction_override=None
            ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor
                )
        return loss_cls
