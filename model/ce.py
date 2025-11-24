import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _assert_finite(x, where):
    if not torch.is_tensor(x):
        raise AssertionError(f"[{where}] not a tensor: {type(x)}")
    if not torch.isfinite(x).all():
        _stats(where, x)
        raise AssertionError(f"[NaN/Inf DETECTED] @ {where}")
def _stats(name, t):
    if not torch.is_tensor(t):
        print(f"[{name}] not tensor: {type(t)}"); return
    with torch.no_grad():
        fin = torch.isfinite(t)
        any_nan = (~fin).any().item()
        mn = t[fin].min().item() if fin.any() else float('nan')
        mx = t[fin].max().item() if fin.any() else float('nan')
        mean = t[fin].mean().item() if fin.any() else float('nan')
        print(f"[{name}] shape={tuple(t.shape)} min={mn:.3e} max={mx:.3e} mean={mean:.3e} any_nan={bool(any_nan)}")

    
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def _squeeze_binary_labels(label):
    if label.size(1) == 1:
        squeeze_label = label.view(len(label), -1)
    else:
        inds = torch.nonzero(label >= 1).squeeze()
        squeeze_label = inds[:,-1]
    return squeeze_label

def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    if label.size(-1) != pred.size(0):
        label = _squeeze_binary_labels(label)

    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def partial_cross_entropy(pred,
                          label,
                          weight=None,
                          reduction='mean',
                          avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    mask = label == -1
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    if mask.sum() > 0:
        loss *= (1-mask).float()
        avg_factor = (1-mask).float().sum()

    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

def kpos_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    target = label.float() / torch.sum(label, dim=1, keepdim=True).float()

    loss = - target * F.log_softmax(pred, dim=1)
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_kpos=False,
                 partial=False,
                 reduction='mean',
                 loss_weight=1.0,
                 thrds=None):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.use_kpos = use_kpos
        self.partial = partial
        self.reduction = reduction
        self.loss_weight = loss_weight
        if self.use_sigmoid and thrds is not None:
            self.thrds=inverse_sigmoid(thrds)
        else:
            self.thrds = thrds

        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        elif self.use_kpos:
            self.cls_criterion = kpos_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.thrds is not None:
            cut_high_mask = (label == 1) * (cls_score > self.thrds[1])
            cut_low_mask = (label == 0) * (cls_score < self.thrds[0])
            if weight is not None:
                weight *= (1 - cut_high_mask).float() * (1 - cut_low_mask).float()
            else:
                weight = (1 - cut_high_mask).float() * (1 - cut_low_mask).float()

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

def inverse_sigmoid(Y):
    X = []
    for y in Y:
        y = max(y,1e-14)
        if y == 1:
            x = 1e10
        else:
            x = -np.log(1/y-1)
        X.append(x)

    return X

    
class BCELosswithDiffLogits(nn.Module):
    def __init__(self):
        super(BCELosswithDiffLogits, self).__init__()

    def forward(self, logits, targets, weight=None,
                reduction='none', avg_factor=None):

        _assert_finite(logits,  "bce.logits@input")
        _assert_finite(targets, "bce.targets@input")

        with torch.cuda.amp.autocast(enabled=False):
            x = logits.float()
            y = targets.float()

            if x.dim() == 2:
                # ---- Case B: [B, C] → 일반 BCE ----
                p = torch.sigmoid(x).clamp(1e-6, 1-1e-6)
                if weight is None:
                    loss = - (y * torch.log(p) +
                                (1 - y) * torch.log1p(-p))
                else:
                    w = weight.float()
                    loss = - w * (y * torch.log(p) +
                                    (1 - y) * torch.log1p(-p))

            elif x.dim() == 3:
                # ---- Case A: [B, C, C] → SLP pairwise BCE ----
                B, C, _ = x.shape
                p = torch.sigmoid(x).clamp(1e-6, 1-1e-6)

                # label [B,C] → [B,C,1] → [B,C,C]
                y = y.view(B, C, 1).expand(B, C, C)

                if weight is None:
                    loss = - (y * torch.log(p) +
                                (1 - y) * torch.log1p(-p))
                else:
                    # weight [B,C]면 [B,C,1] → [B,C,C]로 확장
                    if weight.dim() == 2:
                        w = weight.view(B, C, 1).expand(B, C, C).float()
                    else:
                        w = weight.float()
                    loss = - w * (y * torch.log(p) +
                                    (1 - y) * torch.log1p(-p))
                # SLP 논문에서 subclass 방향으로 합/평균하는 역할
                loss = loss.mean(dim=2)   # [B,C]

            else:
                raise ValueError(f"BCELosswithDiffLogits: unexpected logits.dim()={x.dim()}")

        _assert_finite(loss, "bce.loss@raw")

        # avg_factor 있으면 수동 평균
        if avg_factor is not None and reduction == 'mean':
            loss = loss.sum() / (avg_factor + 1e-6)
        else:
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()
        return loss
