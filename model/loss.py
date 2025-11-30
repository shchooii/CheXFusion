import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from model.ce import BCELosswithDiffLogits, cross_entropy, binary_cross_entropy, partial_cross_entropy, kpos_cross_entropy, weight_reduce_loss
from model.loss_asl_wasl_slp import ASLwithClassWeight2
from torch.nn import Parameter

pos_counts = torch.tensor([
    67597, 4361, 76900, 16038, 38574, 4255, 30119, 1158, 11883, 4049, 10218, 2533,
    79931, 5529, 41869, 7663, 69240, 675, 3369, 788, 48093, 543, 14983, 2453, 89140, 3499
], dtype=torch.float32)

N_total = 264_849  # total_instance_num
neg_counts = N_total - pos_counts

def get_loss(type, class_instance_nums, total_instance_num):
    if type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif type == 'wbce':
        return BCEwithClassWeights(class_instance_nums, total_instance_num)
    elif type == 'asl':
        return AsymmetricLoss()
    elif type == 'wasl':
        return ASLwithClassWeight(class_instance_nums, total_instance_num)
    elif type == 'apl':
        return APLLoss()
    elif type == 'wapl':
        return APLwithClassWeight(class_instance_nums=class_instance_nums, total_instance_num=total_instance_num)
    elif type == 'ral':
        return Ralloss()
    elif type == 'wral':
        return RalwithClassWeight(class_instance_nums=class_instance_nums,total_instance_num=total_instance_num)
    elif type == 'focal':
        return FocalLoss()
    elif type == 'hill':
        return Hill()
    elif type == 'mfm':
        return MultiGrainedFocalLoss()
    elif type == 'slp':
        return ResampleLoss2(class_instance_nums=class_instance_nums, total_instance_num=total_instance_num)
    elif type == 'waslslp1':
        return ASLwithClassWeight2(
            class_instance_nums, total_instance_num,
            use_slp=True,
            slp_cfg=dict(
                coef_alpha=1.0/3.0,
                coef_beta=0.7,
                up_mult=12,  # CXR-LT에서 추천
                dw_mult=9,
            )
        )
    elif type == 'waslslp2':
        return ASLwithClassWeight2(
            class_instance_nums, total_instance_num,
            use_slp=True,
            slp_cfg=dict(
                coef_alpha=0.4,
                coef_beta=0.8,
                up_mult=15,  # CXR-LT에서 추천
                dw_mult=9,
            )
        )
    elif type == 'waslslp3':
        return ASLwithClassWeight2(
            class_instance_nums, total_instance_num,
            use_slp=True,
            slp_cfg=dict(
                coef_alpha=0.5,
                coef_beta=0.5,
                up_mult=9,  # CXR-LT에서 추천
                dw_mult=6,
            )
        )
    else:
        raise ValueError(f'Unknown loss type: {type}')


class MultiGrainedFocalLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, gamma_class_ng=1.2,
                 clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True,
                 distribution_path=None, co_occurrence_matrix=None):
        super(MultiGrainedFocalLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_class_ng = gamma_class_ng
        self.gamma_class_pos = 1
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.distribution_path = distribution_path
        
        # class_instance_nums 상수로 “직접” 내장
        class_counts = [
            67597, 4361, 76900, 16038, 38574, 4255, 30119, 1158, 11883, 4049,
            10218, 2533, 79931, 5529, 41869, 7663, 69240, 675, 3369, 788,
            48093, 543, 14983, 2453, 89140, 3499
        ]
        # 버퍼로 보관(디바이스 전환 자동 대응)
        self.register_buffer("class_counts", torch.tensor(class_counts, dtype=torch.float32))
        self.register_buffer("weight", torch.ones(len(class_counts), dtype=torch.float32))
        self._weight_ready = False

        # self.spls_loss = SPLC(batch_size=32)

    @torch.no_grad()
    def create_weight(self, device=None, dtype=torch.float32):
        """
        외부 입력 없이 내부 상수 class_counts로 weight 계산.
        호출은 forward에서 x.device/dtype에 맞춰 1회 자동 실행.
        """
        dist = self.class_counts.to(device=device, dtype=dtype)  # [C]
        total = dist.sum()
        prob = dist / (total + self.eps)         # p_i
        prob = prob / (prob.max() + self.eps)    # 정규화(최대=1)
        # 논문/원코드와 동일한 변환: (-log p + 1)^(1/6)
        weight = torch.pow(-torch.log(prob.clamp_min(self.eps)) + 1.0, 1.0 / 6)

        # 버퍼 갱신
        self.weight = weight
        self._weight_ready = True
        

    @torch.no_grad()                  
    def create_co_occurrence_matrix(self, co_occurrence_matrix):
        co_occurrence_matrix = torch.tensor(np.load(co_occurrence_matrix)).cuda()
        self.co_occurrence_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)


    def forward(self, x, y):
        
        if not self._weight_ready:
            self.create_weight(device=x.device, dtype=x.dtype)

        weight = self.weight.to(dtype=x.dtype, device=x.device)

        # positive -
        x_sigmoid = torch.pow(torch.sigmoid(x), 1)
        gamma_class_pos = 1
        xs_pos = x_sigmoid * gamma_class_pos
        xs_neg = 1 - x_sigmoid

        # negative +
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # basic CE
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = (los_pos + los_neg) * weight

        # asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                one_sided_gamma = (self.gamma_pos) * y + (self.gamma_neg + weight) * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            loss = loss * one_sided_w

        loss =- loss.sum()
        return loss
    
    
class BCEwithClassWeights(nn.Module):
    def __init__(self, class_instance_nums, total_instance_num):
        super(BCEwithClassWeights, self).__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1-p)
        self.neg_weights = torch.exp(p)

    def forward(self, pred, label):
        # https://www.cse.sc.edu/~songwang/document/cvpr21d.pdf (equation 4)
        weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()
        loss = nn.functional.binary_cross_entropy_with_logits(pred, label, weight=weight)
        return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps


    def forward(self, x, y):

        # Calculating Probabilities
        x_sigmoid = torch.pow(torch.sigmoid(x),1) 
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        # loss=loss*self.weight
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                one_sided_gamma = (self.gamma_pos) * y + self.gamma_neg  * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            loss = loss * one_sided_w
        return -loss.sum()


# class ASLwithClassWeight(nn.Module):
#     def __init__(self, class_instance_nums, total_instance_num, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
#         super(ASLwithClassWeight, self).__init__()
#         class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
#         p = class_instance_nums / total_instance_num
#         self.pos_weights = torch.exp(1-p)
#         self.neg_weights = torch.exp(p)
#         self.gamma_neg = gamma_neg
#         self.gamma_pos = gamma_pos
#         self.clip = clip
#         self.eps = eps

#     def forward(self, pred, label):
#         weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()

#         # Calculating Probabilities
#         xs_pos = torch.sigmoid(pred)
#         xs_neg = 1.0 - xs_pos

#         # Asymmetric Clipping
#         if self.clip is not None and self.clip > 0:
#             xs_neg.add_(self.clip).clamp_(max=1)

#        # Basic CE calculation
#         los_pos = label * torch.log(xs_pos.clamp(min=self.eps))
#         los_neg = (1 - label) * torch.log(xs_neg.clamp(min=self.eps))
#         loss = los_pos + los_neg
#         loss *= weight

#         # Asymmetric Focusing
#         if self.gamma_neg > 0 or self.gamma_pos > 0:
#             pt0 = xs_pos * label 
#             pt1 = xs_neg * (1 - label)  # pt = p if t > 0 else 1-p
#             pt = pt0 + pt1
#             one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1 - label)
#             one_sided_w = torch.pow(1 - pt, one_sided_gamma)
#             loss *= one_sided_w

#         return -loss.mean()


class ASLwithClassWeight(nn.Module):
    def __init__(self, class_instance_nums, total_instance_num, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(ASLwithClassWeight, self).__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1-p)
        self.neg_weights = torch.exp(p)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, pred, label):
        weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()

        # Calculating Probabilities
        xs_pos = torch.sigmoid(pred)
        xs_pos = xs_pos.clamp(min=self.eps, max=1.0 - self.eps)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

       # Basic CE calculation
        los_pos = label * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - label) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        loss *= weight

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * label 
            pt1 = xs_neg * (1 - label)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1 - label)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


class APLLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(APLLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # parameters of Taylor expansion polynomials
        self.epsilon_pos = 1.0
        self.epsilon_neg = 0.0
        self.epsilon_pos_pow = -2.5

    def forward(self, x, y):
        """"
        x: input logits with size (batch_size, number of labels).
        y: binarized multi-label targets with size (batch_size, number of labels).
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Taylor expansion polynomials
        los_pos = y * (torch.log(xs_pos.clamp(min=self.eps)) + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps)) + self.epsilon_pos_pow * 0.5 * torch.pow(1 - xs_pos.clamp(min=self.eps), 2) )
        los_neg = (1 - y) * (torch.log(xs_neg.clamp(min=self.eps)) + self.epsilon_neg * (xs_neg.clamp(min=self.eps)) )
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                one_sided_gamma = (self.gamma_pos) * y + self.gamma_neg  * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w
        return -loss.sum()


class APLwithClassWeight(nn.Module):
    def __init__(self,
                 class_instance_nums=None,
                 total_instance_num=None,
                 gamma_neg=4, gamma_pos=0,
                 clip=0.05, eps=1e-8):
        super(APLwithClassWeight, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

        # Taylor expansion coefficients
        self.epsilon_pos = 1.0
        self.epsilon_neg = 0.0
        self.epsilon_pos_pow = -2.5

        # ----- class weights (optional) -----
        if class_instance_nums is not None and total_instance_num is not None:
            cls = torch.as_tensor(class_instance_nums, dtype=torch.float32)
            p = cls / float(total_instance_num)
            pos_w = torch.exp(1 - p)   # positive term weight
            neg_w = torch.exp(p)       # negative term weight
            self.register_buffer("pos_w", pos_w)
            self.register_buffer("neg_w", neg_w)
        else:
            self.register_buffer("pos_w", None)
            self.register_buffer("neg_w", None)

    def forward(self, x, y):
        """
        x: logits (N, C)
        y: binary targets (N, C)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Base APL (pos/neg Taylor terms)
        los_pos = y * ( torch.log(xs_pos.clamp(min=self.eps))
                        + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps))
                        + self.epsilon_pos_pow * 0.5 * (1 - xs_pos.clamp(min=self.eps))**2 )

        los_neg = (1 - y) * ( torch.log(xs_neg.clamp(min=self.eps))
                              + self.epsilon_neg * xs_neg.clamp(min=self.eps) )

        # ----- class weights -----
        if self.pos_w is not None and self.neg_w is not None:
            pos_w = self.pos_w.to(x.device, x.dtype)  # (C,)
            neg_w = self.neg_w.to(x.device, x.dtype)
            los_pos = los_pos * pos_w                 # (N,C) * (C,)
            los_neg = los_neg * neg_w

        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt  = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = (1 - pt).pow(one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()


class Ralloss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, lamb=1.5, epsilon_neg=0.0, epsilon_pos=1.0, epsilon_pos_pow=-2.5, disable_torch_grad_focal_loss=False):
        super(Ralloss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # parameters of Taylor expansion polynomials
        self.epsilon_pos = epsilon_pos
        self.epsilon_neg = epsilon_neg
        self.epsilon_pos_pow = epsilon_pos_pow
        self.margin = 1.0
        self.lamb = lamb

    def forward(self, x, y):
        """"
        x: input logits with size (batch_size, number of labels).
        y: binarized multi-label targets with size (batch_size, number of labels).
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Taylor expansion polynomials
        los_pos = y * (torch.log(xs_pos.clamp(min=self.eps)) + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps)) + self.epsilon_pos_pow * 0.5 * torch.pow(1 - xs_pos.clamp(min=self.eps), 2))
        los_neg = (1 - y) * (torch.log(xs_neg.clamp(min=self.eps)) + self.epsilon_neg * (xs_neg.clamp(min=self.eps)) ) * (self.lamb - x_sigmoid) * x_sigmoid ** 2 * (self.lamb - xs_neg)
        loss = los_pos + los_neg

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        # loss=loss*self.weight
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                one_sided_gamma = (self.gamma_pos) * y + self.gamma_neg  * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            loss = loss * one_sided_w
        return -loss.sum()



class RalwithClassWeight(nn.Module):
    def __init__(self,
                 class_instance_nums=None,
                 total_instance_num=None,
                 gamma_neg=4, gamma_pos=0,
                 clip=0.05, eps=1e-8,
                 lamb=1.5, epsilon_neg=0.0, epsilon_pos=1.0, epsilon_pos_pow=-2.5):
        super(RalwithClassWeight, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

        # RAL params
        self.epsilon_pos = epsilon_pos
        self.epsilon_neg = epsilon_neg
        self.epsilon_pos_pow = epsilon_pos_pow
        self.lamb = lamb
        self.margin = 1.0

        # ----- class weights (optional) -----
        if class_instance_nums is not None and total_instance_num is not None:
            cls = torch.as_tensor(class_instance_nums, dtype=torch.float32)
            p = cls / float(total_instance_num)
            pos_w = torch.exp(1 - p)   # positive term weight
            neg_w = torch.exp(p)       # negative term weight
            self.register_buffer("pos_w", pos_w)
            self.register_buffer("neg_w", neg_w)
        else:
            self.register_buffer("pos_w", None)
            self.register_buffer("neg_w", None)

    def forward(self, x, y):
        """
        x: logits (N, C)
        y: binary targets (N, C)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Positive term: CE + Taylor relaxation (as in RAL)
        los_pos = y * ( torch.log(xs_pos.clamp(min=self.eps))
                        + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps))
                        + self.epsilon_pos_pow * 0.5 * (1 - xs_pos.clamp(min=self.eps))**2 )

        # Negative term: CE + relaxation to down-weight possibly false negatives
        ce_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        relax_neg = (1 - y) * (-(self.lamb - x_sigmoid) * (x_sigmoid ** 2))
        los_neg = ce_neg + relax_neg

        # ----- class weights -----
        if self.pos_w is not None and self.neg_w is not None:
            pos_w = self.pos_w.to(x.device, x.dtype)
            neg_w = self.neg_w.to(x.device, x.dtype)
            los_pos = los_pos * pos_w
            los_neg = los_neg * neg_w

        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt  = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = (1 - pt).pow(one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()


class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 use_kpos=False,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 gamma=2,
                 balance_param=0.25):
        super(FocalLoss, self).__init__()
        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.use_kpos = use_kpos
        self.partial = partial
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.gamma = gamma
        self.balance_param = balance_param

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
        reduction = (reduction_override if reduction_override else self.reduction)
        logpt = - self.cls_criterion(cls_score, label, weight, reduction='none',
                                     avg_factor=avg_factor)
        # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        loss = self.loss_weight * balanced_focal_loss
        return loss.sum()
def create_loss():
    return FocalLoss(gamma = 2.0)


class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "
    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 
    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.
    .. note::
        Sigmoid will be done in loss. 
    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2
    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward
        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
def create_loss():
    return Hill()


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

def _assert_finite(x, where):
    if not torch.is_tensor(x):
        raise AssertionError(f"[{where}] not a tensor: {type(x)}")
    if not torch.isfinite(x).all():
        _stats(where, x)
        raise AssertionError(f"[NaN/Inf DETECTED] @ {where}")


class ResampleLoss2(nn.Module):

    def __init__(self,
                 class_instance_nums,
                 total_instance_num,
                 up_mult=12,dw_mult=9,
                 use_sigmoid=True,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                #  focal=dict(
                #     focal=False,
                #     mode='focal',          # 또는 생략 (default = 'focal')
                #     gamma=2,
                #     balance_param=2.0
                # ),
                 focal=dict(
                    focal=True,
                    mode='asl',
                    gamma_neg=4,
                    gamma_pos=0,
                    clip=0.05,
                    balance_param=1.0
                ),
                #  focal=dict(
                #     focal=True,
                #     mode='apl',
                #     gamma_neg=4,
                #     gamma_pos=0,
                #     clip=0.05,
                #     epsilon_pos=1.0,
                #     epsilon_neg=0.0,
                #     epsilon_pos_pow=-2.5,
                #     balance_param=1.0
                # ),
                # focal=dict(
                #     focal=True,
                #     mode='ral',
                #     gamma_neg=4,
                #     gamma_pos=0,
                #     clip=0.05,
                #     epsilon_pos=1.0,
                #     epsilon_neg=0.0,
                #     epsilon_pos_pow=-2.5,
                #     lamb=1.5,
                #     balance_param=1.0
                # ),
                # focal=dict(
                #     focal=True,
                #     mode='mfm',
                #     gamma_neg=4,
                #     gamma_pos=0,
                #     clip=0.05,
                #     balance_param=1.0,
                #     gamma_class_ng=1.2,
                #     gamma_class_pos=1.0
                # ),
                 CB_loss=dict(
                     CB_beta=0.99,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 map_param=dict(
                     alpha=0.1,
                     beta=7.0,
                     gamma=0.25
                 ),
                 logit_reg=dict(
                     neg_scale=1.0,
                     init_bias=1.0
                 ),
                 coef_param=dict(
                      coef_alpha=1.0 / 3.0,
                      coef_beta=0.7
                 ),
                 reweight_func='CB',  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 ):
        super(ResampleLoss2, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.up_mult = up_mult
        self.dw_mult = dw_mult
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = BCELosswithDiffLogits() #binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        # self.focal = focal['focal']
        # self.gamma = focal['gamma']
        # self.balance_param = focal['balance_param']
        # focal / ASL / APL / RAL 공통 파라미터
        self.focal = focal.get('focal', False)
        self.focal_mode = focal.get('mode', 'focal')  # 'focal', 'asl', 'apl', 'ral'

        # 기본 focal
        self.gamma = focal.get('gamma', 2)
        self.balance_param = focal.get('balance_param', 2.0)

        # ASL 용
        self.gamma_neg = focal.get('gamma_neg', 4)
        self.gamma_pos = focal.get('gamma_pos', 0)
        self.clip = focal.get('clip', 0.05)

        # APL / RAL 용 Taylor 계수
        self.epsilon_pos = focal.get('epsilon_pos', 1.0)
        self.epsilon_neg = focal.get('epsilon_neg', 0.0)
        self.epsilon_pos_pow = focal.get('epsilon_pos_pow', -2.5)

        # RAL 용 lambda
        self.lamb = focal.get('lamb', 1.5)

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        # coef params
        self.coef_alpha = coef_param['coef_alpha']
        self.coef_beta = coef_param['coef_beta']
        
        # 🔹 co-occurrence sparsification 하이퍼파라미터
        #   - tao_th: P(j|c)가 이 값보다 작으면 0으로 날림
        #   - tao_topk: 각 row(c)마다 상위 몇 개 co-occ만 살릴지
        self.tao_th = 0.15        # 예: 0.15 ~ 0.20 정도부터 시도
        self.tao_topk = 3         # 각 class당 top-3 co-occ만 사용
        
        # ⚙️ class frequency 계산 (config에서 직접)
        self.class_freq = torch.tensor(class_instance_nums, dtype=torch.float32).cuda()
        self.num_classes = len(class_instance_nums)
        self.train_num = float(total_instance_num)
        self.neg_class_freq = self.train_num - self.class_freq
        self.class_freq = torch.clamp(self.class_freq, min=1.0)
        self.neg_class_freq = torch.clamp(self.neg_class_freq, min=1.0)
        
        # ---------- MFM(Multi-Grained Focal)용 파라미터 ----------
        # gamma_neg / gamma_pos / clip / eps 는 이미 self.gamma_neg, self.gamma_pos, self.clip, self.eps로 갖고 있으니
        # MFM 특유 파라미터만 추가
        self.gamma_class_ng  = focal.get('gamma_class_ng', 1.2)
        self.gamma_class_pos = focal.get('gamma_class_pos', 1.0)
        self.eps = 1e-6

        self.register_buffer("mfm_weight", torch.ones(self.num_classes, dtype=torch.float32))
        self._mfm_weight_ready = False
        
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        eps = 1e-6
        self.init_bias = -torch.log(self.train_num / (self.class_freq + eps) - 1 + eps) * init_bias / self.neg_scale
        # self.init_bias = - torch.log(
        #     self.train_num / self.class_freq - 1) * init_bias / self.neg_scale

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

        # print('\033[1;35m loading from {} | {} | {} | s\033[0;0m'.format(freq_file, reweight_func, logit_reg))
        # print('\033[1;35m rebalance reweighting mapping params: {:.2f} | {:.2f} | {:.2f} \033[0;0m'.format(self.map_alpha, self.map_beta, self.map_gamma))

    def forward(self,
                norm_prop, 
                nonzero_var_tensor, 
                zero_var_tensor, 
                normalized_sigma_cj, 
                normalized_ro_cj, 
                normalized_tao_cj,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        
        _assert_finite(cls_score, "resample2.cls_score@input")
        _assert_finite(label, "resample2.label@input")
        
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)
        if weight is not None:
            _assert_finite(weight, "resample2.weight@after_reweight")  
        
        
        
        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, norm_prop, nonzero_var_tensor, zero_var_tensor, normalized_sigma_cj, normalized_ro_cj, 
                normalized_tao_cj,weight)
        _assert_finite(cls_score, "resample2.cls_score@after_logit_reg")
        if weight is not None:
            _assert_finite(weight, "resample2.weight@after_logit_reg")
    
        if self.focal:
            # -------------------------
            # mode = 'focal' (기존 focal)
            # -------------------------
            if self.focal_mode == 'focal':
                logpt = - self.cls_criterion(
                    cls_score.clone(), label, weight=None, reduction='none',
                    avg_factor=avg_factor)
                pt = torch.exp(logpt)
                loss = self.cls_criterion(
                    cls_score, label.float(), weight=weight, reduction='none')
                loss = ((1 - pt) ** self.gamma) * loss
                loss = self.balance_param * loss
                loss = weight_reduce_loss(
                    loss, weight=None, reduction=reduction, avg_factor=avg_factor
                )
            
            # -------------------------
            # mode = 'focal2' (순수 Focal + weight)
            # -------------------------
            elif self.focal_mode == 'focal2':
                # 1) base BCE (weight 없이)
                base_loss = self.cls_criterion(
                    cls_score, label.float(),
                    weight=None,
                    reduction='none',
                    avg_factor=None
                )   # shape: [B,C] 또는 [B,C,C]

                _assert_finite(base_loss, "resample2.focal2.base_loss")

                # 2) pt = exp(-BCE)
                logpt = -base_loss
                pt = torch.exp(logpt)
                _assert_finite(pt, "resample2.focal2.pt")

                # 3) Focal term
                focal_loss = ((1 - pt) ** self.gamma) * base_loss  # (N,C) or (N,C,C)

                # 4) reweight 적용 (rebalance/CB 등)
                if weight is not None:
                    # weight: [B,C] → 필요하면 [B,C,C]로 확장
                    if focal_loss.dim() == 3 and weight.dim() == 2:
                        B, C, _ = focal_loss.shape
                        w = weight.view(B, C, 1).expand(B, C, C)
                    else:
                        w = weight
                    focal_loss = focal_loss * w

                focal_loss = self.balance_param * focal_loss

                # 5) reduction (mean/sum/none + avg_factor)
                loss = weight_reduce_loss(
                    focal_loss,
                    weight=None,
                    reduction=reduction,
                    avg_factor=avg_factor
                )
                _assert_finite(loss, "resample2.focal2.loss@after_reduce")

            # -------------------------
            # mode = 'asl' (Asymmetric Loss)
            # -------------------------
            elif self.focal_mode == 'asl':
                x = cls_score
                y = label.float()

                x_sigmoid = torch.sigmoid(x)
                xs_pos = x_sigmoid
                xs_neg = 1.0 - x_sigmoid

                # asymmetric clipping
                if self.clip is not None and self.clip > 0:
                    xs_neg = (xs_neg + self.clip).clamp(max=1.0)

                # CE 부분
                los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
                los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
                loss = los_pos + los_neg  # (N,C)

                # reweight (CB / rebalance 등)
                if weight is not None:
                    loss = loss * weight

                # asymmetric focusing
                if self.gamma_neg > 0 or self.gamma_pos > 0:
                    with torch.no_grad():
                        pt0 = xs_pos * y
                        pt1 = xs_neg * (1 - y)
                        pt = pt0 + pt1
                        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                    loss = loss * one_sided_w

                loss = - loss                     # CE sign 반전
                loss = self.balance_param * loss  # balance scale
                loss = weight_reduce_loss(
                    loss, weight=None, reduction=reduction, avg_factor=avg_factor
                )

            # -------------------------
            # mode = 'apl' (Asymmetric Polynomial Loss)
            # -------------------------
            elif self.focal_mode == 'apl':
                x = cls_score
                y = label.float()

                x_sigmoid = torch.sigmoid(x)
                xs_pos = x_sigmoid
                xs_neg = 1.0 - x_sigmoid

                if self.clip is not None and self.clip > 0:
                    xs_neg = (xs_neg + self.clip).clamp(max=1.0)

                # APL의 Taylor 기반 CE
                xs_pos_clamp = xs_pos.clamp(min=self.eps)
                xs_neg_clamp = xs_neg.clamp(min=self.eps)

                los_pos = y * (
                    torch.log(xs_pos_clamp)
                    + self.epsilon_pos * (1 - xs_pos_clamp)
                    + self.epsilon_pos_pow * 0.5 * (1 - xs_pos_clamp) ** 2
                )
                los_neg = (1 - y) * (
                    torch.log(xs_neg_clamp)
                    + self.epsilon_neg * xs_neg_clamp
                )
                loss = los_pos + los_neg   # (N,C)

                if weight is not None:
                    loss = loss * weight

                # asymmetric focusing (APL도 pt에 따라 gamma 적용 가능)
                if self.gamma_neg > 0 or self.gamma_pos > 0:
                    with torch.no_grad():
                        pt0 = xs_pos * y
                        pt1 = xs_neg * (1 - y)
                        pt = pt0 + pt1
                        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                    loss = loss * one_sided_w

                loss = - loss
                loss = self.balance_param * loss
                loss = weight_reduce_loss(
                    loss, weight=None, reduction=reduction, avg_factor=avg_factor
                )

            # -------------------------
            # mode = 'ral' (Robust Asymmetric Loss)
            # -------------------------
            elif self.focal_mode == 'ral':
                x = cls_score
                y = label.float()

                x_sigmoid = torch.sigmoid(x)
                xs_pos = x_sigmoid
                xs_neg = 1.0 - x_sigmoid

                if self.clip is not None and self.clip > 0:
                    xs_neg = (xs_neg + self.clip).clamp(max=1.0)

                xs_pos_clamp = xs_pos.clamp(min=self.eps)
                xs_neg_clamp = xs_neg.clamp(min=self.eps)

                # RAL: pos = APL식, neg = CE + relaxation
                los_pos = y * (
                    torch.log(xs_pos_clamp)
                    + self.epsilon_pos * (1 - xs_pos_clamp)
                    + self.epsilon_pos_pow * 0.5 * (1 - xs_pos_clamp) ** 2
                )

                ce_neg = (1 - y) * torch.log(xs_neg_clamp)
                relax_neg = (1 - y) * (-(self.lamb - x_sigmoid) * (x_sigmoid ** 2))
                los_neg = ce_neg + relax_neg

                loss = los_pos + los_neg  # (N,C)

                if weight is not None:
                    loss = loss * weight

                # asymmetric focusing
                if self.gamma_neg > 0 or self.gamma_pos > 0:
                    with torch.no_grad():
                        pt0 = xs_pos * y
                        pt1 = xs_neg * (1 - y)
                        pt = pt0 + pt1
                        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                    loss = loss * one_sided_w

                loss = - loss
                loss = self.balance_param * loss
                loss = weight_reduce_loss(
                    loss, weight=None, reduction=reduction, avg_factor=avg_factor
                )
            # -------------------------
            # mode = 'mfm' (Multi-Grained Focal Loss)
            # -------------------------
            elif self.focal_mode == 'mfm':
                x = cls_score
                y = label.float()

                # 1) MFM weight 준비 (한 번만 계산)
                if not self._mfm_weight_ready:
                    self._create_mfm_weight(device=x.device, dtype=x.dtype)

                w_cls = self.mfm_weight.to(device=x.device, dtype=x.dtype)  # [C]

                # 2) sigmoid, pos/neg prob
                x_sigmoid = torch.sigmoid(x)            # [N, C]
                xs_pos = x_sigmoid * self.gamma_class_pos
                xs_neg = 1.0 - x_sigmoid

                # 3) negative clipping
                if self.clip is not None and self.clip > 0:
                    xs_neg = (xs_neg + self.clip).clamp(max=1.0)

                xs_pos_clamp = xs_pos.clamp(min=self.eps)
                xs_neg_clamp = xs_neg.clamp(min=self.eps)

                # 4) base CE (클래스 weight 곱)
                los_pos = y * torch.log(xs_pos_clamp)           # [N, C]
                los_neg = (1 - y) * torch.log(xs_neg_clamp)     # [N, C]
                loss = (los_pos + los_neg) * w_cls              # per-class weight

                # 5) asymmetric focusing (γ_neg + weight 형태)
                if self.gamma_neg > 0 or self.gamma_pos > 0:
                    with torch.no_grad():
                        pt0 = xs_pos * y
                        pt1 = xs_neg * (1 - y)
                        pt = pt0 + pt1                           # [N, C]
                        gamma_eff = self.gamma_pos * y \
                                    + (self.gamma_neg + w_cls) * (1 - y)
                        one_sided_w = torch.pow(1 - pt, gamma_eff)
                    loss = loss * one_sided_w

                # 6) sign 반전 + balance_param + reduction
                loss = -loss
                loss = self.balance_param * loss
                loss = weight_reduce_loss(
                    loss, weight=None,
                    reduction=reduction,
                    avg_factor=avg_factor
                )

            else:
                raise ValueError(f"Unknown focal_mode: {self.focal_mode}")

        else:
            # focal / asl / apl / ral 모두 사용 안 할 때
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                        reduction=reduction)
        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in ['rebalance']:
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in ['CB']:
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' == self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' == self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def pgd_like(self, x, y,step,sign):
        y = y.to(torch.float32)
        iters = int(torch.max(step).item()+1)
        logit=torch.zeros_like(x)
        for k in range(iters):
            grad = torch.sigmoid(x)-y
            x = x + grad*sign/x.shape[1]
            logit = logit + x*(step==k)
        return logit

    def pgd_like_diff_sign(self, x, y, step, sign):


        y = y.to(torch.float32)

        iters = int(torch.max(step).item()+1)
        logit = torch.zeros_like(x)
        for k in range(iters):
            grad = torch.sigmoid(x)-y
            x = x + grad*sign/x.shape[1]
            logit = logit + x*(step==k)
        return logit

    def lpl(self,logits, labels):

        # compute split
        quant = self.train_num*0.5
        split = torch.where(self.class_freq>quant,1,0)

        # compute head bound  
        head_dw_steps = torch.ones_like(split)*self.dw_mult

        # compute tail bound
        max_tail = torch.max(self.class_freq*(1-split))
        tail_up_steps = torch.floor(-torch.log(self.class_freq/max_tail)+0.5)*self.up_mult

        logits_head_dw = self.pgd_like(logits, labels, head_dw_steps, -1.0) - logits   # 极小化（正头部，负尾部）
        logits_tail_up = self.pgd_like(logits, labels, tail_up_steps, 1.0) - logits    # 极大化 （正尾 ，负头）

        head = torch.sum(logits_head_dw*labels*split,dim=0)/(torch.sum(labels*split,dim=0)+1e-6)
        tail = torch.sum(logits_tail_up*labels*(1-split),dim=0)/(torch.sum(labels*(1-split),dim=0)+1e-6)

        # compute perturb
        perturb = head+tail

        return perturb.detach()
    
    def lpl_imbalance(self, logits, labels,
                      prop,               # [C]  self.Prop (EMA / batch proportion)
                      nonzero_var_tensor, # [C]  σ_c^(+)
                      zero_var_tensor,    # [C]  σ_c^(-)
                      normalized_sigma_cj,  # [C,C]
                      normalized_ro_cj,     # [C,C]
                      normalized_tao_cj):   # [C,C]

        """
        logits: [B, C, C]
        labels: [B, C, C]
        prop:   [C]        (class proportion)
        nonzero_var_tensor: [C]
        zero_var_tensor:    [C]
        normalized_*:       [C,C], 이미 [0,1]로 정규화된 형태
        """

        C = labels.size(1)
        eps = 1e-6

        # ----- (1) coef_cc: 원본 논문식 버전 -----
        # ratio = σ_pos / σ_neg
        ratio = nonzero_var_tensor / (zero_var_tensor + eps)

        lam  = self.coef_alpha   # λ
        beta = self.coef_beta    # β
        
        # 🔹 2-1) co-occurrence sparsification (weak pair 제거 + top-K만 유지)
        tao = normalized_tao_cj.clone()
        
         # (1) threshold 기반 마스킹
        if self.tao_th is not None and self.tao_th > 0:
            tao = torch.where(tao >= self.tao_th,
                              tao,
                              torch.zeros_like(tao))

        # (2) row-wise top-K (각 c당 상위 K개 co-occ만)
        if self.tao_topk is not None and self.tao_topk > 0:
            # tao: [C, C]
            vals, idx = torch.topk(tao, k=min(self.tao_topk, C), dim=1)
            mask = torch.zeros_like(tao)
            mask.scatter_(1, idx, 1.0)
            tao = tao * mask
        
        normalized_tao_cj = tao
        ratio = nonzero_var_tensor / (zero_var_tensor + eps)

        # 원본 코드 형태:
        # coef_cc = (1-(1-λ)*β)*prop + (1-λ)*β*(σ_pos / σ_neg)
        coef_cc = (1.0 - (1.0 - lam) * beta) * prop \
                  + (1.0 - lam) * beta * ratio        # [C]

        # ----- (2) coef_cj: subclass-wise 계수 (off-diagonal + diag=coef_cc) -----
        # α_cj = λ τ_cj + (1-λ) [ β σ_cj + (1-β) ρ_cj ]
        coef_cj = lam * normalized_tao_cj \
                  + (1.0 - lam) * (beta * normalized_sigma_cj
                                   + (1.0 - beta) * normalized_ro_cj)  # [C,C]

        eye = torch.eye(C, device=logits.device, dtype=coef_cj.dtype)
        coef_cj = coef_cj * (1.0 - eye) + torch.diag(coef_cc)  # 대각선 교체


        # ----- (3) head / tail 분리 (평균 기준 threshold) -----
        quant = torch.sum(coef_cj) / (C * C)

        split = torch.where(coef_cj > quant, 1.0, 0.0)  # head(1) vs tail(0)
        head_mask = split
        tail_mask = 1.0 - split

        # ----- (4) 그룹별(min–max) 정규화 함수 -----
        def _group_minmax(x, mask):
            """
            x:    [C,C]
            mask: [C,C] (0/1)
            반환: mask가 1인 위치만 min–max 정규화된 값, 나머지는 0
            """
            x_masked = x * mask
            if (mask > 0).sum() == 0:
                # 해당 그룹이 아예 없으면 그대로 반환
                return x_masked

            valid = x_masked[mask > 0]
            v_min = valid.min()
            v_max = valid.max()
            scale = (v_max - v_min + eps)

            x_norm = (x_masked - v_min) / scale
            x_norm = torch.clamp(x_norm, min=0.0)
            return x_norm

        # head / tail 각각 자기 그룹 안에서만 0~1로 스케일링
        head_coef = _group_minmax(coef_cj, head_mask)
        tail_coef = _group_minmax(coef_cj, tail_mask)

        _assert_finite(head_coef, "lpl.head_coef@after_norm")
        _assert_finite(tail_coef, "lpl.tail_coef@after_norm")

        # ----- (5) 계수 → PGD step 수 -----
        head_dw_steps = torch.floor(head_coef * self.dw_mult).to(logits.device)
        tail_up_steps = torch.floor(tail_coef * self.up_mult).to(logits.device)

        _assert_finite(head_dw_steps, "lpl.head_dw_steps")
        _assert_finite(tail_up_steps, "lpl.tail_up_steps")

        # ----- (6) PGD-like perturbation -----
        logits_head_dw = self.pgd_like_diff_sign(
            logits, labels, head_dw_steps, -1.0
        ) - logits
        logits_tail_up = self.pgd_like_diff_sign(
            logits, labels, tail_up_steps,  1.0
        ) - logits

        _assert_finite(logits_head_dw, "lpl.logits_head_dw")
        _assert_finite(logits_tail_up, "lpl.logits_tail_up")

        perturb = logits_head_dw + logits_tail_up
        _assert_finite(perturb, "lpl.perturb@output")

        return perturb.detach()



    # def logit_reg_functions(self, labels, logits, norm_prop, nonzero_var_tensor, zero_var_tensor, normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj, weight=None):
    #     if not self.logit_reg:
    #         return logits, weight

    #     if 'init_bias' in self.logit_reg:
    #         batch_size = logits.size(0)
    #         num_classes = logits.size(1)
    #         logits = logits.view(batch_size,num_classes,1).expand(batch_size,num_classes,num_classes).clone()
    #         labels = labels.view(batch_size,num_classes,1).expand(batch_size,num_classes,num_classes).clone()
    #         logits += self.lpl_imbalance(logits, labels, norm_prop,nonzero_var_tensor, zero_var_tensor, normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj)

    #     if 'neg_scale' in self.logit_reg:
    #         logits = logits * (1 - labels) * self.neg_scale  + logits * labels
    #         weight = weight.view(weight.size(0),weight.size(1),1).expand(weight.size(0),weight.size(1),weight.size(1))
    #         weight = weight / self.neg_scale * (1 - labels) + weight * labels

    #     return logits, weight
    
    def logit_reg_functions(self, labels, logits, norm_prop,
                        nonzero_var_tensor, zero_var_tensor,
                        normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj,
                        weight=None):
        """
        labels: [B, C]
        logits: [B, C]
        반환 logits: 항상 [B, C]  (SLP 켜져도 최종은 [B, C]로 접음)
        weight: [B, C] 그대로 유지
        """
        _assert_finite(logits, "logit_reg.logits@input")
        _assert_finite(labels, "logit_reg.labels@input")

        if not self.logit_reg:
            return logits, weight

        # --- SLP perturbation (init_bias 키가 있으면 사용) ---
        if 'init_bias' in self.logit_reg:
            B, C = logits.size()

            # [B,C] → [B,C,C]
            logits_exp = logits.view(B, C, 1).expand(B, C, C).clone()
            labels_exp = labels.view(B, C, 1).expand(B, C, C).clone()

            _assert_finite(norm_prop, "logit_reg.norm_prop@input")
            _assert_finite(nonzero_var_tensor, "logit_reg.nonzero_var@input")
            _assert_finite(zero_var_tensor, "logit_reg.zero_var@input")
            _assert_finite(normalized_sigma_cj, "logit_reg.sigma_cj@input")
            _assert_finite(normalized_ro_cj, "logit_reg.ro_cj@input")
            _assert_finite(normalized_tao_cj, "logit_reg.tao_cj@input")

            perturb = self.lpl_imbalance(
                logits_exp, labels_exp,
                norm_prop, nonzero_var_tensor, zero_var_tensor,
                normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj
            )
            _assert_finite(perturb, "logit_reg.perturb@from_lpl")

            logits_exp = logits_exp + perturb
            _assert_finite(logits_exp, "logit_reg.logits@after_lpl")

            # ⭐ 핵심: subclass 차원 평균 내서 다시 [B, C]로 접는다
            logits = logits_exp.mean(dim=2)
            _assert_finite(logits, "logit_reg.logits@after_mean")

        # --- neg_scale (음성쪽 scale) ---
        if 'neg_scale' in self.logit_reg:
            neg_scale = self.neg_scale
            # 여기서는 항상 logits: [B, C], labels: [B, C]
            lbl = labels

            logits = logits * (1 - lbl) * neg_scale + logits * lbl
            _assert_finite(logits, "logit_reg.logits@after_neg_scale")

        # weight는 건드리지 않는다 (항상 [B,C])
        return logits, weight



    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        repeat_rate = torch.clamp(repeat_rate, min=1e-6)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        # elif 'average_w' in self.CB_mode:
        #     weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
        #               (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        #     weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
        #              torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'average_w' in self.CB_mode:
            # device는 gt_labels 기준으로 맞춰주는 게 안전
            weight_ = (1 - self.CB_beta) / (1 - torch.pow(self.CB_beta, self.class_freq))
            weight_ = weight_.to(gt_labels.device)
            # 각 샘플별 양성 개수
            pos_cnt = torch.sum(gt_labels, dim=1, keepdim=True)
            # 양성 없으면 0 -> 1로 올려서 0으로 나누는 걸 방지
            pos_cnt_safe = pos_cnt.clamp_min(1.0)
            # 원래 수식 유지, 분모만 safe 버전으로 교체
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / pos_cnt_safe
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight

    def data_normal(self, data):
        d_min = torch.min(data)
        d_max = torch.max(data)
        dst = d_max-d_min
        norm_data = torch.div(data-d_min,dst)
        reverse_norm_data = torch.div(d_max-data,dst)
        return norm_data, reverse_norm_data

    def none_zero_normal(self, data):
        ones = torch.ones_like(data)
        d_min = torch.min(torch.where(data==0,ones,data))
        d_max = torch.max(data)
        dst = d_max - d_min
        norm_data =torch.div(data-d_min,dst)
        norm_data =torch.clamp(norm_data,min=0.0)
        reverse_norm_data =torch.div(d_max-data,dst)
        zero = torch.zeros_like(reverse_norm_data)
        reverse_norm_data =torch.where(data>1,zero,data)
        return norm_data, reverse_norm_data
    
    @torch.no_grad()
    def _create_mfm_weight(self, device=None, dtype=torch.float32):
        """
        논문/기존 MultiGrainedFocalLoss와 동일하게
        class_freq로부터 per-class weight 계산.
        """
        dist = self.class_freq.to(device=device, dtype=dtype)  # [C]
        total = dist.sum()
        prob = dist / (total + self.eps)             # p_i
        prob = prob / (prob.max() + self.eps)        # max 정규화
        # (-log p + 1)^(1/6)
        weight = torch.pow(-torch.log(prob.clamp_min(self.eps)) + 1.0, 1.0 / 6)

        self.mfm_weight = weight
        self._mfm_weight_ready = True
    