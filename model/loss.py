import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from model.ce import cross_entropy, binary_cross_entropy, partial_cross_entropy, kpos_cross_entropy
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
    elif type == 'mfm':
        return MultiGrainedFocalLoss()
    elif type == 'focal':
        return FocalLoss()
    elif type == 'wfocal':
        return FocalWithClassWeight(
            gamma=2,
            class_instance_nums=class_instance_nums,
            total_instance_num=total_instance_num
        )
    elif type == 'hill':
        return Hill()

    elif type == 'whill':
        return HillWithClassWeight(
            lamb=1.5, margin=1.0, gamma=2.0,
            class_instance_nums=class_instance_nums,
            total_instance_num=total_instance_num
        )
    elif type == 'cb':
        return ResampleLoss(
            use_sigmoid=True,                 
            reweight_func='CB',               
            weight_norm='by_instance',        
            CB_loss=dict(
                CB_beta=0.9999,               
                CB_mode='by_class'            
            ),
            focal=dict(focal=False, gamma=2, balance_param=2.0),  
            map_param=dict(alpha=10.0, beta=0.2, gamma=0.1),      
            logit_reg=dict(neg_scale=5.0, init_bias=0.1),         
            class_freq=pos_counts,
            neg_class_freq=neg_counts
        )
    elif type == 'db':
        return ResampleLoss(
            use_sigmoid=True,
            reweight_func='rebalance', 
            weight_norm='by_instance',
            map_param=dict(                  
                alpha=10.0,             
                beta=0.2,   
                gamma=0.1
            ),
            focal=dict(focal=False, gamma=2, balance_param=2.0),
            logit_reg=dict(neg_scale=5.0, init_bias=0.1),
            class_freq=pos_counts,
            neg_class_freq=neg_counts
        )
    else:
        raise ValueError(f'Unknown loss type: {type}')


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



class FocalWithClassWeight(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 use_kpos=False,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 gamma=2,
                 balance_param=0.25,
                 class_instance_nums=None,
                 total_instance_num=None):
        super(FocalWithClassWeight, self).__init__()
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

        # class weights (optional)
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

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        # base CE (no reduction)
        logpt = - self.cls_criterion(cls_score, label, weight, reduction='none',
                                     avg_factor=avg_factor)      # (N,C)
        pt = torch.exp(logpt)                                    # (N,C)
        focal = -((1 - pt) ** self.gamma) * logpt                # (N,C)

        # class weight per sign (y*pos + (1-y)*neg)
        if self.pos_w is not None and self.neg_w is not None:
            pos_w = self.pos_w.to(cls_score.device, cls_score.dtype)  # (C,)
            neg_w = self.neg_w.to(cls_score.device, cls_score.dtype)
            cw = label * pos_w + (1 - label) * neg_w                  # (N,C)
            focal = focal * cw

        focal = self.balance_param * focal
        loss = self.loss_weight * focal

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


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


class HillWithClassWeight(nn.Module):
    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,
                 reduction: str = 'mean',
                 class_instance_nums=None,
                 total_instance_num=None) -> None:
        super(HillWithClassWeight, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

        # optional class weights
        if class_instance_nums is not None and total_instance_num is not None:
            cls = torch.as_tensor(class_instance_nums, dtype=torch.float32)
            p = cls / float(total_instance_num)
            pos_w = torch.exp(1 - p)   # w^+
            neg_w = torch.exp(p)       # w^-
            self.register_buffer("pos_w", pos_w)
            self.register_buffer("neg_w", neg_w)
        else:
            self.register_buffer("pos_w", None)
            self.register_buffer("neg_w", None)

    def forward(self, logits, targets):
        eps = 1e-8
        # margin & probs
        logits_margin = logits - self.margin
        p_pos = torch.sigmoid(logits_margin)   # (N,C)
        p_neg = torch.sigmoid(logits)          # (N,C)

        # focal-like weight
        pt = (1 - p_pos) * targets + (1 - targets)   # (N,C)
        focal_w = pt ** self.gamma                   # (N,C)

        # Hill base terms
        pos_term = targets * torch.log(p_pos.clamp(min=eps))                 # (N,C)
        neg_term = (1 - targets) * -(self.lamb - p_neg) * (p_neg ** 2)       # (N,C)

        loss_elem = -(pos_term + neg_term)                                   # (N,C)

        # class weights: cw = y*w+ + (1-y)*w-
        if (self.pos_w is not None) and (self.neg_w is not None):
            pos_w = self.pos_w.to(logits.device, logits.dtype)               # (C,)
            neg_w = self.neg_w.to(logits.device, logits.dtype)               # (C,)
            cw = targets * pos_w + (1 - targets) * neg_w                     # (N,C)
            loss_elem = loss_elem * cw

        # focal after weighting (위치는 상관없음: 모두 스칼라-요소 곱)
        loss_elem = loss_elem * focal_w

        if self.reduction == 'mean':
            return loss_elem.mean()
        elif self.reduction == 'sum':
            return loss_elem.sum()
        else:
            return loss_elem


class ResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 focal=dict(
                     focal=False,
                     balance_param=2.0,
                     gamma=2,
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='by_class'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 class_freq=None, neg_class_freq=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.balance_param = focal['balance_param']

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']
        
        self.eps = 1e-8
        self.class_freq = class_freq.clamp_min(self.eps).cuda()
        self.neg_class_freq = neg_class_freq.clamp_min(self.eps).cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = (self.class_freq + self.neg_class_freq).mean()
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg['neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

        # print('\033[1;35m loading from {} | {} | {} | s\033[0;0m'.format(freq_file, reweight_func, logit_reg))
        # print('\033[1;35m rebalance reweighting mapping params: {:.2f} | {:.2f} | {:.2f} \033[0;0m'.format(self.map_alpha, self.map_beta, self.map_gamma))

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

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = - self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(logpt)
            loss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            loss = ((1 - pt) ** self.gamma) * loss
            loss = self.balance_param * loss
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            base = (1 - self.CB_beta) / (1 - torch.pow(self.CB_beta, self.class_freq.clamp_min(1.)))
            weight = base.cuda().unsqueeze(0).expand_as(gt_labels)
        elif 'average_n' in self.CB_mode:
            denom = torch.sum(gt_labels, dim=1, keepdim=True).clamp_min(self.eps)
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / denom
            weight = (1 - self.CB_beta) / (1 - torch.pow(self.CB_beta, avg_n.clamp_min(1.)))
        elif 'average_w' in self.CB_mode:
            base = (1 - self.CB_beta) / (1 - torch.pow(self.CB_beta,
                                                   self.class_freq.clamp_min(self.eps)))
            base = base.cuda()
            denom = torch.sum(gt_labels, dim=1, keepdim=True).clamp_min(self.eps)
            weight = torch.sum(gt_labels * base, dim=1, keepdim=True) / denom
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