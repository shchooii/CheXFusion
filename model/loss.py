import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


def get_loss(type, class_instance_nums, total_instance_num):
    if type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif type == 'wbce':
        return BCEwithClassWeights(class_instance_nums, total_instance_num)
    elif type == 'asl':
        return ASLwithClassWeight(class_instance_nums, total_instance_num)
    elif type == 'apl':
        return APLLoss()
    elif type == 'ral':
        return Ralloss()
    elif type == 'mfm':
        return MultiGrainedFocalLoss() 
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

        # self.spls_loss = SPLC(batch_size=32)

    @torch.no_grad()                  
    def create_weight(self, distribution):
        dist = torch.as_tensor(distribution, dtype=torch.float32, device='cuda')
        total = dist.sum()
        prob = dist / total
        prob = prob / (prob.max() + self.eps) 
        weight = torch.pow(-torch.log(prob.clamp_min(self.eps)) + 1.0, 1.0 / 6)
        self.weight = weight.cuda().detach()
        

    @torch.no_grad()                  
    def create_co_occurrence_matrix(self, co_occurrence_matrix):
        co_occurrence_matrix = torch.tensor(np.load(co_occurrence_matrix)).cuda()
        self.co_occurrence_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)


    def forward(self, x, y):
        
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
    