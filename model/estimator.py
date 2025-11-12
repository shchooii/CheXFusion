import torch
import torch.nn as nn
import numpy as np

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()
        self.Prop = torch.zeros(class_num).cuda()
        self.Cov_pos = torch.zeros(class_num).cuda()
        self.Cov_neg = torch.zeros(class_num).cuda()
        self.Sigma_cj = torch.zeros(class_num, class_num).cuda()
        self.Ro_cj = torch.zeros(class_num, class_num).cuda()
        self.Tao_cj = torch.zeros(class_num, class_num).cuda()


    def update_CV(self, features, labels, logits):
        """
        통계 업데이트 (그림/정의와 정합):
        - Prop: p_i = n_i / N
        - Cov_pos[i]: class i 양성(c=1) 집합에서의 logits 분산
        - Cov_neg[i]: class i 음성(c=0) 집합에서의 logits 분산
        - Sigma_cj[i,j]: i와 j가 동시에 1인 샘플들의 logits 분산
        - Ro_cj[i,j]: |i∧j| / n_j
        - Tao_cj[i,j]: (N - n_i) / (n_j - |i∧j|)
        모든 행렬은 [C,C]로 만들고 루프 밖에서 min–max 정규화.
        """
        device = logits.device
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        # ------- proportion p_i -------
        onehot = labels.to(device)
        pr_C = onehot.sum(0) / max(N, 1)

        # ------- per-class variance (nonzero / zero) on logits -------
        nonzero_var_list, zero_var_list = [], []
        for c in range(C):
            pos_mask = (onehot[:, c] == 1)
            neg_mask = ~pos_mask

            if pos_mask.sum() > 1:
                nz = torch.var(logits[pos_mask, :], unbiased=False)
                nz = torch.nan_to_num(nz, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                nz = torch.tensor(0.0, device=device)

            if neg_mask.sum() > 1:
                zv = torch.var(logits[neg_mask, :], unbiased=False)
                zv = torch.nan_to_num(zv, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                zv = torch.tensor(0.0, device=device)

            nonzero_var_list.append(nz)
            zero_var_list.append(zv)

        nonzero_var_tensor = torch.stack(nonzero_var_list, dim=0)  # [C]
        zero_var_tensor    = torch.stack(zero_var_list,  dim=0)    # [C]

        # ------- co-occurrence matrices: sigma/rho/tau (C x C) -------
        n_per_cls = onehot.sum(dim=0).to(torch.float32)  # [C]
        sigma_rows, rho_rows, tau_rows = [], [], []

        for i in range(C):
            mi = (onehot[:, i] == 1)
            ni = n_per_cls[i]

            sigma_j, rho_j, tau_j = [], [], []
            for j in range(C):
                mj = (onehot[:, j] == 1)
                nj = n_per_cls[j]
                co_mask = (mi & mj)
                co_cnt = co_mask.sum()

                # sigma_ij: 공출현 샘플 logits 분산 (없으면 0)
                if co_cnt > 1:
                    s_val = torch.var(logits[co_mask, :], unbiased=False)
                    s_val = torch.nan_to_num(s_val, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    s_val = torch.tensor(0.0, device=device)

                # rho_ij = |i∧j| / n_j
                rho_val = (co_cnt.to(torch.float32)) / (nj + 1e-7)
                # tao_ij = (N - n_i) / (n_j - |i∧j|)
                tao_val = (N - ni) / (nj - co_cnt.to(torch.float32) + 1e-7)

                sigma_j.append(s_val)
                rho_j.append(torch.as_tensor(rho_val, device=device, dtype=torch.float32))
                tau_j.append(torch.as_tensor(tao_val, device=device, dtype=torch.float32))

            sigma_rows.append(torch.stack(sigma_j))  # [C]
            rho_rows.append(torch.stack(rho_j))      # [C]
            tau_rows.append(torch.stack(tau_j))      # [C]

        sigma_mat = torch.stack(sigma_rows, dim=0).to(device=device, dtype=torch.float32)  # [C,C]
        rho_mat   = torch.stack(rho_rows,   dim=0).to(device=device, dtype=torch.float32)  # [C,C]
        tau_mat   = torch.stack(tau_rows,   dim=0).to(device=device, dtype=torch.float32)  # [C,C]

        # ------- stable min–max normalization -------
        def _minmax(x, eps=1e-6):
            xmin = x.min()
            xmax = x.max()
            return (x - xmin) / (xmax - xmin + eps)

        normalized_sigma_cj = _minmax(sigma_mat)
        normalized_ro_cj    = _minmax(rho_mat)
        normalized_tao_cj   = _minmax(tau_mat)

        # ===== 기존 EMA 누적 로직(특징/공분산/평균) =====
        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = NxCxFeatures * NxCxA_onehot

        Amount_CxA = NxCxA_onehot.sum(0)               # [C, A]
        Amount_CxA[Amount_CxA == 0] = 1
        ave_CxA = features_by_sort.sum(0) / Amount_CxA # [C, A]

        var_temp = features_by_sort - ave_CxA.expand(N, C, A) * NxCxA_onehot
        var_temp = torch.bmm(                                  # [C, A, A]
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ) / Amount_CxA.view(C, A, 1).expand(C, A, A)

        sum_weight_PR     = onehot.sum(0).view(C)                    # [C]
        sum_weight_PR_neg = (N - onehot.sum(0)).view(C)              # [C]
        sum_weight_CV     = onehot.sum(0).view(C, 1, 1).expand(C, A, A)
        sum_weight_AV     = onehot.sum(0).view(C, 1).expand(C, A)

        # [C,C] co-occurrence count for EMA weight_CJ
        sum_weight_CJ = torch.zeros(C, C, device=device)
        for i in range(C):
            for j in range(i, C):
                if i == j:
                    v = (onehot[:, i] == 1).sum()
                    sum_weight_CJ[i, j] = v
                else:
                    v = ((onehot[:, i] == 1) & (onehot[:, j] == 1)).sum()
                    sum_weight_CJ[i, j] = v
                    sum_weight_CJ[j, i] = v

        # ------ EMA weights (with NaN guards) ------
        weight_PR = sum_weight_PR / (sum_weight_PR + self.Amount.view(C))
        weight_PR[weight_PR != weight_PR] = 0

        weight_PR_neg = sum_weight_PR_neg / (sum_weight_PR_neg + self.Amount.view(C))
        weight_PR_neg[weight_PR_neg != weight_PR_neg] = 0

        weight_CV = sum_weight_CV / (sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV / (sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0

        weight_CJ = sum_weight_CJ / (sum_weight_CJ + self.Amount.view(C))
        weight_CJ[weight_CJ != weight_CJ] = 0

        # ------ additional_CV (이전 로직 유지) ------
        additional_CV = weight_CV * (1 - weight_CV) * torch.bmm(
            (self.Ave - ave_CxA).view(C, A, 1),
            (self.Ave - ave_CxA).view(C, 1, A)
        )

        # ------ EMA updates ------
        self.Prop = (self.Prop * (1 - weight_PR) + pr_C * weight_PR).detach()
        self.CoVariance = (self.CoVariance * (1 - weight_CV) + var_temp * weight_CV).detach() + additional_CV.detach()
        self.Ave = (self.Ave * (1 - weight_AV) + ave_CxA * weight_AV).detach()

        self.Cov_pos = (self.Cov_pos * (1 - weight_PR) + nonzero_var_tensor * weight_PR).detach()
        self.Cov_neg = (self.Cov_neg * (1 - weight_PR_neg) + zero_var_tensor * weight_PR_neg).detach()

        # co-occurrence matrices (정규화된 값으로 EMA)
        self.Sigma_cj = (self.Sigma_cj * (1 - weight_CJ) + normalized_sigma_cj * weight_CJ).detach()
        self.Ro_cj    = (self.Ro_cj    * (1 - weight_CJ) + normalized_ro_cj    * weight_CJ).detach()
        self.Tao_cj   = (self.Tao_cj   * (1 - weight_CJ) + normalized_tao_cj   * weight_CJ).detach()

        # 샘플 수 누적
        self.Amount += onehot.sum(0)

        return self.Prop, self.Cov_pos, self.Cov_neg, self.Sigma_cj, self.Ro_cj, self.Tao_cj

    # def update_CV(self, features, labels, logits): #features 128,640; label:128 


    #     N = features.size(0)
    #     C = self.class_num
    #     A = features.size(1)
    #     device = logits.device

    #     NxCxFeatures = features.view(
    #         N, 1, A
    #     ).expand(
    #         N, C, A
    #     )
    #     onehot = labels



    #     NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    #     features_by_sort = NxCxFeatures.mul(NxCxA_onehot)#对应位置的feature

    #     Amount_CxA = NxCxA_onehot.sum(0) #每一类的数量
    #     Amount_CxA[Amount_CxA == 0] = 1 #20*256


    #     pr_C = onehot.sum(0) / N


    #     nonzero_var_list=[]
    #     zero_var_list=[]
    #     sigma_cj=[]
    #     ro_cj=[]
    #     tao_cj=[]
    #     for c in range(labels.size(1)):
    #         mask_c = labels[:,c]==1
    #         sigma_j=[]
    #         ro_j=[]
    #         tao_j=[]
    #         for j in range(labels.size(1)):
    #             mask_j = labels[:,j]==1
    #             result = logits[mask_c & mask_j, :]
    #             if result.numel() <= 1:
    #                 sigma_val = torch.tensor(0.0, device=logits.device)
    #             else:
    #                 sigma_val = torch.var(result, unbiased=False)
    #                 sigma_val = torch.nan_to_num(sigma_val, nan=0.0, posinf=0.0, neginf=0.0)
                
    #             sigma_j.append(sigma_val)
    #             ro_j.append(result.size(0)/(onehot.sum(0)[j]+torch.tensor(0.0000001)))
    #             tao_j.append((torch.tensor(N)-onehot.sum(0)[c])/(onehot.sum(0)[j]-torch.tensor(result.size(0))+torch.tensor(0.00000001)))

    #         sigma_cj.append(sigma_j)
    #         # sigma_cj_ = torch.tensor(np.array(sigma_cj).tolist())
    #         sigma_cj_ = torch.tensor(sigma_cj, device=features.device, dtype=torch.float32)
    #         if isinstance(sigma_cj, torch.Tensor):
    #             sigma_cj_ = sigma_cj.detach().cpu()
    #         else:
    #             sigma_cj_ = torch.tensor(sigma_cj)

    #         max_val_sigma = torch.max(sigma_cj_)
    #         min_val_sigma = torch.min(sigma_cj_)
    #         normalized_sigma_cj = [(t-min_val_sigma)/(max_val_sigma - min_val_sigma+0.0000001) for t in sigma_cj_]
    #         ro_cj.append(ro_j)
    #         # ro_cj_ = torch.tensor(np.array(ro_cj).tolist())
    #         ro_cj_    = torch.tensor(ro_cj,    device=features.device, dtype=torch.float32)        
    #         max_val_ro = torch.max(ro_cj_)
    #         min_val_ro = torch.min(ro_cj_)
    #         normalized_ro_cj = [(t-min_val_ro)/(max_val_ro - min_val_ro+0.0000001) for t in ro_cj_]
    #         tao_cj.append(tao_j)
    #         # tao_cj_ = torch.tensor(np.array(tao_cj).tolist())
    #         tao_cj_   = torch.tensor(tao_cj,   device=features.device, dtype=torch.float32)
    #         max_val_tao = torch.max(tao_cj_)
    #         min_val_tao = torch.min(tao_cj_)
    #         normalized_tao_cj = [(t-min_val_tao)/(max_val_tao - min_val_tao+0.0000001) for t in tao_cj_]

    #         nonzero_indices = torch.nonzero(labels[:, c])#与c共现
    #         not_labels = torch.logical_not(labels)
    #         zero_indices = torch.nonzero(not_labels[:, c])

    #         # if nonzero_indices.shape[0]==0:
    #         #     nonzero_var = torch.tensor(0).cuda()
    #         # else:
    #         #     nonzero_var = torch.var(logits[nonzero_indices.squeeze()])
    #         # nonzero_var_list.append(nonzero_var)
    #         # zero_var = torch.var(logits[zero_indices.squeeze()])
    #         # zero_var_list.append(zero_var)
            
    #         # ✅ nonzero_var (공존)
    #         if nonzero_indices.numel() > 1:
    #             nonzero_var = torch.var(logits[nonzero_indices.squeeze()], unbiased=False)
    #             nonzero_var = torch.nan_to_num(nonzero_var, nan=0.0, posinf=0.0, neginf=0.0)
    #         else:
    #             nonzero_var = torch.tensor(0.0, device=device)

    #         # ✅ zero_var (비공존)
    #         if zero_indices.numel() > 1:
    #             zero_var = torch.var(logits[zero_indices.squeeze()], unbiased=False)
    #             zero_var = torch.nan_to_num(zero_var, nan=0.0, posinf=0.0, neginf=0.0)
    #         else:
    #             zero_var = torch.tensor(0.0, device=device)

    #         nonzero_var_list.append(nonzero_var)
    #         zero_var_list.append(zero_var)
            
            
    #     nonzero_var_tensor = torch.stack(nonzero_var_list,dim=0)
    #     zero_var_tensor = torch.stack(zero_var_list,dim=0)

    #     ave_CxA = features_by_sort.sum(0) / Amount_CxA #20,256 #平均特征

    #     var_temp = features_by_sort - \
    #                ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

    #     var_temp = torch.bmm(
    #         var_temp.permute(1, 2, 0),
    #         var_temp.permute(1, 0, 2)
    #     ).div(Amount_CxA.view(C, A, 1).expand(C, A, A)) #10，640，640

    #     sum_weight_PR = onehot.sum(0).view(C)

    #     sum_weight_PR_neg = N - sum_weight_PR 

    #     sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

    #     sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A) #mini-batch中每类样本数量

    #     sum_weight_CJ = torch.zeros(onehot.shape[1],onehot.shape[1])
    #     for i in range(C):
    #         for j in range(i, C):
    #             if i == j:
    #                 sum_weight_CJ[i, j] = (onehot[:, i] == 1).sum()
    #             else:
    #                 sum_weight_CJ[i, j] = (onehot[:, i] == 1).logical_and(onehot[:, j] == 1).sum()
    #                 sum_weight_CJ[j, i] = sum_weight_CJ[i, j]
    #     sum_weight_CJ = sum_weight_CJ.cuda()

    #     weight_PR = sum_weight_PR.div(
    #         sum_weight_PR + self.Amount.view(C)
    #     )

    #     weight_PR[weight_PR != weight_PR] = 0

    #     weight_PR_neg = sum_weight_PR_neg.div(
    #         sum_weight_PR_neg + self.Amount.view(C)
    #     )

    #     weight_PR_neg[weight_PR_neg != weight_PR_neg] = 0

    #     weight_CV = sum_weight_CV.div(
    #         sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)#m/n+m  
    #     )
    #     weight_CV[weight_CV != weight_CV] = 0

    #     weight_AV = sum_weight_AV.div(
    #         sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
    #     )
    #     weight_AV[weight_AV != weight_AV] = 0

    #     weight_CJ = sum_weight_CJ.div(
    #         sum_weight_CJ + self.Amount.view(C)
    #     )
    #     weight_CJ[weight_CJ != weight_CJ] = 0

    #     additional_CV = weight_CV.mul(1 - weight_CV).mul(
    #         torch.bmm(
    #             (self.Ave - ave_CxA).view(C, A, 1),
    #             (self.Ave - ave_CxA).view(C, 1, A)
    #         )
    #     )
        
    #     self.Prop = (self.Prop.mul(1 - weight_PR)+ pr_C.mul(weight_PR)).detach()
    #     self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
    #                   .mul(weight_CV)).detach() + additional_CV.detach() #10,640,640

    #     self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach() #10,640

    #     self.Cov_pos = (self.Cov_pos.mul(1 - weight_PR)+ nonzero_var_tensor.mul(weight_PR)).detach()
    #     self.Cov_neg = (self.Cov_neg.mul(1 - weight_PR_neg)+ zero_var_tensor.mul(weight_PR_neg)).detach()

      
    #     self.Amount += onehot.sum(0)
    #     normalized_sigma_cj = torch.stack(normalized_sigma_cj,dim=1).cuda()
    #     normalized_ro_cj = torch.stack(normalized_ro_cj,dim=1).cuda()
    #     normalized_tao_cj = torch.stack(normalized_tao_cj,dim=1).cuda()

    #     self.Sigma_cj = (self.Sigma_cj.mul(1 - weight_CJ)+ normalized_sigma_cj.mul(weight_CJ)).detach()
    #     self.Ro_cj = (self.Ro_cj.mul(1 - weight_CJ)+ normalized_ro_cj.mul(weight_CJ)).detach()
    #     self.Tao_cj = (self.Tao_cj.mul(1 - weight_CJ)+ normalized_tao_cj.mul(weight_CJ)).detach()
        


    #     return  self.Prop, self.Cov_pos, self.Cov_neg,self.Sigma_cj, self.Ro_cj, self.Tao_cj