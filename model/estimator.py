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
        features : [N, A]
        labels   : [N, C] (multi-hot)
        logits   : [N, C]  ← 반드시 각 클래스 c의 축 s_c 로 계산에 사용
        반환값   : Prop, Cov_pos, Cov_neg, Sigma_cj, Ro_cj, Tao_cj (전부 EMA 반영)
        """
        device = logits.device
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        onehot = labels.to(device)

        # ---------- Proportion p_c ----------
        pr_C = onehot.sum(0) / max(N, 1)

        # ---------- σ_c^(+), σ_c^(-) : c-축(s_c) 분산만 ----------
        nonzero_var_list, zero_var_list = [], []
        for c in range(C):
            pos = (onehot[:, c] == 1)
            neg = ~pos

            s_c_pos = logits[pos, c]      # s_c on positive subset
            s_c_neg = logits[neg, c]      # s_c on negative subset

            if s_c_pos.numel() > 1:
                nz = torch.var(s_c_pos, unbiased=False)
            else:
                nz = torch.tensor(0.0, device=device)

            if s_c_neg.numel() > 1:
                zv = torch.var(s_c_neg, unbiased=False)
            else:
                zv = torch.tensor(0.0, device=device)

            nz = torch.nan_to_num(nz, nan=0.0, posinf=0.0, neginf=0.0)
            zv = torch.nan_to_num(zv, nan=0.0, posinf=0.0, neginf=0.0)

            nonzero_var_list.append(nz)
            zero_var_list.append(zv)

        nonzero_var_tensor = torch.stack(nonzero_var_list)  # [C]
        zero_var_tensor    = torch.stack(zero_var_list)     # [C]

        # ---------- σ_cj, ρ_cj, τ_cj : 항상 s_c 를 기준으로 ----------
        n_per_cls = onehot.sum(dim=0).to(torch.float32)  # [C]

        sigma_rows, rho_rows, tau_rows = [], [], []
        for c in range(C):
            pos_c = (onehot[:, c] == 1)

            sigma_j, rho_j, tau_j = [], [], []
            for j in range(C):
                pos_j = (onehot[:, j] == 1)
                co = (pos_c & pos_j)
                co_cnt = co.sum()

                # σ_cj: 공출현(co)에서 s_c 분산
                if co_cnt.item() > 1:
                    s_c_co = logits[co, c]  # **c축만**
                    s_val = torch.var(s_c_co, unbiased=False)
                    s_val = torch.nan_to_num(s_val, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    s_val = torch.tensor(0.0, device=device)

                # ρ_cj = |c∧j| / n_j
                nj = n_per_cls[j]
                rho_val = (co_cnt.to(torch.float32)) / (nj + 1e-7)

                # τ_cj = (N - n_c) / (n_j - |c∧j| + ε)
                nc = n_per_cls[c]
                tau_val = (N - nc) / (nj - co_cnt.to(torch.float32) + 1e-7)

                sigma_j.append(s_val)
                rho_j.append(torch.as_tensor(rho_val, device=device, dtype=torch.float32))
                tau_j.append(torch.as_tensor(tau_val, device=device, dtype=torch.float32))

            sigma_rows.append(torch.stack(sigma_j))  # [C]
            rho_rows.append(torch.stack(rho_j))      # [C]
            tau_rows.append(torch.stack(tau_j))      # [C]

        sigma_mat = torch.stack(sigma_rows, dim=0)  # [C, C]
        rho_mat   = torch.stack(rho_rows,   dim=0)  # [C, C]
        tau_mat   = torch.stack(tau_rows,   dim=0)  # [C, C]

        # ---------- 안정적인 Min–Max 정규화 ----------
        def _minmax(x, eps=1e-6):
            xmin = x.min()
            xmax = x.max()
            return (x - xmin) / (xmax - xmin + eps)

        normalized_sigma_cj = _minmax(sigma_mat)
        normalized_ro_cj    = _minmax(rho_mat)
        normalized_tao_cj   = _minmax(tau_mat)

        # ---------- 아래는 기존 EMA 업데이트 로직(디바이스/NaN 가드 유지) ----------
        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = NxCxFeatures * NxCxA_onehot

        Amount_CxA = NxCxA_onehot.sum(0)  # [C, A]
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA  # [C, A]
        var_temp = features_by_sort - ave_CxA.expand(N, C, A) * NxCxA_onehot
        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ) / Amount_CxA.view(C, A, 1).expand(C, A, A)

        sum_weight_PR     = onehot.sum(0).view(C)                    # [C]
        sum_weight_PR_neg = (N - onehot.sum(0)).view(C)              # [C]
        sum_weight_CV     = onehot.sum(0).view(C, 1, 1).expand(C, A, A)
        sum_weight_AV     = onehot.sum(0).view(C, 1).expand(C, A)

        # co-occurrence 수 기반 EMA weight_CJ
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

        # EMA 가중치들
        def safe_div(num, den):
            return num / (den + 1e-7)

        weight_PR     = safe_div(sum_weight_PR,     sum_weight_PR     + self.Amount.view(C))
        weight_PR_neg = safe_div(sum_weight_PR_neg, sum_weight_PR_neg + self.Amount.view(C))
        weight_CV     = safe_div(sum_weight_CV,     sum_weight_CV     + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_AV     = safe_div(sum_weight_AV,     sum_weight_AV     + self.Amount.view(C, 1).expand(C, A))
        weight_CJ     = safe_div(sum_weight_CJ,     sum_weight_CJ     + self.Amount.view(C))

        weight_PR[weight_PR != weight_PR] = 0
        weight_PR_neg[weight_PR_neg != weight_PR_neg] = 0
        weight_CV[weight_CV != weight_CV] = 0
        weight_AV[weight_AV != weight_AV] = 0
        weight_CJ[weight_CJ != weight_CJ] = 0

        additional_CV = weight_CV * (1 - weight_CV) * torch.bmm(
            (self.Ave - ave_CxA).view(C, A, 1),
            (self.Ave - ave_CxA).view(C, 1, A)
        )

        # EMA 업데이트
        self.Prop       = (self.Prop      * (1 - weight_PR)     + pr_C              * weight_PR    ).detach()
        self.CoVariance = (self.CoVariance* (1 - weight_CV)     + var_temp          * weight_CV    ).detach() + additional_CV.detach()
        self.Ave        = (self.Ave       * (1 - weight_AV)     + ave_CxA           * weight_AV    ).detach()

        self.Cov_pos    = (self.Cov_pos   * (1 - weight_PR)     + nonzero_var_tensor* weight_PR    ).detach()
        self.Cov_neg    = (self.Cov_neg   * (1 - weight_PR_neg) + zero_var_tensor   * weight_PR_neg).detach()

        self.Sigma_cj   = (self.Sigma_cj  * (1 - weight_CJ)     + normalized_sigma_cj*weight_CJ    ).detach()
        self.Ro_cj      = (self.Ro_cj     * (1 - weight_CJ)     + normalized_ro_cj   *weight_CJ    ).detach()
        self.Tao_cj     = (self.Tao_cj    * (1 - weight_CJ)     + normalized_tao_cj  *weight_CJ    ).detach()

        # 누적 샘플수
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