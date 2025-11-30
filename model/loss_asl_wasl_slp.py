import torch
import torch.nn as nn


# -------------------------------------------------------------------
# 유틸: NaN / Inf 디버깅용 (원하면 끄거나 주석 처리해도 됨)
# -------------------------------------------------------------------
def _assert_finite(x: torch.Tensor, name: str):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"[{name}] has non-finite values (NaN or Inf).")


# -------------------------------------------------------------------
# SLP용 Logit Regularizer (logit만 perturb)
#   - ResampleLoss2의 lpl_imbalance + logit_reg_functions를
#     "loss와 분리된 모듈"로 뽑아낸 버전이라고 보면 됨.
# -------------------------------------------------------------------
class SLPLogitReg(nn.Module):
    def __init__(
        self,
        coef_alpha: float = 1.0 / 3.0,  # λ
        coef_beta: float = 0.7,         # β
        up_mult: float = 12.0,          # tail(weak) up-step 배수
        dw_mult: float = 9.0,           # head(strong) down-step 배수
        tao_th: float = None,           # co-occ τ_cj threshold (None이면 안 씀)
        tao_topk: int = None,           # 각 class별 top-k co-occ만 사용 (None이면 안 씀)
        eps: float = 1e-6,
    ):
        super().__init__()
        self.coef_alpha = coef_alpha
        self.coef_beta = coef_beta
        self.up_mult = up_mult
        self.dw_mult = dw_mult
        self.tao_th = tao_th
        self.tao_topk = tao_topk
        self.eps = eps

    def pgd_like_diff_sign(self, x, y, step, sign):
        """
        x:    [B, C, C]
        y:    [B, C, C]
        step: [C, C]
        sign: +1 or -1
        """
        y = y.to(torch.float32)
        iters = int(torch.max(step).item() + 1)
        logit = torch.zeros_like(x)

        for k in range(iters):
            grad = torch.sigmoid(x) - y
            x = x + grad * sign / x.shape[1]
            logit = logit + x * (step == k)

        return logit

    def _group_minmax(self, x: torch.Tensor, mask: torch.Tensor, eps: float):
        """
        x:    [C, C]
        mask: [C, C] (0/1)
        mask == 1인 원소들만 min-max 정규화해서 0~1 스케일로.
        """
        x_masked = x * mask
        if (mask > 0).sum() == 0:
            return x_masked

        valid = x_masked[mask > 0]
        v_min = valid.min()
        v_max = valid.max()
        scale = (v_max - v_min + eps)

        x_norm = (x_masked - v_min) / scale
        x_norm = torch.clamp(x_norm, min=0.0)
        return x_norm

    def _sparsify_tao(self, tao: torch.Tensor):
        """
        co-occurrence(τ_cj)를 좀 더 sparse하게 만드는 옵션:
        - tao_th: threshold 아래는 0
        - tao_topk: 각 row(c)별로 top-k만 남김
        """
        eps = self.eps
        C = tao.size(0)

        # (1) threshold
        if self.tao_th is not None and self.tao_th > 0:
            tao = torch.where(tao >= self.tao_th, tao, torch.zeros_like(tao))

        # (2) row-wise top-k
        if self.tao_topk is not None and self.tao_topk > 0:
            k = min(self.tao_topk, C)
            vals, idx = torch.topk(tao, k=k, dim=1)
            mask = torch.zeros_like(tao)
            mask.scatter_(1, idx, 1.0)
            tao = tao * mask

        # 필요하면 다시 0~1 scale로
        if (tao > 0).any():
            valid = tao[tao > 0]
            tmin = valid.min()
            tmax = valid.max()
            if tmax > tmin:
                tao = (tao - tmin) / (tmax - tmin + eps)
                tao = torch.clamp(tao, min=0.0)

        return tao

    def forward(
        self,
        logits: torch.Tensor,              # [B, C]
        labels: torch.Tensor,              # [B, C]
        prop: torch.Tensor,                # [C]   (class proportion)
        nonzero_var_tensor: torch.Tensor,  # [C]   σ_c^(+)
        zero_var_tensor: torch.Tensor,     # [C]   σ_c^(-)
        normalized_sigma_cj: torch.Tensor, # [C,C]
        normalized_ro_cj: torch.Tensor,    # [C,C]
        normalized_tao_cj: torch.Tensor,   # [C,C]
    ) -> torch.Tensor:
        """
        반환: perturbed_logits [B, C]
        """

        B, C = logits.size()
        eps = self.eps
        lam = self.coef_alpha
        beta = self.coef_beta

        # ----- (0) co-occurrence sparsify (옵션) -----
        tao = normalized_tao_cj
        if self.tao_th is not None or self.tao_topk is not None:
            tao = self._sparsify_tao(tao)
        normalized_tao_cj = tao
        _assert_finite(normalized_tao_cj, "SLP.tao_cj@after_sparsify")

        # ----- (1) coef_cc: class-wise 계수 -----
        ratio = nonzero_var_tensor / (zero_var_tensor + eps)  # σ_pos / σ_neg
        coef_cc = (1.0 - (1.0 - lam) * beta) * prop \
                  + (1.0 - lam) * beta * ratio        # [C]
        _assert_finite(coef_cc, "SLP.coef_cc")

        # ----- (2) coef_cj: subclass-wise 계수 -----
        # α_cj = λ τ_cj + (1-λ)[β σ_cj + (1-β) ρ_cj]
        coef_cj = lam * normalized_tao_cj \
                  + (1.0 - lam) * (
                      beta * normalized_sigma_cj
                      + (1.0 - beta) * normalized_ro_cj
                  )  # [C, C]

        eye = torch.eye(C, device=logits.device, dtype=coef_cj.dtype)
        coef_cj = coef_cj * (1.0 - eye) + torch.diag(coef_cc)
        _assert_finite(coef_cj, "SLP.coef_cj@after_diag")

        # ----- (3) 평균 기준 head / tail split -----
        quant = torch.sum(coef_cj) / (C * C)
        split = torch.where(coef_cj > quant, 1.0, 0.0)
        head_mask = split
        tail_mask = 1.0 - split

        # ----- (4) 그룹별 min–max 정규화 -----
        head_coef = self._group_minmax(coef_cj, head_mask, eps)
        tail_coef = self._group_minmax(coef_cj, tail_mask, eps)
        _assert_finite(head_coef, "SLP.head_coef@after_norm")
        _assert_finite(tail_coef, "SLP.tail_coef@after_norm")

        # ----- (5) 계수 → PGD step 수 -----
        head_dw_steps = torch.floor(head_coef * self.dw_mult).to(logits.device)
        tail_up_steps = torch.floor(tail_coef * self.up_mult).to(logits.device)
        _assert_finite(head_dw_steps, "SLP.head_dw_steps")
        _assert_finite(tail_up_steps, "SLP.tail_up_steps")

        # ----- (6) PGD-like perturbation -----
        logits_exp = logits.view(B, C, 1).expand(B, C, C).clone()
        labels_exp = labels.view(B, C, 1).expand(B, C, C).clone()

        logits_head_dw = self.pgd_like_diff_sign(
            logits_exp, labels_exp, head_dw_steps, -1.0
        ) - logits_exp
        logits_tail_up = self.pgd_like_diff_sign(
            logits_exp, labels_exp, tail_up_steps,  1.0
        ) - logits_exp

        perturb = logits_head_dw + logits_tail_up
        _assert_finite(perturb, "SLP.perturb")

        logits_exp = logits_exp + perturb
        _assert_finite(logits_exp, "SLP.logits@after_perturb")

        # subclass 축 평균으로 다시 [B, C]로 접기
        logits_perturbed = logits_exp.mean(dim=2)
        _assert_finite(logits_perturbed, "SLP.logits_perturbed")

        return logits_perturbed


# -------------------------------------------------------------------
# WASL (ASL + class weight) + optional SLP
#   - use_slp=False  → 기존 WASL과 동일
#   - use_slp=True   → logits를 SLP로 한 번 튕긴 후 ASL 계산
# -------------------------------------------------------------------
class ASLwithClassWeight2(nn.Module):
    """
    WASL:
      - Asymmetric Loss (ASL)
      - + class-instance 기반 고정 weight (pos/neg)
      - + (옵션) SLP를 통한 logit perturbation

    Args
    ----
    class_instance_nums: list[int] or 1D Tensor, [C]
    total_instance_num:  int, 전체 샘플 수
    gamma_neg, gamma_pos, clip, eps: ASL 하이퍼파라미터

    use_slp: bool
      - True  => SLPLogitReg 사용 (forward에서 통계 인자 필요)
      - False => SLP 없이 순수 WASL
    slp_cfg: dict or None
      - SLPLogitReg에 전달할 하이퍼파라미터 dict
        (coef_alpha, coef_beta, up_mult, dw_mult, tao_th, tao_topk, eps ...)
    """

    def __init__(
        self,
        class_instance_nums,
        total_instance_num,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        use_slp: bool = False,
        slp_cfg: dict = None,
    ):
        super().__init__()

        # class별 비율 p_c
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / float(total_instance_num)

        # 간단한 exp 기반 class weight
        pos_weights = torch.exp(1.0 - p)  # rare class → 큰 weight
        neg_weights = torch.exp(p)        # head class → 음성 weight 더 큼

        # register_buffer로 device 이동 자동 관리
        self.register_buffer("pos_weights", pos_weights)
        self.register_buffer("neg_weights", neg_weights)

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

        self.use_slp = use_slp
        if use_slp:
            if slp_cfg is None:
                # CXR-LT 용 기본값 (필요하면 바꿔서 전달)
                slp_cfg = dict(
                    coef_alpha=1.0 / 3.0,
                    coef_beta=0.7,
                    up_mult=12.0,
                    dw_mult=9.0,
                    tao_th=None,   # 예: 0.15 로 두면 weak co-occ 제거
                    tao_topk=None, # 예: 3 으로 두면 class당 top-3 pair만
                    eps=1e-6,
                )
            self.slp_reg = SLPLogitReg(**slp_cfg)
        else:
            self.slp_reg = None
    
    def forward(
        self,
        norm_prop: torch.Tensor = None,                # [C]
        nonzero_var_tensor: torch.Tensor = None,       # [C]
        zero_var_tensor: torch.Tensor = None,          # [C]
        normalized_sigma_cj: torch.Tensor = None,      # [C,C]
        normalized_ro_cj: torch.Tensor = None,         # [C,C]
        normalized_tao_cj: torch.Tensor = None,        # [C,C]
        pred: torch.Tensor = None,                     # [B, C]
        label: torch.Tensor = None,                    # [B, C]
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ) -> torch.Tensor:

        # --------------------------------------------------------
        # 1) SLP로 logit perturb (옵션)
        #    - CxrModel2 / ResampleLoss2와 동일하게 통계값 먼저 받고,
        #      마지막 두 인자로 logits(pred), labels 들어온다고 가정.
        # --------------------------------------------------------
        if self.use_slp and self.slp_reg is not None:
            pred = self.slp_reg(
                pred,
                label,
                norm_prop,
                nonzero_var_tensor,
                zero_var_tensor,
                normalized_sigma_cj,
                normalized_ro_cj,
                normalized_tao_cj,
            )

        # --------------------------------------------------------
        # 2) WASL: class weight 계산 (pos/neg)
        # --------------------------------------------------------
        w_pos = self.pos_weights.to(pred.device)  # [C]
        w_neg = self.neg_weights.to(pred.device)  # [C]

        weight_cls = label * w_pos + (1.0 - label) * w_neg  # [B, C]
        _assert_finite(weight_cls, "ASL.WASL2.weight")

        # --------------------------------------------------------
        # 3) ASL 본체
        # --------------------------------------------------------
        xs_pos = torch.sigmoid(pred)
        xs_pos = xs_pos.clamp(min=self.eps, max=1.0 - self.eps)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = xs_neg.add(self.clip).clamp(max=1.0)

        los_pos = label * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - label) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg  # [B, C]

        loss = loss * weight_cls

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * label
            pt1 = xs_neg * (1.0 - label)
            pt = pt0 + pt1

            one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1.0 - label)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        loss = -loss.mean()

        return loss
