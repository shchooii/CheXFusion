import torch
import lightning.pytorch as pl
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AveragePrecision, AUROC
from transformers import get_cosine_schedule_with_warmup

from model.loss import get_loss
from model.layers import FusionBackbone2


# =========================================================
# Eq.(8) Additive Attention Block (성능형 구성)
#   f_b = Attn(f_hat_b, [f_h, f_t]) + f_hat_b
#   - MHA + Residual + LN + FFN + Residual + LN
# =========================================================
class AdditiveEnvAttention(nn.Module):
    def __init__(self, dim=768, nhead=8, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)

        hidden = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, f_hat_b, f_h, f_t):
        """
        f_hat_b: [B, D]
        f_h, f_t: [B, D]
        """
        q = f_hat_b.unsqueeze(1)                 # [B,1,D]
        kv = torch.stack([f_h, f_t], dim=1)      # [B,2,D]

        # attn_out: [B,1,D]
        attn_out, _ = self.mha(query=q, key=kv, value=kv, need_weights=False)
        x = self.ln1(q + self.drop1(attn_out))   # residual + LN

        ffn_out = self.ffn(x)                    # [B,1,D]
        x = self.ln2(x + ffn_out)                # residual + LN
        F_ctx = x.squeeze(1)                     # [B,D]

        # Eq.(8): f_b = Attn(...) + f_hat_b
        f_b = f_ctx + f_hat_b
        return f_b


class CxrModel3(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        classes,
        loss_init_args,
        timm_init_args,

        lambda_b: float = 0.2,
        lambda_irm: float = 1e-3,

        mu: float = 0.9,
        alpha: float = 2.0,
        temperature: float = 2.0,

        # Eq.9/10 params
        rho: float = 0.05,
        eta: float = 1e-6,
        Ng: int = 1,                  # 기본값, 아래에서 head별로 override 가능

        # Eq.9 strength
        plm_gamma: float = 1.0,       # Eq.(9) term을 얼마나 세게 섞을지 (빡세게=1.0부터)

        weight_decay: float = 1e-4,
        warmup_steps: int = 0,
        total_steps: int = 250000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classes"])
        self.classes = classes
        self.lr = lr

        self.backbone = FusionBackbone2(
            timm_init_args=timm_init_args,
            pretrained_path="export/convnext_stage1_for_fusion2.pth",
        )

        self.criterion_cls = get_loss(**loss_init_args)
        self.irm_base = nn.BCEWithLogitsLoss(reduction="mean")

        self.lambda_b = lambda_b
        self.lambda_irm = lambda_irm
        self.alpha = alpha
        self.temperature = temperature

        self.register_buffer("et", torch.zeros(768))
        self.mu = mu

        self.rho = rho
        self.eta = eta
        self.Ng = max(int(Ng), 1)
        self.plm_gamma = float(plm_gamma)
        
        # -------- Eq.(8) env feature 분리 (projection) --------
        # backbone 공유 + projection만 분리 (현실적인 논문 구현 방향)
        self.proj_h = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768, bias=False))
        self.proj_t = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768, bias=False))
        self.proj_b = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768, bias=False))

        # Eq.(8) attention block (성능형)
        self.env_attn = AdditiveEnvAttention(dim=768, nhead=8, dropout=0.1, ffn_mult=4)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.val_ap = AveragePrecision(task="binary")
        self.val_auc = AUROC(task="binary")

        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    # -------------------------
    # pool feature
    # -------------------------
    def _mean_pool_feature(self, x_trans, mask):
        valid = (~mask).float()
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x_trans * valid.unsqueeze(-1)).sum(dim=1) / denom  # [B,768]

    # -------------------------
    # MLDecoder weights (class-wise)
    # -------------------------
    def _get_decoder_weight(self, head: nn.Module):
        W = head.decoder.duplicate_pooling   # [E, D, F]
        E, D, F = W.shape
        W = W.permute(0, 2, 1).contiguous()  # [E,F,D]
        W = W.view(E * F, D)                 # [C,D] (앞에서부터)
        C = head.decoder.num_classes
        return W[:C]                         # [C,768]
    

    # =========================================================
    # ✅ Eq.(9) PLM term
    #   z_x = (rho/Ng) * ( (W f_x)/(||W||+eta) ) * ||f_x||
    #   - 여기서 W: [C,768], f_x:[B,768]
    # =========================================================
    def _logit_eq9(self, f_x, head: nn.Module):
        W = self._get_decoder_weight(head)                 # [C,768]
        W_norm = W.norm(dim=1).clamp(min=0.0) + self.eta   # [C]

        # Ng를 "실제 MLDecoder 그룹 구조"로 더 가깝게: duplicate_factor 사용
        Ng_eff = getattr(head.decoder, "duplicate_factor", self.Ng)
        Ng_eff = max(int(Ng_eff), 1)

        fx_norm = f_x.norm(dim=1, keepdim=True).clamp(min=0.0)  # [B,1]
        proj = (f_x @ W.t()) / W_norm.unsqueeze(0)              # [B,C]
        z9 = (self.rho / Ng_eff) * proj * fx_norm               # [B,C]
        return z9

    # =========================================================
    # Eq.(10) logit adjustment (feature-sim 기반)
    # =========================================================
    def _logit_adjust_eq10(self, z, head, f_ref, sign):
        W = self._get_decoder_weight(head)
        W_norm = W.norm(dim=1) + self.eta

        et = self.et.to(W.device)
        et = et / (et.norm() + self.eta)

        delta_cls = (W @ et) / W_norm  # [C]
        cos = F.cosine_similarity(f_ref, et.unsqueeze(0), dim=1)  # [B]

        Ng_eff = getattr(head.decoder, "duplicate_factor", self.Ng)
        Ng_eff = max(int(Ng_eff), 1)

        delta = (self.rho / Ng_eff) * cos.unsqueeze(1) * delta_cls.unsqueeze(0)
        return z + sign * delta

    # -------------------------
    # HTB distill (sample-wise)
    # -------------------------
    def _distill_bce(self, student_logits, teacher_prob, T):
        loss = F.binary_cross_entropy_with_logits(student_logits / T, teacher_prob, reduction="none")
        return loss.mean(dim=1).mean() * (T * T)

    # -------------------------
    # IRM
    # -------------------------
    def _irm_penalty(self, logits, y):
        w = torch.tensor(1.0, device=logits.device, requires_grad=True)
        risk = self.irm_base(logits * w, y)
        grad = torch.autograd.grad(risk, [w], create_graph=True)[0]
        return grad.pow(2)

    def _irm_loss(self, zh, zt, zb, y):
        return (self._irm_penalty(zh, y) + self._irm_penalty(zt, y) + self._irm_penalty(zb, y)) / 3.0

    def forward(self, image):
        return self.backbone(image)

    # =========================================================
    # shared_step
    # =========================================================
    def shared_step(self, batch, batch_idx, phase: str):
        image, label = batch

        # backbone raw logits + token features
        zh_raw, zt_raw, zb_raw, x_trans, mask = self(image)

        # -----------------------------
        # Eq.(8) env features
        # -----------------------------
        f_base = self._mean_pool_feature(x_trans, mask)  # [B,768]
        f_h = self.proj_h(f_base)
        f_t = self.proj_t(f_base)
        f_hat_b = self.proj_b(f_base)
        f_b = self.env_attn(f_hat_b, f_h, f_t)        # [B,768]  (Eq.8)

        # -----------------------------
        # ✅ Eq.(9) logits (PLM term)
        # -----------------------------
        z9_h = self._logit_eq9(f_h, self.backbone.head_h)      # [B,C]
        z9_t = self._logit_eq9(f_t, self.backbone.head_t)      # [B,C]
        z9_b = self._logit_eq9(f_b, self.backbone.head_b)      # [B,C]

        # raw logits와 결합 (Eq.9 “빡세게 살림”)
        zh = zh_raw + self.plm_gamma * z9_h
        zt = zt_raw + self.plm_gamma * z9_t
        zb = zb_raw + self.plm_gamma * z9_b

        # -----------------------------
        # main loss (balanced)
        # -----------------------------
        loss_cls = self.criterion_cls(zb, label)

        # -----------------------------
        # Eq.(7) e_t update
        #   - zb가 x_trans를 통해 역전파되도록 “그래프 사용”이 보장되는 경로에서 grad 추출
        # -----------------------------
        if phase == "train":
            grad_x = torch.autograd.grad(loss_cls, x_trans, retain_graph=True, create_graph=False)[0]
            grad_base = self._mean_pool_feature(grad_x, mask)      # [B,768]
            grad_fb = self.proj_b(grad_base).detach()              # [B,768]
            self.et.mul_(self.mu).add_(grad_fb.sum(dim=0))

            # 폭주 클립(실전 안정화)
            n = self.et.norm()
            if n > 1e3:
                self.et.mul_(1e3 / (n + 1e-6))

        # -----------------------------
        # Eq.(10) head/tail logit adjustment
        #   - sim은 Eq.(8)의 f_b 기준으로 두는 게 일관적
        # -----------------------------
        zhat_h = self._logit_adjust_eq10(zh, self.backbone.head_h, f_b, sign=+1.0)
        zhat_t = self._logit_adjust_eq10(zt, self.backbone.head_t, f_b, sign=-1.0)

        # -----------------------------
        # Eq.(11) HTB
        # -----------------------------
        T = self.temperature
        p_h = torch.sigmoid(zhat_h / T).detach()
        p_t = torch.sigmoid(zhat_t / T).detach()

        kl_h = self._distill_bce(zb, p_h, T)
        kl_t = self._distill_bce(zb, p_t, T)

        with torch.no_grad():
            lh = self.criterion_cls(zhat_h, label)
            lt = self.criterion_cls(zhat_t, label)
            a = self.alpha
            denom = (lh ** a) + (lt ** a) + 1e-8
            wh = (lh ** a) / denom
            wt = (lt ** a) / denom

        loss_htb = wh * kl_h + wt * kl_t

        # -----------------------------
        # IRM
        # -----------------------------
        if self.lambda_irm > 0 and phase == "train":
            loss_irm = self._irm_loss(zh, zt, zb, label)
        else:
            loss_irm = zb.new_tensor(0.0)

        # -----------------------------
        # total
        # -----------------------------
        loss = loss_cls + self.lambda_b * loss_htb + self.lambda_irm * loss_irm
        pred = torch.sigmoid(zb).detach()

        return {
            "loss": loss,
            "loss_cls": loss_cls.detach(),
            "loss_htb": loss_htb.detach(),
            "loss_irm": loss_irm.detach(),
            "pred": pred,
            "label": label,
        }

    # Lightning hooks (너가 요구한 형식 유지)
    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx, phase='train')
        self.log_dict({"loss": res["loss"].detach()}, prog_bar=True)
        self.log_dict({"train_loss": res["loss"].detach()}, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(
            {
                "train_loss_cls": res["loss_cls"],
                "train_loss_htb": res["loss_htb"],
                "train_loss_irm": res["loss_irm"],
            },
            prog_bar=False,
            on_step=True,
            on_epoch=True,
        )
        return res["loss"]

    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx, phase="val")
        self.log_dict({"val_loss": res["loss"].detach()}, prog_bar=True)
        self.validation_step_outputs.append(res)

    def on_validation_epoch_end(self):
        preds = torch.cat([x["pred"] for x in self.validation_step_outputs])
        labels = torch.cat([x["label"] for x in self.validation_step_outputs])

        val_ap = []
        val_auroc = []
        for i in range(26):
            self.val_ap.reset()
            self.val_auc.reset()
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            val_ap.append(ap)
            val_auroc.append(auroc)
            print(f"{self.classes[i]}_ap: {ap}")

        head_idx = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
        tail_idx = [7, 11, 17, 18, 19, 21, 23, 25]

        self.log_dict(
            {
                "val_ap": sum(val_ap) / 26,
                "val_auroc": sum(val_auroc) / 26,
                "val_head_ap": sum([val_ap[i] for i in head_idx]) / len(head_idx),
                "val_medium_ap": sum([val_ap[i] for i in medium_idx]) / len(medium_idx),
                "val_tail_ap": sum([val_ap[i] for i in tail_idx]) / len(tail_idx),
            },
            prog_bar=True,
        )
        self.validation_step_outputs = []

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        sch = get_cosine_schedule_with_warmup(opt, self.warmup_steps, self.total_steps)
        return [opt], [{"scheduler": sch, "interval": "step"}]
