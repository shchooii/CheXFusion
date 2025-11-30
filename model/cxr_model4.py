import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC
from transformers import get_cosine_schedule_with_warmup
from model.layers import FusionBackbone3Head
from model.loss import get_loss
from model.estimator import EstimatorCV
import torch.nn as nn
import torch.nn.functional as F

def _assert_finite(t: torch.Tensor, name: str):
    if not torch.isfinite(t.detach()).all():
        bad = t.detach()
        msg = (
            f"[NaN/Inf DETECTED] {name} has non-finite values: "
            f"min={bad.min().item() if bad.numel() else 'NA'}, "
            f"max={bad.max().item() if bad.numel() else 'NA'}, "
            f"any_nan={(~torch.isfinite(bad)).any().item()}"
        )
        raise AssertionError(msg)


class HeadTailBalancerLoss(nn.Module):
    """
    head, tail, balance(logits) + labels를 받아서
    외부에서 넘겨받은 loss(예: ASL, SLP)를 이용해 HTB 보조 loss를 계산.
    
    - loss는 "배치별 손실 벡터"를 반환하는 형태(reduction='none')여야
      k_h, k_t 가 샘플별로 잘 작동함.
    """
    def __init__(self, loss: nn.Module, gamma: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.loss = loss      # 예: AsymmetricLoss, ResampleLoss2(SLP) 등
        self.gamma = gamma
        self.eps = eps

    def forward(self,
                head: torch.Tensor,      # [B, C]
                tail: torch.Tensor,      # [B, C]
                balance: torch.Tensor,   # [B, C]
                labels: torch.Tensor,    # [B, C]
                slp_stats: tuple = None  # SLP 통계 (Prop, nonzero_var, ...)
                ) -> torch.Tensor:

        # Helper to call loss with correct arguments
        def call_loss(logits, lbl):
            if slp_stats is not None:
                # SLP loss: (*stats, logits, labels, reduction_override='none')
                return self.loss(*slp_stats, logits, lbl, reduction_override='none')
            else:
                # Standard loss: (logits, labels, reduction_override='none')
                # 일부 loss는 reduction_override를 지원하지 않을 수 있으니 확인 필요하지만,
                # get_loss로 가져오는 loss들은 대부분 지원함 (custom loss들).
                # 만약 지원 안하면 reduction='none'으로 init 해야 함.
                return self.loss(logits, lbl, reduction_override='none')

        # 1) head / tail 성능 기반 가중치 k_h, k_t 계산
        with torch.no_grad():
            h_acc = call_loss(head, labels).pow(self.gamma)   # [B]
            t_acc = call_loss(tail, labels).pow(self.gamma)   # [B]
            denom = h_acc + t_acc + self.eps                  # [B]
            k_h = h_acc / denom                               # [B]
            k_t = t_acc / denom                               # [B]

        # 2) 확률 분포로 변환 (원래 코드 그대로 softmax 사용)
        p_h = F.softmax(head, dim=-1)      # [B, C]
        p_t = F.softmax(tail, dim=-1)      # [B, C]
        p_b = F.softmax(balance, dim=-1)   # [B, C]

        # 3) head / tail 분포에 balance 분포를 엮어서 다시 loss 계산
        #    SLP loss를 base로 쓸 때도 이 "확률곱"을 logits 위치에 넣어줌.
        loss_h = call_loss(p_h * p_b, labels)  # [B]
        loss_t = call_loss(p_t * p_b, labels)  # [B]

        # 4) 샘플별로 k_h, k_t 가중해서 평균
        loss = (k_h * loss_h + k_t * loss_t).mean()
        return loss


class HTBAuxiliaryLoss(nn.Module):
    """
    - 외부에서 받은 base_loss (예: SLP)를 HTB 내부에서도 그대로 사용.
    """
    def __init__(self, base_loss: nn.Module, gamma: float = 2.0):
        super().__init__()
        self.htb = HeadTailBalancerLoss(loss=base_loss, gamma=gamma)

    def forward(self,
                head: torch.Tensor,
                tail: torch.Tensor,
                balance: torch.Tensor,
                labels: torch.Tensor,
                slp_stats: tuple = None) -> torch.Tensor:
        return self.htb(head, tail, balance, labels, slp_stats=slp_stats)


class CxrModel4(pl.LightningModule):
    def __init__(
        self,
        lr,
        classes,
        loss_init_args,
        timm_init_args,
        alpha_aux: float = 1.0,   # HTB 보조 loss 비율
        htb_gamma: float = 2.0,
    ):
        super(CxrModel4, self).__init__()
        self.lr = lr
        self.classes = classes
        self.alpha_aux = alpha_aux

        # COMIC Backbone (3-Head)
        self.backbone = FusionBackbone3Head(timm_init_args, "export/convnext_stage1_for_fusion2.pth")

        self.validation_step_outputs = []
        self.val_ap = AveragePrecision(task='binary')
        self.val_auc = AUROC(task="binary")

        # SLP Estimator
        self.cv_estimator = EstimatorCV(feature_num=768, class_num=len(classes))

        # 1) Main Loss (SLP)
        self.criterion_cls = get_loss(**loss_init_args)

        # 2) HTB Aux Loss - Uses the SAME SLP loss instance as base
        self.aux_loss = HTBAuxiliaryLoss(
            base_loss=self.criterion_cls,
            gamma=htb_gamma,
        )

    def forward(self, image):
        # Backbone3Head → (logits_head, logits_tail, logits_bal)
        return self.backbone(image)

    def shared_step(self, batch, batch_idx):
        image, label = batch
        
        # Get 3-head logits + features for SLP
        logits_head, logits_tail, logits_bal, feat = self.backbone.forward_with_features(image)
        _assert_finite(logits_bal, "logits_bal@shared_step")

        # SLP: CV update (using balanced logits as main)
        Prop, nonzero_var, zero_var, Sigma_cj, Ro_cj, Tao_cj = \
            self.cv_estimator.update_CV(feat, label, logits_bal.detach())
        
        slp_stats = (Prop, nonzero_var, zero_var, Sigma_cj, Ro_cj, Tao_cj)

        # 1) Main Loss (SLP applied to logits_bal)
        slp_loss = self.criterion_cls(
            *slp_stats,
            logits_bal, label
        )
        _assert_finite(slp_loss, "slp_loss@shared_step")

        # 2) HTB Aux Loss (SLP applied inside HTB)
        aux_loss = self.aux_loss(logits_head, logits_tail, logits_bal, label, slp_stats=slp_stats)
        
        # Total Loss
        loss = slp_loss + self.alpha_aux * aux_loss

        # Pred for metrics (using balanced logits)
        with torch.no_grad():
            probs = torch.sigmoid(logits_bal)
            _assert_finite(probs, "probs@shared_step")

        return dict(
            loss=loss,
            pred=probs,
            label=label,
        )

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'loss': res['loss'].detach()}, prog_bar=True)
        self.log_dict({'train_loss': res['loss'].detach()},
                      prog_bar=True, on_step=False, on_epoch=True)
        return res['loss']
        
    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'val_loss': res['loss'].detach()}, prog_bar=True)
        self.validation_step_outputs.append(res)

    def on_validation_epoch_end(self):
        preds = torch.cat([x['pred'] for x in self.validation_step_outputs])
        labels = torch.cat([x['label'] for x in self.validation_step_outputs])

        val_ap = []
        val_auroc = []
        C = len(self.classes)
        for i in range(C):
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            val_ap.append(ap)
            val_auroc.append(auroc)
            print(f'{self.classes[i]}_ap: {ap}')
        
        head_idx   = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
        tail_idx   = [7, 11, 17, 18, 19, 21, 23, 25]

        self.log_dict({'val_ap': sum(val_ap)/C}, prog_bar=True)
        self.log_dict({'val_auroc': sum(val_auroc)/C}, prog_bar=True)
        self.log_dict({
            'val_head_ap':   sum([val_ap[i] for i in head_idx])   / len(head_idx),
            'val_medium_ap': sum([val_ap[i] for i in medium_idx]) / len(medium_idx),
            'val_tail_ap':   sum([val_ap[i] for i in tail_idx])   / len(tail_idx),
        }, prog_bar=True)

        self.validation_step_outputs = []
    
    def test_step(self, batch, batch_idx):
        # validation과 거의 동일
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'test_loss': res['loss'].detach()}, prog_bar=True)
        # test용 버퍼에 따로 모으고 싶으면 새 리스트 사용
        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = []
        self.test_step_outputs.append(res)

    def on_test_epoch_end(self):
        preds = torch.cat([x['pred'] for x in self.test_step_outputs])
        labels = torch.cat([x['label'] for x in self.test_step_outputs])

        val_ap = []
        val_auroc = []
        for i in range(26):
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            val_ap.append(ap)
            val_auroc.append(auroc)
            print(f'[TEST] {self.classes[i]}_ap: {ap}')

        head_idx = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
        tail_idx = [7, 11, 17, 18, 19, 21, 23, 25]

        self.log_dict({'test_ap': sum(val_ap)/26}, prog_bar=True)
        self.log_dict({'test_auroc': sum(val_auroc)/26}, prog_bar=True)
        self.log_dict({'test_head_ap': sum([val_ap[i] for i in head_idx]) / len(head_idx)}, prog_bar=True)
        self.log_dict({'test_medium_ap': sum([val_ap[i] for i in medium_idx]) / len(medium_idx)}, prog_bar=True)
        self.log_dict({'test_tail_ap': sum([val_ap[i] for i in tail_idx]) / len(tail_idx)}, prog_bar=True)
        self.test_step_outputs = []

    def predict_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        pred = res['pred']
        image, label = batch
        batch_flip = (image.flip(-1), label)
        pred_flip = self.shared_step(batch_flip, batch_idx)['pred']
        pred = (pred + pred_flip) / 2
        return pred

    def configure_optimizers(self):
        optimizer = AdamW(self.backbone.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 250000)
        return [optimizer], [scheduler]
