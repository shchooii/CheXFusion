import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC
from transformers import get_cosine_schedule_with_warmup
from model.layers import Backbone, FusionBackbone, Backbone3Head
from model.loss import get_loss
import torch.nn as nn
import torch.nn.functional as F


class HeadTailBalancerLoss(nn.Module):
    """
    head, tail, balance(logits) + labels를 받아서
    외부에서 넘겨받은 loss(예: ASL)를 이용해 HTB 보조 loss를 계산.
    
    - loss는 "배치별 손실 벡터"를 반환하는 형태(reduction='none')여야
      k_h, k_t 가 샘플별로 잘 작동함.
    """
    def __init__(self, loss: nn.Module, gamma: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.loss = loss      # 예: AsymmetricLoss, FocalLoss 등
        self.gamma = gamma
        self.eps = eps

    def forward(self,
                head: torch.Tensor,      # [B, C]
                tail: torch.Tensor,      # [B, C]
                balance: torch.Tensor,   # [B, C]
                labels: torch.Tensor     # [B, C]
                ) -> torch.Tensor:

        # 1) head / tail 성능 기반 가중치 k_h, k_t 계산
        with torch.no_grad():
            # loss(head, labels) → [B] 라고 가정 (reduction='none')
            h_acc = self.loss(head, labels).pow(self.gamma)   # [B]
            t_acc = self.loss(tail, labels).pow(self.gamma)   # [B]
            denom = h_acc + t_acc + self.eps                  # [B]
            k_h = h_acc / denom                               # [B]
            k_t = t_acc / denom                               # [B]

        # 2) 확률 분포로 변환 (원래 코드 그대로 softmax 사용)
        p_h = F.softmax(head, dim=-1)      # [B, C]
        p_t = F.softmax(tail, dim=-1)      # [B, C]
        p_b = F.softmax(balance, dim=-1)   # [B, C]

        # 3) head / tail 분포에 balance 분포를 엮어서 다시 loss 계산
        #    loss는 (logits_or_prob, labels) → [B] 반환한다고 가정
        loss_h = self.loss(p_h * p_b, labels)  # [B]
        loss_t = self.loss(p_t * p_b, labels)  # [B]

        # 4) 샘플별로 k_h, k_t 가중해서 평균
        #    k_h, k_t: [B] → [B, 1]로 브로드캐스트
        loss = (k_h * loss_h + k_t * loss_t).mean()
        return loss


class HTBAuxiliaryLoss(nn.Module):
    """
    - 외부에서 받은 base_loss (예: ASL, ResampleLoss 등)를
      HTB 내부에서도 그대로 사용.
    - forward에서는 단순히 HeadTailBalancerLoss 한 번만 호출.
    - main loss는 모델 쪽에서 base_loss(bal, labels)로 따로 계산.
    """
    def __init__(self, base_loss: nn.Module, gamma: float = 2.0):
        super().__init__()
        self.htb = HeadTailBalancerLoss(loss=base_loss, gamma=gamma)

    def forward(self,
                head: torch.Tensor,
                tail: torch.Tensor,
                balance: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        return self.htb(head, tail, balance, labels)
                        
                        
class CxrModel3(pl.LightningModule):
    def __init__(
        self,
        lr,
        classes,
        loss_init_args,
        timm_init_args,
        alpha_aux: float = 1.0,   # HTB 보조 loss 비율
        htb_gamma: float = 2.0,
    ):
        super(CxrModel3, self).__init__()
        self.lr = lr
        self.classes = classes
        self.alpha_aux = alpha_aux

        # self.backbone = FusionBackbone(...)
        # self.backbone = FusionBackbone3Head(...)
        self.backbone = Backbone3Head(timm_init_args)

        self.validation_step_outputs = []
        self.val_ap = AveragePrecision(task='binary')
        self.val_auc = AUROC(task="binary")

        # --- 기존 loss: BAL logits에만 사용 ---
        #   여기서 ASL, BCE, Resample 등 어떤 것이든 가능
        self.base_criterion = get_loss(**loss_init_args)

        # --- HTB 보조 loss: base_criterion을 그대로 내부에서 재사용 ---
        self.aux_loss = HTBAuxiliaryLoss(
            base_loss=self.base_criterion,
            gamma=htb_gamma,
        )

    def forward(self, image):
        # Backbone3Head → (logits_head, logits_tail, logits_bal)
        return self.backbone(image)

    def shared_step(self, batch, batch_idx):
        image, label = batch
        logits_head, logits_tail, logits_bal = self(image)

        # 1) 기존 main loss (bal만 사용)
        base_loss = self.base_criterion(logits_bal, label)

        # 2) HTB 보조 loss
        aux_loss = self.aux_loss(logits_head, logits_tail, logits_bal, label)

        loss = base_loss + self.alpha_aux * aux_loss

        pred = torch.sigmoid(logits_bal).detach()

        return dict(
            loss=loss,
            pred=pred,
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