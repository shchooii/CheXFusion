import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC
from transformers import get_cosine_schedule_with_warmup

from model.layers import FusionBackboneMultiView2 # 위 FusionBackbone로 교체
from model.loss import get_loss

# multi-veiw with htb
class CxrModel6(pl.LightningModule):
    def __init__(
        self,
        lr,
        classes,
        loss_init_args,
        timm_init_args,
        lambda_pa=0.3,          # PA head supervised loss 비율
        lambda_lat=0.3,         # LAT head supervised loss 비율        # tail-positive consistency
        lambda_htb_view=0.1,    # view-HTB distillation 비율
        htb_alpha=2.0,          # κ 계산용 exponent
    ):
        super(CxrModel6, self).__init__()
        self.lr = lr
        self.classes = classes

        self.lambda_pa = lambda_pa
        self.lambda_lat = lambda_lat
        self.lambda_htb_view = lambda_htb_view
        self.htb_alpha = htb_alpha

        # backbone: CheXFusion + MLDecoder head
        self.backbone = FusionBackboneMultiView2(
            timm_init_args,
            pretrained_path='export/convnext_stage1_for_fusion2.pth',
            num_classes=len(classes),
        )

        self.validation_step_outputs = []
        self.val_ap = AveragePrecision(task='binary')
        self.val_auc = AUROC(task="binary")

        # 메인 loss 함수 (WASL / ASL+WBCE / MFM 등 config에 따라)
        self.criterion_cls = get_loss(**loss_init_args)

        # tail index (CXR-LT 기준)
        self.register_buffer(
            "tail_idx",
            torch.tensor([7, 11, 17, 18, 19, 21, 23, 25], dtype=torch.long)
        )

    def forward(self, image):
        # 기본 forward는 fusion logits만 반환
        logits_fusion, _, _ = self.backbone.forward_with_views(image)
        return logits_fusion

    def shared_step(self, batch, batch_idx):
        image, label = batch        # image: [B,S,3,H,W], label: [B,26]

        # 1) multi-head logits: fusion / PA-only / LAT-only
        logits_fusion, logits_pa, logits_lat = self.backbone.forward_with_pa_lat_logits(image)
        device = logits_fusion.device

        # 2) 메인 supervised losses (전부 self.criterion_cls 사용)
        loss_fusion = self.criterion_cls(logits_fusion, label)

        # view 존재 여부 (0 view / 1 view / padding 판별)
        flat_sum = image.sum(dim=(2, 3, 4))   # [B,S]
        valid_mask = (flat_sum != 0)          # [B,S]
        B, S = valid_mask.shape

        # 기본값 0
        loss_pa = logits_fusion.new_tensor(0.0)
        loss_lat = logits_fusion.new_tensor(0.0)

        # view0 = PA
        if S >= 1 and valid_mask[:, 0].any():
            idx_pa = valid_mask[:, 0]
            loss_pa = self.criterion_cls(
                logits_pa[idx_pa],
                label[idx_pa],
            )

        # view1 = LAT
        if S >= 2 and valid_mask[:, 1].any():
            idx_lat = valid_mask[:, 1]
            loss_lat = self.criterion_cls(
                logits_lat[idx_lat],
                label[idx_lat],
            )

        # ─────────────────────────────────────────────
        # 3) Tail-focused view HTB (teacher: PA/LAT, student: fusion)
        #    - teacher / student 모두 확률 space에서 MSE로 맞춤
        #    - tail class 에 대해서만 consistency 적용
        # ─────────────────────────────────────────────
        loss_htb_view = logits_fusion.new_tensor(0.0)

        has_pa  = (S >= 1) and valid_mask[:, 0].any()
        has_lat = (S >= 2) and valid_mask[:, 1].any()

        if has_pa or has_lat:
            eps   = 1e-8
            gamma = self.htb_alpha

            # (1) κ_head, κ_tail (논문 HTB의 kappa 역할, batch 단위 scalar)
            head_loss_main = loss_pa.detach().clamp(min=0.0)
            tail_loss_main = loss_lat.detach().clamp(min=0.0)

            head_score = head_loss_main ** gamma
            tail_score = tail_loss_main ** gamma
            denom = head_score + tail_score + eps

            kappa_head = head_score / denom   # κ_head
            kappa_tail = tail_score / denom   # κ_tail

            # (2) 확률로 변환
            prob_f   = torch.sigmoid(logits_fusion)  # [B,26]
            prob_pa  = torch.sigmoid(logits_pa)      # [B,26]
            prob_lat = torch.sigmoid(logits_lat)     # [B,26]

            tail_idx = self.tail_idx.to(device)      # [T]
            prob_f_t   = prob_f[:, tail_idx]         # [B,T]
            prob_pa_t  = prob_pa[:, tail_idx]
            prob_lat_t = prob_lat[:, tail_idx]

            # (3) view 존재하는 샘플만 사용
            loss_htb_pa  = logits_fusion.new_tensor(0.0)
            loss_htb_lat = logits_fusion.new_tensor(0.0)

            if has_pa:
                idx_pa = valid_mask[:, 0]            # [B]
                # tail class 에 대해서만 consistency
                loss_htb_pa = F.mse_loss(
                    prob_f_t[idx_pa],
                    prob_pa_t[idx_pa]
                )

            if has_lat:
                idx_lat = valid_mask[:, 1]
                loss_htb_lat = F.mse_loss(
                    prob_f_t[idx_lat],
                    prob_lat_t[idx_lat]
                )

            loss_htb_view = kappa_head * loss_htb_pa + kappa_tail * loss_htb_lat

        # ─────────────────────────────────────────────
        # 4) 최종 loss 합산
        # ─────────────────────────────────────────────
        loss = (
            loss_fusion
            + self.lambda_pa       * loss_pa
            + self.lambda_lat      * loss_lat
            + self.lambda_htb_view * loss_htb_view
        )

        self.log_dict({
            "L_fusion_step": loss_fusion,
            "L_pa_step": loss_pa,
            "L_lat_step": loss_lat,
            "L_htb_step": loss_htb_view,
            "wL_pa_step": self.lambda_pa * loss_pa,
            "wL_lat_step": self.lambda_lat * loss_lat,
            "wL_htb_step": self.lambda_htb_view * loss_htb_view,
        }, prog_bar=True, on_step=True, on_epoch=False)

        pred = torch.sigmoid(logits_fusion).detach()

        return dict(
            loss=loss,
            pred=pred,
            label=label,
        )

    # ---------------- Lightning 기본 루틴 ----------------

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'loss': res['loss'].detach()}, prog_bar=True)
        self.log_dict(
            {'train_loss': res['loss'].detach()},
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
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
        for i in range(len(self.classes)):
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            val_ap.append(ap)
            val_auroc.append(auroc)
            print(f'{self.classes[i]}_ap: {ap}')

        head_idx = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
        tail_idx = [7, 11, 17, 18, 19, 21, 23, 25]

        self.log_dict({'val_ap': sum(val_ap) / len(val_ap)}, prog_bar=True)
        self.log_dict({'val_auroc': sum(val_auroc) / len(val_auroc)}, prog_bar=True)
        self.log_dict(
            {'val_head_ap': sum([val_ap[i] for i in head_idx]) / len(head_idx)},
            prog_bar=True
        )
        self.log_dict(
            {'val_medium_ap': sum([val_ap[i] for i in medium_idx]) / len(medium_idx)},
            prog_bar=True
        )
        self.log_dict(
            {'val_tail_ap': sum([val_ap[i] for i in tail_idx]) / len(tail_idx)},
            prog_bar=True
        )
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        # fusion 기준 TTA (flip)
        image, label = batch
        logits_fusion, _, _ = self.backbone.forward_with_views(image)
        pred = torch.sigmoid(logits_fusion)

        batch_flip = (image.flip(-1), label)
        logits_fusion_flip, _, _ = self.backbone.forward_with_views(batch_flip[0])
        pred_flip = torch.sigmoid(logits_fusion_flip)

        pred = (pred + pred_flip) / 2
        return pred

    def test_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'test_loss': res['loss'].detach()}, prog_bar=True)
        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = []
        self.test_step_outputs.append(res)

    def on_test_epoch_end(self):
        preds = torch.cat([x['pred'] for x in self.test_step_outputs])
        labels = torch.cat([x['label'] for x in self.test_step_outputs])

        val_ap = []
        val_auroc = []
        for i in range(len(self.classes)):
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            val_ap.append(ap)
            val_auroc.append(auroc)
            print(f'[TEST] {self.classes[i]}_ap: {ap}')

        head_idx = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
        tail_idx = [7, 11, 17, 18, 19, 21, 23, 25]

        self.log_dict({'test_ap': sum(val_ap) / len(val_ap)}, prog_bar=True)
        self.log_dict({'test_auroc': sum(val_auroc) / len(val_auroc)}, prog_bar=True)
        self.log_dict(
            {'test_head_ap': sum([val_ap[i] for i in head_idx]) / len(head_idx)},
            prog_bar=True
        )
        self.log_dict(
            {'test_medium_ap': sum([val_ap[i] for i in medium_idx]) / len(medium_idx)},
            prog_bar=True
        )
        self.log_dict(
            {'test_tail_ap': sum([val_ap[i] for i in tail_idx]) / len(tail_idx)},
            prog_bar=True
        )
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.backbone.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 250000)
        return [optimizer], [scheduler]
