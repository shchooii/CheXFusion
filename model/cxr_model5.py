import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC
from transformers import get_cosine_schedule_with_warmup
from model.layers import FusionBackboneMultiView
from model.loss import get_loss


class CxrModel5(pl.LightningModule):
    def __init__(self, lr, classes, loss_init_args, timm_init_args,
                 lambda_pa=0.3, lambda_lat=0.3):
        super(CxrModel5, self).__init__()
        self.lr = lr
        self.classes = classes
        self.lambda_pa = lambda_pa
        self.lambda_lat = lambda_lat

        self.backbone = FusionBackboneMultiView(
            timm_init_args,
            pretrained_path='export/convnext_stage1_for_fusion.pth',
            num_classes=len(classes),
        )
        self.validation_step_outputs = []
        self.val_ap = AveragePrecision(task='binary')
        self.val_auc = AUROC(task="binary")

        self.criterion_cls = get_loss(**loss_init_args)

    def forward(self, image):
        # 기존 코드와 호환: fusion logits만 사용
        return self.backbone(image)

    def shared_step(self, batch, batch_idx):
        image, label = batch             # image: [B,S,3,H,W]
        logits_fusion, logits_pa, logits_lat = self.backbone.forward_with_pa_lat_logits(image)

        # ─── 1) fusion loss ───
        loss_fusion = self.criterion_cls(logits_fusion, label)

        # ─── 2) view 존재 여부 계산 ───
        # image 기준으로 padding 여부 판단
        flat_sum = image.sum(dim=(2, 3, 4))   # [B,S]
        valid_mask = (flat_sum != 0)          # [B,S]

        B, S = valid_mask.shape
        loss_pa = logits_fusion.new_tensor(0.0)
        loss_lat = logits_fusion.new_tensor(0.0)

        # ─── 3) PA loss (view 0) ───
        if S >= 1:
            mask_pa = valid_mask[:, 0]        # [B]
            if mask_pa.any():
                idx_pa = mask_pa
                logits_pa_valid = logits_pa[idx_pa]     # [B_pa,C]
                label_pa_valid = label[idx_pa]
                loss_pa = self.criterion_cls(logits_pa_valid, label_pa_valid)

        # ─── 4) LAT loss (view 1) ───
        if S >= 2:
            mask_lat = valid_mask[:, 1]
            if mask_lat.any():
                idx_lat = mask_lat
                logits_lat_valid = logits_lat[idx_lat]  # [B_lat,C]
                label_lat_valid = label[idx_lat]
                loss_lat = self.criterion_cls(logits_lat_valid, label_lat_valid)

        # ─── 5) total loss ───
        loss = loss_fusion + self.lambda_pa * loss_pa + self.lambda_lat * loss_lat

        pred = torch.sigmoid(logits_fusion).detach()

        return dict(
            loss=loss,
            pred=pred,
            label=label,
        )

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'loss': res['loss'].detach()}, prog_bar=True)
        self.log_dict({'train_loss': res['loss'].detach()}, prog_bar=True,
                      on_step=False, on_epoch=True)
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

        self.log_dict({'val_ap': sum(val_ap)/len(self.classes)}, prog_bar=True)
        self.log_dict({'val_auroc': sum(val_auroc)/len(self.classes)}, prog_bar=True)
        self.log_dict({'val_head_ap': sum([val_ap[i] for i in head_idx]) / len(head_idx)}, prog_bar=True)
        self.log_dict({'val_medium_ap': sum([val_ap[i] for i in medium_idx]) / len(medium_idx)}, prog_bar=True)
        self.log_dict({'val_tail_ap': sum([val_ap[i] for i in tail_idx]) / len(tail_idx)}, prog_bar=True)
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        # predict는 fusion만 사용
        pred = self.shared_step(batch, batch_idx)['pred']
        image, label = batch
        batch_1 = (image.flip(-1), label)
        pred_1 = self.shared_step(batch_1, batch_idx)['pred']
        pred = (pred + pred_1) / 2
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

        self.log_dict({'test_ap': sum(val_ap)/len(self.classes)}, prog_bar=True)
        self.log_dict({'test_auroc': sum(val_auroc)/len(self.classes)}, prog_bar=True)
        self.log_dict({'test_head_ap': sum([val_ap[i] for i in head_idx]) / len(head_idx)}, prog_bar=True)
        self.log_dict({'test_medium_ap': sum([val_ap[i] for i in medium_idx]) / len(medium_idx)}, prog_bar=True)
        self.log_dict({'test_tail_ap': sum([val_ap[i] for i in tail_idx]) / len(tail_idx)}, prog_bar=True)
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.backbone.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 250000)
        return [optimizer], [scheduler]
