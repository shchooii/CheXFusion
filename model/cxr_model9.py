import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC
from transformers import get_cosine_schedule_with_warmup

from model.layers import FusionBackboneMultiView2
from model.loss import get_loss

class CxrModel9(pl.LightningModule):
    def __init__(
        self, lr, classes, loss_init_args, timm_init_args,
        lambda_pa=0.3, lambda_lat=0.3, lambda_consistency=0.5, 
        htb_alpha=2.0, mt_weight=2.0
    ):
        super(CxrModel9, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.classes = classes
        self.lambda_pa = lambda_pa
        self.lambda_lat = lambda_lat
        self.lambda_consistency = lambda_consistency
        self.htb_alpha = htb_alpha
        self.mt_weight = mt_weight # Medium/Tail 가중치

        self.backbone = FusionBackboneMultiView2(
            timm_init_args,
            pretrained_path='export/convnext_stage1_for_fusion2.pth',
            num_classes=len(classes),
        )

        self.val_ap = AveragePrecision(task='binary')
        self.val_auc = AUROC(task="binary")
        self.criterion_cls = get_loss(**loss_init_args)
        
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Class Indices 설정
        # Head Index (제외할 대상)
        self.register_buffer("head_idx", torch.tensor([0, 2, 4, 12, 14, 16, 20, 24], dtype=torch.long))
        # Medium + Tail Index (집중할 대상)
        mt_list = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22] + [7, 11, 17, 18, 19, 21, 23, 25]
        self.register_buffer("mt_idx", torch.tensor(mt_list, dtype=torch.long))

    def forward(self, image):
        logits_fusion, _, _ = self.backbone.forward_with_pa_lat_logits(image)
        return logits_fusion

    def shared_step(self, batch, batch_idx, mode='train'):
        image, label = batch
        logits_fusion, logits_pa, logits_lat = self.backbone.forward_with_pa_lat_logits(image)
        
        loss_fusion = self.criterion_cls(logits_fusion, label)

        if mode in ['val', 'test']:
            return dict(loss=loss_fusion, pred=torch.sigmoid(logits_fusion).detach(), label=label)

        flat_sum = image.sum(dim=(2, 3, 4))
        valid_mask = (flat_sum != 0)
        has_pa = valid_mask[:, 0:2].any(dim=1)
        has_lat = valid_mask[:, 2:4].any(dim=1)

        loss_pa = logits_fusion.new_tensor(0.0)
        loss_lat = logits_fusion.new_tensor(0.0)
        if has_pa.any():
            loss_pa = self.criterion_cls(logits_pa[has_pa], label[has_pa])
        if has_lat.any():
            loss_lat = self.criterion_cls(logits_lat[has_lat], label[has_lat])

        # [Class-Focused Logic]
        loss_cons = logits_fusion.new_tensor(0.0)
        kappa_pa, kappa_lat = 0.0, 0.0

        if has_pa.any() or has_lat.any():
            # 1. Kappa 계산
            with torch.no_grad():
                s_pa = loss_pa.detach() ** self.htb_alpha
                s_lat = loss_lat.detach() ** self.htb_alpha
                denom = s_pa + s_lat + 1e-8
                kappa_pa = s_pa / denom
                kappa_lat = s_lat / denom

            prob_fusion = torch.sigmoid(logits_fusion)
            prob_pa = torch.sigmoid(logits_pa)
            prob_lat = torch.sigmoid(logits_lat)

            # 2. Class Weighting (Head=0, MT=mt_weight)
            # 전체 Prob에서 MT 부분만 추출해서 계산 (Head는 아예 계산 제외)
            
            pf_mt = prob_fusion[:, self.mt_idx]
            ppa_mt = prob_pa[:, self.mt_idx]
            plat_mt = prob_lat[:, self.mt_idx]

            # 3. Consistency Calculation (MT Only)
            if has_pa.any():
                mse_mt = F.mse_loss(pf_mt[has_pa], ppa_mt[has_pa])
                loss_cons += kappa_pa * (mse_mt * self.mt_weight)

            if has_lat.any():
                mse_mt = F.mse_loss(pf_mt[has_lat], plat_mt[has_lat])
                loss_cons += kappa_lat * (mse_mt * self.mt_weight)

        total_loss = loss_fusion + (self.lambda_pa * loss_pa) + \
                     (self.lambda_lat * loss_lat) + (self.lambda_consistency * loss_cons)

        return dict(loss=total_loss, pred=torch.sigmoid(logits_fusion).detach(), label=label,
                    loss_pa=loss_pa, loss_lat=loss_lat, loss_cons=loss_cons,
                    kappa_pa=kappa_pa, kappa_lat=kappa_lat)

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx, mode='train')
        self.log('train_loss', res['loss'], prog_bar=True, on_epoch=True)
        self.log('L_cons', res['loss_cons'], prog_bar=False, on_epoch=True)
        if isinstance(res['kappa_pa'], torch.Tensor) or res['kappa_pa'] > 0:
            self.log('k_pa', res['kappa_pa'], prog_bar=False, on_epoch=True)
            self.log('k_lat', res['kappa_lat'], prog_bar=False, on_epoch=True)
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
            self.val_ap.reset()
            self.val_auc.reset()
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
            self.val_ap.reset()
            self.val_auc.reset()
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