import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC
from transformers import get_cosine_schedule_with_warmup

from model.layers import FusionBackboneMultiView2, FusionBackboneMultiView3
from model.loss import get_loss

class CxrModel7(pl.LightningModule):
    def __init__(
        self, lr, classes, loss_init_args, timm_init_args,
        lambda_pa=0.3, lambda_lat=0.3, lambda_consistency=0.2
    ):
        super(CxrModel7, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.classes = classes
        self.lambda_pa = lambda_pa
        self.lambda_lat = lambda_lat
        self.lambda_consistency = lambda_consistency

        self.backbone = FusionBackboneMultiView3(
            timm_init_args,
            pretrained_path='export/convnext_stage1_for_fusion2.pth',
            num_classes=len(classes),
        )

        self.val_ap = AveragePrecision(task='binary')
        self.val_auc = AUROC(task="binary")
        self.criterion_cls = get_loss(**loss_init_args)
        
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        logits_fusion, _, _ = self.backbone.forward_with_pa_lat_logits(image)
        return logits_fusion

    def shared_step(self, batch, batch_idx, mode='train'):
        image, label = batch
        logits_fusion, logits_pa, logits_lat = self.backbone.forward_with_pa_lat_logits(image)
        
        # 1. Main Loss
        loss_fusion = self.criterion_cls(logits_fusion, label)

        if mode in ['val', 'test']:
            pred = torch.sigmoid(logits_fusion).detach()
            return dict(loss=loss_fusion, pred=pred, label=label)

        # 2. View Check
        flat_sum = image.sum(dim=(2, 3, 4))
        valid_mask = (flat_sum != 0)
        has_pa = valid_mask[:, 0:2].any(dim=1)
        has_lat = valid_mask[:, 2:4].any(dim=1)

        # 3. Aux Loss
        loss_pa = logits_fusion.new_tensor(0.0)
        loss_lat = logits_fusion.new_tensor(0.0)
        if has_pa.any():
            loss_pa = self.criterion_cls(logits_pa[has_pa], label[has_pa])
        if has_lat.any():
            loss_lat = self.criterion_cls(logits_lat[has_lat], label[has_lat])

        # 4. Consistency (Basic MSE)
        loss_cons = logits_fusion.new_tensor(0.0)
        prob_fusion = torch.sigmoid(logits_fusion)
        prob_pa = torch.sigmoid(logits_pa)
        prob_lat = torch.sigmoid(logits_lat)

        if has_pa.any():
            loss_cons += F.mse_loss(prob_fusion[has_pa], prob_pa[has_pa])
        if has_lat.any():
            loss_cons += F.mse_loss(prob_fusion[has_lat], prob_lat[has_lat])

        total_loss = loss_fusion + (self.lambda_pa * loss_pa) + \
                     (self.lambda_lat * loss_lat) + (self.lambda_consistency * loss_cons)

        return dict(loss=total_loss, pred=prob_fusion.detach(), label=label,
                    loss_pa=loss_pa, loss_lat=loss_lat, loss_cons=loss_cons)

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx, mode='train')
        self.log('train_loss', res['loss'], prog_bar=True, on_epoch=True)
        self.log('L_cons', res['loss_cons'], prog_bar=False, on_epoch=True)
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