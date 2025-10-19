import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from typing import Optional
import numpy as np
from sklearn.metrics import precision_recall_curve

# ─────────────────────────────
# 내부 모듈 import
# ─────────────────────────────
from model.layers import FusionBackbone
from model.loss import get_loss

# metric 관련 헬퍼 import (utils/metrics_custom.py)
from metrics.metrics_tracker import (
    build_metric_collection,
    EpochBuffer,
    log_to_wandb,
)

from metrics.metrics_per_class import log_per_class_table

# threshold 튜닝 함수 import (utils/threshold_tuning.py)
from metrics.threshold_tuning import (
    best_f1_per_class_sklearn,
    f1_score_db_tuning,
)


# ==========================================================
# CxrModel LightningModule
# ==========================================================
class CxrModel(pl.LightningModule):
    def __init__(self, lr, classes, loss_init_args, timm_init_args):
        super(CxrModel, self).__init__()
        self.save_hyperparameters(ignore=["classes"])
        self.lr = lr
        self.classes = classes
        self.backbone = FusionBackbone(timm_init_args, 'model.pth')

        num_classes = len(self.classes)
        self.criterion_cls = get_loss(**loss_init_args)

        # metric & buffer setup
        self.train_metrics = build_metric_collection(num_classes)
        self.val_metrics = build_metric_collection(num_classes)
        self.test_metrics = build_metric_collection(num_classes)
        self._train_buf = EpochBuffer(self.train_metrics)
        self._val_buf = EpochBuffer(self.val_metrics)
        self._test_buf = EpochBuffer(self.test_metrics)

        # val에서 튜닝된 threshold 저장용
        self.val_best_thr: Optional[torch.Tensor] = None
        self.val_group_thr: Optional[torch.Tensor] = None

    # ==========================================================
    # Forward & Shared Step
    # ==========================================================
    def forward(self, image):
        return self.backbone(image)

    def shared_step(self, batch, batch_idx):
        image, label = batch
        logits = self(image)
        loss = self.criterion_cls(logits, label)
        probs = torch.sigmoid(logits)
        return dict(loss=loss, probs=probs, logits=logits, label=label)

    # ==========================================================
    # Train / Val / Test Step
    # ==========================================================
    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self._train_buf.update_batch(logits=res['probs'], targets=res['label'], loss=res['loss'])
        self.log_dict({'train/loss': res['loss'].detach()}, prog_bar=True, on_epoch=True)
        return res['loss']

    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self._val_buf.update_batch(logits=res['probs'], targets=res['label'], loss=res['loss'])
        self.log('val/loss', res['loss'].detach(), prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self._test_buf.update_batch(logits=res['probs'], targets=res['label'], loss=res['loss'])
        return res

    # ==========================================================
    # Epoch End Hooks
    # ==========================================================
    def on_train_epoch_end(self):
        metrics = self._train_buf.compute_and_reset()
        if metrics:
            log_to_wandb("train", metrics, step=self.current_epoch)
            self.log_dict({f"train/{k}": v for k, v in metrics.items()
                           if k in ["f1_micro", "precision_micro", "recall_micro", "auc_micro", "map"]},
                          prog_bar=True)

    def on_validation_epoch_end(self):
        """
        validation 단계:
        1. per-class AP/AUC 표 W&B 업로드
        2. per-class / per-group F1 기반 threshold 튜닝
        3. 튜닝된 threshold로 val 재평가
        4. 기본 threshold(0.5) 평가도 함께 로깅
        """
        if len(self._val_buf._logits) == 0:
            return

        probs = torch.cat(self._val_buf._logits, dim=0)
        targets = torch.cat(self._val_buf._targets, dim=0)
        device, dtype = probs.device, probs.dtype

        # (1) per-class table logging
        try:
            log_per_class_table("val", probs.cpu().numpy(), targets.cpu().numpy(), self.classes, step=self.current_epoch)
        except Exception:
            pass

        # (2) per-class threshold tuning
        best_f1_pc, best_thr_pc = best_f1_per_class_sklearn(probs, targets)
        self.val_best_thr = torch.tensor(best_thr_pc, device=device, dtype=dtype)

        # (3) per-group tuning
        head_idx = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
        tail_idx = [7, 11, 17, 18, 19, 21, 23, 25]
        groups = {"head": head_idx, "medium": medium_idx, "tail": tail_idx}
        best_f1_g, thr_vec_g = f1_score_db_tuning(probs, targets, groups, average="micro", type="per_group")
        self.val_group_thr = thr_vec_g

        # (4) tuned threshold metrics (per-class)
        mc_pc = self.val_metrics.copy()
        mc_pc.set_threshold(self.val_best_thr)
        tuned_pc = mc_pc.compute(logits=probs.cpu(), targets=targets.cpu())
        tuned_pc = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in tuned_pc.items()}
        log_to_wandb("val_tuned_pc", tuned_pc, step=self.current_epoch)

        # (5) tuned threshold metrics (per-group)
        mc_g = self.val_metrics.copy()
        mc_g.set_threshold(self.val_group_thr)
        tuned_g = mc_g.compute(logits=probs.cpu(), targets=targets.cpu())
        tuned_g = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in tuned_g.items()}
        log_to_wandb("val_tuned_group", tuned_g, step=self.current_epoch)

        # (6) default threshold(0.5) metric logging
        metrics = self._val_buf.compute_and_reset()
        log_to_wandb("val", metrics, step=self.current_epoch)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()
                       if k in ["f1_micro", "precision_micro", "recall_micro", "auc_micro", "map"]},
                      prog_bar=True)

    def on_test_epoch_end(self):
        """
        test 단계:
        validation에서 튜닝된 threshold(self.val_best_thr, self.val_group_thr)를 그대로 사용
        """
        if len(self._test_buf._logits) == 0:
            return

        probs = torch.cat(self._test_buf._logits, dim=0)
        targets = torch.cat(self._test_buf._targets, dim=0)

        # 기본 0.5 지표
        metrics = self._test_buf.compute_and_reset()
        log_to_wandb("test_default", metrics, step=self.current_epoch)

        # per-class tuned
        if self.val_best_thr is not None:
            mc_pc = self.test_metrics.copy()
            mc_pc.set_threshold(self.val_best_thr)
            tuned_pc = mc_pc.compute(logits=probs.cpu(), targets=targets.cpu())
            tuned_pc = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in tuned_pc.items()}
            log_to_wandb("test_tuned_pc", tuned_pc, step=self.current_epoch)

        # per-group tuned
        if self.val_group_thr is not None:
            mc_g = self.test_metrics.copy()
            mc_g.set_threshold(self.val_group_thr)
            tuned_g = mc_g.compute(logits=probs.cpu(), targets=targets.cpu())
            tuned_g = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in tuned_g.items()}
            log_to_wandb("test_tuned_group", tuned_g, step=self.current_epoch)

        self.log_dict({f"test/{k}": v for k, v in metrics.items()
                       if k in ["f1_micro", "precision_micro", "recall_micro", "auc_micro", "map"]},
                      prog_bar=False)

    # ==========================================================
    # Optimizer
    # ==========================================================
    def configure_optimizers(self):
        optimizer = AdamW(self.backbone.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 250000)
        return [optimizer], [scheduler]
