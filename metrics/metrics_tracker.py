# utils/metrics_tracker.py (새 파일)
from typing import Dict, List, Optional
import torch
import wandb

from metrics.metrics_basic import (  # 너가 붙여준 Metric/MetricCollection 정의 파일 경로로 바꿔
    MetricCollection, LossMetric, F1Score, Precision, Recall, AUC,
    MeanAveragePrecision, PrecisionAtRecall, Precision_K, Recall_K
)

def build_metric_collection(num_classes: int) -> MetricCollection:
    metrics = [
        LossMetric(name="loss", filter_codes=False),
        F1Score(number_of_classes=num_classes, average="micro", name="f1", threshold=0.5),
        F1Score(number_of_classes=num_classes, average="macro", name="f1", threshold=0.5),
        Precision(number_of_classes=num_classes, average="micro", name="precision", threshold=0.5),
        Precision(number_of_classes=num_classes, average="macro", name="precision", threshold=0.5),
        Recall(number_of_classes=num_classes, average="micro", name="recall", threshold=0.5),
        Recall(number_of_classes=num_classes, average="macro", name="recall", threshold=0.5),
        AUC(average="micro", name="auc"),
        AUC(average="macro", name="auc"),
        MeanAveragePrecision(name="map"),
        PrecisionAtRecall(name="precision@recall"),
        Precision_K(k=5, name="precision"),
        Precision_K(k=10, name="precision"),
        Recall_K(k=5, name="recall"),
        Recall_K(k=10, name="recall"),
    ]
    return MetricCollection(metrics=metrics, code_indices=None, code_system_name=None)

class EpochBuffer:
    """에폭 동안 logits/targets/loss를 버퍼링해서 MetricCollection 업데이트 + compute"""
    def __init__(self, metric_collection: MetricCollection):
        self.mc = metric_collection
        self._logits: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._losses: List[torch.Tensor] = []

    def update_batch(self, logits: torch.Tensor, targets: torch.Tensor, loss: Optional[torch.Tensor] = None):
        # MetricCollection은 확률/스코어를 logits로 받는다고 가정(너 코드에서 sigmoid 후 확률로 전달)
        self.mc.update({"logits": logits.detach(), "targets": targets.detach(), "loss": (loss.detach().unsqueeze(0) if loss is not None else torch.tensor([0.0], device=logits.device))})
        self._logits.append(logits.detach())
        self._targets.append(targets.detach())
        if loss is not None:
            self._losses.append(loss.detach())

    def compute_and_reset(self) -> Dict[str, float]:
        if len(self._logits) == 0:
            # 비어있으면 빈 dict
            return {}
        logits = torch.cat(self._logits, dim=0).cpu()
        targets = torch.cat(self._targets, dim=0).cpu()
        out = self.mc.compute(logits=logits, targets=targets)  # dict[str, torch.Tensor | float]
        # LossMetric은 running mean으로 이미 기록됨. 추가로 평균 loss도 남겨두자.
        if len(self._losses):
            out["loss_epoch_mean"] = torch.stack(self._losses).mean().cpu()
        # 텐서는 float로
        out_float = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                out_float[k] = float(v.item())
            else:
                out_float[k] = float(v)
        # reset
        self.mc.reset_metrics()
        self._logits.clear()
        self._targets.clear()
        self._losses.clear()
        return out_float

def log_to_wandb(namespace: str, metrics: Dict[str, float], step: Optional[int] = None):
    if not metrics:
        return
    # "val/f1_micro": 0.53 같은 느낌으로 네임스페이스 붙이기
    flat = {f"{namespace}/{k}": v for k, v in metrics.items()}
    wandb.log(flat, step=step)
