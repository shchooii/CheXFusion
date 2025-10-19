# utils/per_class_metrics.py
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score

def log_per_class_table(namespace: str, probs: np.ndarray, targets: np.ndarray, class_names: list[str], step: int | None = None):
    rows = []
    for i, name in enumerate(class_names):
        y_true = targets[:, i]
        y_prob = probs[:, i]
        # 양성 없는 클래스는 스킵
        if y_true.sum() == 0:
            ap, auc = float('nan'), float('nan')
        else:
            try:
                ap = average_precision_score(y_true, y_prob)
            except Exception:
                ap = float('nan')
            try:
                auc = roc_auc_score(y_true, y_prob)
            except Exception:
                auc = float('nan')
        rows.append({"class": name, "AP": ap, "AUROC": auc, "positives": int(y_true.sum())})
    table = wandb.Table(dataframe=pd.DataFrame(rows))
    wandb.log({f"{namespace}/per_class": table}, step=step)
