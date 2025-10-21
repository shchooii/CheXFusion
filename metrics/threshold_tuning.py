import torch
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve


def best_f1_per_class_sklearn(logits: torch.Tensor,
                              targets: torch.Tensor):
    """
    각 클래스별로 precision‑recall 곡선을 이용해
    F1 최댓값과 그때의 threshold 반환
    """
    # ─── 1. Tensor → NumPy ───────────────────────────────────────────
    y_true = targets.detach().cpu().numpy()
    y_score = logits.detach().cpu().numpy()

    n_classes = y_true.shape[1]
    best_f1 = np.zeros(n_classes)
    best_thr = np.zeros(n_classes)

    # ─── 2. 클래스별 PR‑Curve → F1 최적화 ─────────────────────────────
    for c in range(n_classes):
        precision, recall, thrs = precision_recall_curve(y_true[:, c],
                                                         y_score[:, c])
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        idx = np.argmax(f1)

        best_f1[c] = f1[idx]
        # precision_recall_curve 는 thresholds 길이가 n‑1 ⇒
        # idx==len(thrs) 이면 threshold = 1.0 으로 간주
        best_thr[c] = thrs[idx] if idx < len(thrs) else 1.0

    return best_f1, best_thr


def f1_score_db_tuning(logits, targets, groups, average="micro", type="single"):
    
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().float().cpu()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().float().cpu()
    if groups is not None and isinstance(groups, torch.Tensor):
        groups = groups.detach().long().cpu()
           
    device, dtype = logits.device, logits.dtype
    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")
    dbs = torch.linspace(0, 1, 100)
    n_cls = targets.size(1)
    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
    if average == "micro":
        f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
    else:
        f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)
    if type == "single":
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db
    if type == "per_class":
        y_true = targets.detach().cpu().numpy()
        y_prob = logits.detach().cpu().numpy()
        best_f1, best_db = [], []
        for c in range(n_cls):
            p, r, th = precision_recall_curve(y_true[:, c], y_prob[:, c])
            f1 = 2 * p * r / (p + r + 1e-12)
            j = f1.argmax()
            best_f1.append(float(f1[j]))
            best_db.append(float(th[j]) if j < len(th) else 1.0)
        best_f1 = torch.tensor(best_f1, device=device, dtype=dtype)
        best_db = torch.tensor(best_db, device=device, dtype=dtype)
        return best_f1, best_db

    if type == "per_group":
        thr_vec = torch.full((targets.shape[1],), 0.5)
        cls_f1 = tp / (tp + 0.5 * (fp + fn) + 1e-10)
        best_f1_g, best_db_g = {}, {}
        for g, idxs in groups.items():
            # idxs = torch.as_tensor(idxs, device=logits.device)
            idxs = (groups == g).nonzero(as_tuple=False).squeeze(1)
            if average == "micro":
                g_tp = tp[:, idxs].sum(1)
                g_fp = fp[:, idxs].sum(1)
                g_fn = fn[:, idxs].sum(1)
                g_f1 = g_tp / (g_tp + 0.5 * (g_fp + g_fn) + 1e-10)
            else:
                g_f1 = cls_f1[:, idxs].mean(1)
            best = g_f1.argmax()
            best_f1_g[g] = g_f1[best].item()
            best_db_g[g] = dbs[best].item()
            thr_vec[torch.as_tensor(idxs, dtype=torch.long)] = best_db_g[g]
        return best_f1_g, thr_vec
