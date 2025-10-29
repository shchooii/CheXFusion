# eval_tune_same.py
import argparse, json, yaml, torch
import lightning as L
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

# ── 모델/데이터
from model.cxr_model import CxrModel
from dataset.cxr_datamodule import CxrDataModule

# ── metrics: mF1 계산은 기존 metrics.py 사용
from utils.metrics import F1Score
# ── threshold 튜닝 유틸
from utils.thresholds import best_f1_per_class_sklearn, f1_score_db_tuning

# ── sklearn으로 mAP/mAUC 산출
from sklearn.metrics import average_precision_score, roc_auc_score


# 기본 head/medium/tail 인덱스 (논문/네 코드 기준)
DEFAULT_GROUPS = {
    "head":   [0, 2, 4, 12, 14, 16, 20, 24],
    "medium": [1, 3, 5, 6, 8, 9, 10, 13, 15, 22],
    "tail":   [7, 11, 17, 18, 19, 21, 23, 25],
}


def safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


wandb = safe_import("wandb")
neptune = safe_import("neptune")


@torch.no_grad()
def run_predict(model, loader, device="cuda", desc="predict"):
    """모델 predict_step으로 확률을 쭉 뽑아서 (N,C) probs, labels 반환"""
    model.eval().to(device)
    all_probs, all_labels = [], []

    # 길이/샘플 수 정보
    try:
        n_batches = len(loader)
    except Exception:
        n_batches = None

    try:
        n_samples = len(loader.dataset)
    except Exception:
        n_samples = None

    if n_samples is not None and n_batches is not None:
        print(f"[{desc}] batches={n_batches}, samples={n_samples}")
    elif n_batches is not None:
        print(f"[{desc}] batches={n_batches}")
    elif n_samples is not None:
        print(f"[{desc}] samples={n_samples}")

    pbar_total = n_batches if n_batches is not None else None
    with tqdm(total=pbar_total, desc=f"{desc}", leave=False) as pbar:
        for bidx, batch in enumerate(loader):
            image, label = batch
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            prob = model.predict_step((image, label), bidx)   # 네 모델 정의에 맞춤
            all_probs.append(prob.float().cpu())
            all_labels.append(label.float().cpu())
            pbar.update(1)

    return torch.cat(all_probs, 0), torch.cat(all_labels, 0)


def binarize_with_thresholds(probs: torch.Tensor, thr):
    """thr: float 스칼라 또는 (C,) 텐서"""
    if isinstance(thr, (float, int)):
        thr_vec = probs.new_full((probs.shape[1],), float(thr))
    elif isinstance(thr, torch.Tensor):
        thr_vec = thr.to(probs.device if probs.is_cuda else "cpu").float()
        if thr_vec.ndim == 0:
            thr_vec = probs.new_full((probs.shape[1],), float(thr_vec.item()))
    else:
        raise ValueError("Unsupported threshold type")
    return (probs > thr_vec).int()


def mf1_from_logits(logits_bin: torch.Tensor, labels_bin: torch.Tensor):
    """
    metrics.py의 F1 정의(분모에 0.5*(FP+FN))와 동일 계산을 위해 F1Score 객체 활용.
    logits_bin/labels_bin은 이미 이진 상태로 전달한다.
    """
    C = labels_bin.shape[1]

    def compute_f1(average):
        met = F1Score(number_of_classes=C, threshold=0.5, average=average)
        met.to("cpu"); met.reset()
        batch = {"logits": logits_bin.cpu().float(), "targets": labels_bin.cpu().float()}
        met.update(batch)
        return float(met.compute().item())

    return compute_f1("micro"), compute_f1("macro")


def groups_from_file_or_default(path: str | None):
    """groups 파일이 있으면 로드, 없으면 DEFAULT_GROUPS 반환"""
    if path is None:
        return DEFAULT_GROUPS
    p = Path(path)
    if not p.exists():
        print(f"[warn] groups_file not found: {path} → DEFAULT_GROUPS 사용")
        return DEFAULT_GROUPS
    if p.suffix.lower() in [".yml", ".yaml"]:
        data = yaml.safe_load(open(p, "r"))
    else:
        data = json.loads(open(p, "r").read())
    groups = {}
    for k, v in data.items():
        groups[str(k)] = list(map(int, v))
    return groups


def eval_blocks(probs: torch.Tensor,
                labels: torch.Tensor,
                thr_mode: str,
                groups: dict | None):
    """
    probs, labels: (N, C) float
    thr_mode: fixed | single | per_class | per_group
    groups: per_group일 때 {"head":[...], ...}. None이면 DEFAULT_GROUPS 사용.
    Returns:
      dict:
        thresholds, mf1_micro, mf1_macro, map_macro, mauc_macro
        (optional) per_group: {g:{mf1_micro, mf1_macro, map_macro, mauc_macro}}
        (optional) per_group_best_f1 (per_group tuning 시)
    """
    out = {}

    # ── 1) threshold 선택
    if thr_mode == "fixed":
        thr = 0.5
    elif thr_mode == "single":
        best_f1, best_db = f1_score_db_tuning(
            torch.from_numpy(probs.numpy()),
            torch.from_numpy(labels.numpy()),
            groups={}, average="micro", type="single"
        )
        thr = float(best_db.item())
    elif thr_mode == "per_class":
        _, thr_vec = best_f1_per_class_sklearn(
            torch.from_numpy(probs.numpy()),
            torch.from_numpy(labels.numpy())
        )
        thr = torch.from_numpy(thr_vec.astype(np.float32))
    elif thr_mode == "per_group":
        gg = groups if groups is not None else DEFAULT_GROUPS
        best_f1_g, thr_vec = f1_score_db_tuning(
            torch.from_numpy(probs.numpy()),
            torch.from_numpy(labels.numpy()),
            groups={k: np.array(v, dtype=int).tolist() for k, v in gg.items()},
            average="micro", type="per_group"
        )
        thr = thr_vec.float().cpu()
        out["per_group_best_f1"] = {k: float(v) for k, v in best_f1_g.items()}
        groups = gg  # 보장
    else:
        raise ValueError(f"Unknown thr_mode: {thr_mode}")

    out["thresholds"] = thr if isinstance(thr, float) else thr.numpy()

    # ── 2) mF1 (metrics.py 정의) 계산
    preds_bin = binarize_with_thresholds(torch.from_numpy(probs.numpy()), thr)
    labs_bin  = (torch.from_numpy(labels.numpy()) > 0.5).int()
    mf1_micro, mf1_macro = mf1_from_logits(preds_bin, labs_bin)
    out["mf1_micro"] = mf1_micro
    out["mf1_macro"] = mf1_macro

    # ── 3) mAP / mAUC (macro)
    map_macro = float(average_precision_score(labels, probs, average="macro"))
    try:
        mauc_macro = float(roc_auc_score(labels, probs, average="macro"))
    except Exception:
        mauc_macro = float(roc_auc_score(labels, probs, average="weighted"))
    out["map_macro"] = map_macro
    out["mauc_macro"] = mauc_macro

    # ── 4) per-group 리포트(선택)
    if groups:
        per_group = {}
        for gname, idxs in groups.items():
            idxs = np.array(idxs, dtype=int)
            p_g = probs[:, idxs]
            y_g = labels[:, idxs]
            if isinstance(thr, float):
                thr_g = thr
            else:
                thr_g = thr[idxs]
            pb = binarize_with_thresholds(torch.from_numpy(p_g.numpy()), thr_g)
            yb = (torch.from_numpy(y_g.numpy()) > 0.5).int()
            g_mf1_micro, g_mf1_macro = mf1_from_logits(pb, yb)
            g_map = float(average_precision_score(y_g, p_g, average="macro"))
            try:
                g_auc = float(roc_auc_score(y_g, p_g, average="macro"))
            except Exception:
                g_auc = float(roc_auc_score(y_g, p_g, average="weighted"))
            per_group[gname] = {
                "mf1_micro": g_mf1_micro,
                "mf1_macro": g_mf1_macro,
                "map_macro": g_map,
                "mauc_macro": g_auc,
            }
        out["per_group"] = per_group

    return out


def summarize_for_table(probs: torch.Tensor,
                        labels: torch.Tensor,
                        groups: dict[str, list[int]]):
    """
    논문 표 형식: mAP_total, mAP_head, mAP_medium, mAP_tail, AUC_total
    - mAP: average_precision_score(..., average="macro")
    - AUC_total: roc_auc_score(..., average="macro") (실패 시 weighted)
    """
    # 전체
    map_total = float(average_precision_score(labels, probs, average="macro"))
    try:
        auc_total = float(roc_auc_score(labels, probs, average="macro"))
    except Exception:
        auc_total = float(roc_auc_score(labels, probs, average="weighted"))

    # 그룹별 mAP
    def map_of(idxs):
        idxs = np.array(idxs, dtype=int)
        return float(average_precision_score(labels[:, idxs], probs[:, idxs], average="macro"))

    map_head   = map_of(groups["head"])
    map_medium = map_of(groups["medium"])
    map_tail   = map_of(groups["tail"])

    return {
        "map_total": map_total,
        "map_head": map_head,
        "map_medium": map_medium,
        "map_tail": map_tail,
        "auc_total": auc_total,
    }


def maybe_init_wandb(args):
    if args.no_wandb or wandb is None:
        return None
    run = wandb.init(project=args.wandb_project, name=args.wandb_name,
                     tags=["eval", "tune"], reinit=True)
    return run


def maybe_init_neptune(args):
    if args.no_neptune or neptune is None:
        return None
    run = neptune.init_run(project=args.neptune_project, name=args.neptune_name)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True, help="val/test 공통으로 사용할 ckpt")
    ap.add_argument("--only_val",  action="store_true")
    ap.add_argument("--only_test", action="store_true")

    # threshold / group 옵션
    ap.add_argument("--thr_mode", choices=["fixed","single","per_class","per_group"], default="fixed")
    ap.add_argument("--groups_file", type=str, default=None,
                    help="per_group JSON/YAML (e.g. {'head':[...],'medium':[...],'tail':[...]} ), 미지정 시 DEFAULT_GROUPS")

    # 로깅 옵션
    ap.add_argument("--wandb_project", type=str, default="CXR-LT")
    ap.add_argument("--wandb_name",    type=str, default="eval")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--neptune_project", type=str, default="shchooii/hihi")
    ap.add_argument("--neptune_name",    type=str, default="eval")
    ap.add_argument("--no_neptune", action="store_true")

    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    L.seed_everything(cfg.get("seed_everything", 42), workers=True)

    # DataModule
    dm = CxrDataModule(cfg["data"]["datamodule_cfg"], cfg["data"]["dataloader_init_args"])

    # 모델 로드 (단일 모델)
    model = CxrModel.load_from_checkpoint(args.ckpt, strict=False, **cfg["model"]).eval()

    # 로깅 세션
    wbrun = maybe_init_wandb(args)
    nprun = maybe_init_neptune(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    thr_desc = args.thr_mode

    thr_vec = None
    results_val = None

    # ── VAL: threshold 튜닝 (필요 시)
    if not args.only_test:
        try: dm.setup(stage="validate")
        except: dm.setup("fit")
        val_loader = dm.val_dataloader()
        val_probs, val_labels = run_predict(model, val_loader, device=device, desc="VAL")

        # per_group일 때 groups_file 없으면 DEFAULT 사용
        groups = None
        if args.thr_mode == "per_group":
            groups = groups_from_file_or_default(args.groups_file)

        results_val = eval_blocks(val_probs, val_labels, thr_mode=args.thr_mode, groups=groups)
        thr_vec = results_val["thresholds"]
        print(f"[VAL/{thr_desc}] mAP={results_val['map_macro']:.6f} | "
              f"mAUC={results_val['mauc_macro']:.6f} | "
              f"mF1(μ)={results_val['mf1_micro']:.6f} | mF1(M)={results_val['mf1_macro']:.6f}")

        # 표 요약 (항상 DEFAULT_GROUPS 기준)
        summary_val = summarize_for_table(val_probs, val_labels, DEFAULT_GROUPS)
        print(f"[VAL/table] mAP_total={summary_val['map_total']:.6f} | "
              f"head={summary_val['map_head']:.6f} | "
              f"medium={summary_val['map_medium']:.6f} | "
              f"tail={summary_val['map_tail']:.6f} | "
              f"AUC_total={summary_val['auc_total']:.6f}")

        if wbrun:
            wandb.log({
                f"val/map_macro@{thr_desc}": results_val["map_macro"],
                f"val/mauc_macro@{thr_desc}": results_val["mauc_macro"],
                f"val/mf1_micro@{thr_desc}": results_val["mf1_micro"],
                f"val/mf1_macro@{thr_desc}": results_val["mf1_macro"],
                "val/table/mAP_total":  summary_val["map_total"],
                "val/table/mAP_head":   summary_val["map_head"],
                "val/table/mAP_medium": summary_val["map_medium"],
                "val/table/mAP_tail":   summary_val["map_tail"],
                "val/table/AUC_total":  summary_val["auc_total"],
            })
            if "per_group" in results_val:
                for g, d in results_val["per_group"].items():
                    wandb.log({
                        f"val/{g}/map_macro@{thr_desc}": d["map_macro"],
                        f"val/{g}/mauc_macro@{thr_desc}": d["mauc_macro"],
                        f"val/{g}/mf1_micro@{thr_desc}": d["mf1_micro"],
                        f"val/{g}/mf1_macro@{thr_desc}": d["mf1_macro"],
                    })

        if nprun:
            nprun[f"val/map_macro@{thr_desc}"] = results_val["map_macro"]
            nprun[f"val/mauc_macro@{thr_desc}"] = results_val["mauc_macro"]
            nprun[f"val/mf1_micro@{thr_desc}"] = results_val["mf1_micro"]
            nprun[f"val/mf1_macro@{thr_desc}"] = results_val["mf1_macro"]
            nprun["val/table/mAP_total"]  = summary_val["map_total"]
            nprun["val/table/mAP_head"]   = summary_val["map_head"]
            nprun["val/table/mAP_medium"] = summary_val["map_medium"]
            nprun["val/table/mAP_tail"]   = summary_val["map_tail"]
            nprun["val/table/AUC_total"]  = summary_val["auc_total"]
            if "per_group" in results_val:
                for g, d in results_val["per_group"].items():
                    nprun[f"val/{g}/map_macro@{thr_desc}"] = d["map_macro"]
                    nprun[f"val/{g}/mauc_macro@{thr_desc}"] = d["mauc_macro"]
                    nprun[f"val/{g}/mf1_micro@{thr_desc}"] = d["mf1_micro"]
                    nprun[f"val/{g}/mf1_macro@{thr_desc}"] = d["mf1_macro"]

    # ── TEST: 튜닝 thr 적용 or 0.5로 평가
    if not args.only_val:
        try: dm.setup(stage="test")
        except: dm.setup(None)
        if getattr(dm, "test_dataset", None) is None:
            raise RuntimeError("test_df_path가 없거나 test_dataset이 없습니다. config와 DataModule.setup('test')를 확인하세요.")
        test_loader = dm.test_dataloader()
        test_probs, test_labels = run_predict(model, test_loader, device=device, desc="TEST")

        if thr_vec is None:  # 튜닝을 안 했다면 0.5로
            use_thr_mode = "fixed"
        else:
            use_thr_mode = args.thr_mode

        groups = None
        if use_thr_mode == "per_group":
            groups = groups_from_file_or_default(args.groups_file)

        results_test = eval_blocks(test_probs, test_labels, thr_mode=use_thr_mode, groups=groups)
        tag = use_thr_mode
        print(f"[TEST/{tag}] mAP={results_test['map_macro']:.6f} | "
              f"mAUC={results_test['mauc_macro']:.6f} | "
              f"mF1(μ)={results_test['mf1_micro']:.6f} | mF1(M)={results_test['mf1_macro']:.6f}")

        # 표 요약 (항상 DEFAULT_GROUPS 기준)
        summary_test = summarize_for_table(test_probs, test_labels, DEFAULT_GROUPS)
        print(f"[TEST/table] mAP_total={summary_test['map_total']:.6f} | "
              f"head={summary_test['map_head']:.6f} | "
              f"medium={summary_test['map_medium']:.6f} | "
              f"tail={summary_test['map_tail']:.6f} | "
              f"AUC_total={summary_test['auc_total']:.6f}")

        if wbrun:
            wandb.log({
                f"test/map_macro@{tag}": results_test["map_macro"],
                f"test/mauc_macro@{tag}": results_test["mauc_macro"],
                f"test/mf1_micro@{tag}": results_test["mf1_micro"],
                f"test/mf1_macro@{tag}": results_test["mf1_macro"],
                "test/table/mAP_total":  summary_test["map_total"],
                "test/table/mAP_head":   summary_test["map_head"],
                "test/table/mAP_medium": summary_test["map_medium"],
                "test/table/mAP_tail":   summary_test["map_tail"],
                "test/table/AUC_total":  summary_test["auc_total"],
            })
            if "per_group" in results_test:
                for g, d in results_test["per_group"].items():
                    wandb.log({
                        f"test/{g}/map_macro@{tag}": d["map_macro"],
                        f"test/{g}/mauc_macro@{tag}": d["mauc_macro"],
                        f"test/{g}/mf1_micro@{tag}": d["mf1_micro"],
                        f"test/{g}/mf1_macro@{tag}": d["mf1_macro"],
                    })

        if nprun:
            nprun[f"test/map_macro@{tag}"] = results_test["map_macro"]
            nprun[f"test/mauc_macro@{tag}"] = results_test["mauc_macro"]
            nprun[f"test/mf1_micro@{tag}"] = results_test["mf1_micro"]
            nprun[f"test/mf1_macro@{tag}"] = results_test["mf1_macro"]
            nprun["test/table/mAP_total"]  = summary_test["map_total"]
            nprun["test/table/mAP_head"]   = summary_test["map_head"]
            nprun["test/table/mAP_medium"] = summary_test["map_medium"]
            nprun["test/table/mAP_tail"]   = summary_test["map_tail"]
            nprun["test/table/AUC_total"]  = summary_test["auc_total"]
            if "per_group" in results_test:
                for g, d in results_test["per_group"].items():
                    nprun[f"test/{g}/map_macro@{tag}"] = d["map_macro"]
                    nprun[f"test/{g}/mauc_macro@{tag}"] = d["mauc_macro"]
                    nprun[f"test/{g}/mf1_micro@{tag}"] = d["mf1_micro"]
                    nprun[f"test/{g}/mf1_macro@{tag}"] = d["mf1_macro"]

    if wbrun: wbrun.finish()
    if nprun: nprun.stop()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
