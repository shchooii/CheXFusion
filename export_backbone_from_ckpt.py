# export_backbone_from_ckpt.py
import os, pathlib, torch, yaml
from model.cxr_model import CxrModel

cfg = yaml.load(open("config.yaml","r"), Loader=yaml.FullLoader)
ckpt = '/home/mixlab/tabular/shchoi/CheXFusion/save/attempt01/epoch=07-val_loss=0.0603-val_ap=0.35565.ckpt'

# 1) CxrModel 복원 (구조/하이퍼는 config와 동일하게)
model = CxrModel.load_from_checkpoint(
    ckpt,
    map_location="cpu",
    strict=False,               # 구조 차이 허용
    **cfg["model"]
)
print(f"[OK] loaded ckpt: {ckpt}")

# 2) 백본 state_dict 꺼내기 (Backbone 기준)
bb = getattr(model, "backbone", None)
if bb is None:
    raise RuntimeError("model.backbone 이 없습니다. CxrModel 정의 확인!")

if hasattr(bb, "model"):        # Backbone(timm_init_args) 구조
    sd = bb.model.state_dict()
else:
    sd = bb.state_dict()        # 혹시 내부 속성명이 다른 경우 대비

# 3) 저장
out_dir = pathlib.Path("export")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "backbone-stage1.pth"
torch.save(sd, out_path.as_posix())
print(f"[SAVE] backbone weights -> {out_path.resolve()}")

# 4) 빠른 검증: 로드 가능 여부
probe = torch.load(out_path, map_location="cpu")
print(f"[CHECK] keys: {len(probe)} params: {sum(v.numel() for v in probe.values()):,}")
