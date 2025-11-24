# export_backbone_from_ckpt.py
import pathlib
import torch
import yaml
from model.cxr_model import CxrModel

cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
ckpt = 'save/attempt01/epoch=07-val_loss=0.0603-val_ap=0.35565.ckpt'

# 1) stage1 CxrModel 로드
model = CxrModel.load_from_checkpoint(
    ckpt,
    map_location="cpu",
    strict=True,          # 구조 진짜 같은지 확인
    **cfg["model"],
)
print(f"[OK] loaded ckpt: {ckpt}")

bb = getattr(model, "backbone", None)
if bb is None:
    raise RuntimeError("model.backbone 이 없습니다. CxrModel 정의 확인!")

# 2) Backbone 전체 state_dict (model.*, head.*, pos_encoding.* ...)
sd = bb.state_dict()
print(f"[INFO] raw backbone keys: {len(sd)}")

# 3) ConvNeXt(+MLDecoder) 구조에 맞춘 state_dict 생성
#    - 'model.' prefix 제거 → ConvNeXt body
#    - 'head.'는 그대로 → MLDecoder
#    - 그 외(pos_encoding 등)는 버림
new_sd = {}
for k, v in sd.items():
    if k.startswith("model."):
        # 'model.' 떼고 저장
        new_k = k[len("model."):]   # e.g. model.stem.0.weight -> stem.0.weight
        new_sd[new_k] = v
    elif k.startswith("head."):
        # MLDecoder는 그대로 사용 (head.decoder....)
        new_sd[k] = v
    else:
        # pos_encoding 같은 건 ConvNeXt 안에 없으니 무시
        # print("[SKIP]", k)
        pass

print(f"[INFO] converted keys: {len(new_sd)}")

# 4) 저장
out_dir = pathlib.Path("export")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "convnext_stage1_for_fusion2.pth"
torch.save(new_sd, out_path.as_posix())
print(f"[SAVE] convnext+decoder weights -> {out_path.resolve()}")

# 5) 빠른 검증: 그냥 한 번 로드해 보기
probe = torch.load(out_path, map_location="cpu")
print(f"[CHECK] keys: {len(probe)} params: {sum(v.numel() for v in probe.values()):,}")
