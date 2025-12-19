import torch
import timm
import torch.nn as nn
import copy    
import einops
import math
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from model.ml_decoder import MLDecoder
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer


class Backbone(nn.Module):
    def __init__(self, timm_init_args):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.model.head = nn.Identity()
        self.pos_encoding = Summer(PositionalEncoding2D(768))
        self.head = MLDecoder(num_classes=26, initial_num_features=768)

    def forward(self, x):
        x = self.model(x)
        x = self.pos_encoding(x)
        x = self.head(x)
        return x
    
    def forward_with_features(self, x):
        """
        Stage 1/2 공용: (logits, feature[=768-d]) 반환
        """
        feats = self.model(x)              # [B, 768, H, W]
        pos   = self.pos_encoding(feats)
        logits= self.head(pos)             # [B, 26]
        fvec  = feats.mean(dim=[2, 3])     # [B, 768]  ← SLP 통계용
        return logits, fvec


class SingleViewBackboneCOMIC(nn.Module):
    """
    Stage1: layer 구조는 Backbone(ConvNeXt -> PosEnc) 그대로
            + MLDecoder head 3개만 붙임 (h/t/b)
    반환:
      zh, zt, zb: [B, C]
      x_trans:    [B, L, 768]  (PosEnc feature를 token으로 펼친 것)
      mask:       [B, L] (False)
    """
    def __init__(self, timm_init_args, num_classes=26, dim=768):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.model.head = nn.Identity()  # feature map 뽑기 (Backbone과 동일)

        self.pos_encoding = Summer(PositionalEncoding2D(dim))

        # MLDecoder 3개만 추가
        self.head_h = MLDecoder(num_classes=num_classes, initial_num_features=dim)
        self.head_t = MLDecoder(num_classes=num_classes, initial_num_features=dim)
        self.head_b = MLDecoder(num_classes=num_classes, initial_num_features=dim)

    def forward(self, x):
        # feats: [B,768,Hf,Wf]
        feats = self.model(x)
        feats_pos = self.pos_encoding(feats)

        # MLDecoder가 [B,C,H,W]도 받는 기존 네 Backbone 방식 그대로
        zh = self.head_h(feats_pos)
        zt = self.head_t(feats_pos)
        zb = self.head_b(feats_pos)

        fvec = feats.mean(dim=[2, 3])  # [B,768]
        return zh, zt, zb, fvec
    
    
class FusionBackbone(nn.Module):
    def __init__(self, timm_init_args, pretrained_path=None):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.model.head = MLDecoder(num_classes=26, initial_num_features=768)
        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path))
        self.model.head = nn.Identity()
        self.conv2d = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(768))
        self.padding_token = nn.Parameter(torch.randn(1, 768, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(4, 768, 1, 1))
        
        self.head = MLDecoder(num_classes=26, initial_num_features=768)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True), num_layers=2)

    def forward(self, x):
        b, s, _, _, _ = x.shape

        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        no_pad = torch.nonzero(x.sum(dim=(1, 2, 3)) != 0).squeeze(1)
        x = x[no_pad]

        with torch.no_grad():
            x = self.model(x).detach()
        
        x = self.conv2d(x)
        x = self.pos_encoding(x)    

        pad_tokens = einops.repeat(self.padding_token, '1 c 1 1 -> (b s) c h w', b=b, s=s, h=x.shape[2], w=x.shape[3]).type_as(x)
        segment_embedding = einops.repeat(self.segment_embedding, 's c 1 1 -> (b s) c h w', b=b, h=x.shape[2], w=x.shape[3]).type_as(x)
        pad_tokens[no_pad] = x + segment_embedding[no_pad]
        x = pad_tokens

        x = einops.rearrange(x, '(b s) c h w -> b (s h w) c', b=b, s=s, h=x.shape[2], w=x.shape[3])
        mask =(x.sum(dim=-1) == 0).transpose(0, 1)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.head(x, mask)

        return x
    
    def forward_with_features(self, x):
        """
        x: [B, S, 3, H, W]
        return:
          logits: [B, 26]
          fvec : [B, 768]  (view-평균 global feature, EstimatorCV 입력용)
        """
        b, s, _, _, _ = x.shape

        # [B,S,C,H,W] → [B*S, C, H, W]
        x_flat = einops.rearrange(x, 'b s c h w -> (b s) c h w')

        # 어떤 view가 padding(전부 0)인지 확인
        flat_sum = x_flat.sum(dim=(1, 2, 3))          # [B*S]
        no_pad = torch.nonzero(flat_sum != 0).squeeze(1)  # 유효 view index

        x_valid = x_flat[no_pad]                      # [N_valid, C, H, W]

        # ----------------- 1) backbone feature-----------------
        with torch.no_grad():
            feats = self.model(x_valid).detach()      # [N_valid, 768, H1, W1]

        # per-view global feature [N_valid, 768]
        feat_per_view = feats.mean(dim=[2, 3])

        # B*S 전체에 놓고, padding view는 0으로 둔다
        BS = b * s
        D = feat_per_view.size(1)
        feat_all = feats.new_zeros((BS, D))
        feat_all[no_pad] = feat_per_view             # [B*S, 768]
        feat_all = feat_all.view(b, s, D)            # [B, S, 768]

        # view-wise 평균 (padding view는 제외)
        valid_mask = (flat_sum != 0).view(b, s).float().unsqueeze(-1)  # [B,S,1]
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)     # [B,1,1]
        fvec = (feat_all * valid_mask).sum(dim=1, keepdim=True) / denom  # [B,1,768]
        fvec = fvec.squeeze(1)                                         # [B,768]

        # ----------------- 2) 기존 forward 경로 그대로 -----------------
        x_enc = self.conv2d(feats)                 # [N_valid, 768, H2, W2]
        x_enc = self.pos_encoding(x_enc)

        pad_tokens = einops.repeat(
            self.padding_token,
            '1 c 1 1 -> (b s) c h w',
            b=b, s=s, h=x_enc.shape[2], w=x_enc.shape[3]
        ).type_as(x_enc)

        segment_embedding = einops.repeat(
            self.segment_embedding,
            's c 1 1 -> (b s) c h w',
            b=b, h=x_enc.shape[2], w=x_enc.shape[3]
        ).type_as(x_enc)

        pad_tokens[no_pad] = x_enc + segment_embedding[no_pad]
        x_seq = pad_tokens                           # [B*S, 768, H2, W2]

        x_seq = einops.rearrange(
            x_seq, '(b s) c h w -> b (s h w) c',
            b=b, s=s, h=x_enc.shape[2], w=x_enc.shape[3]
        )                                            # [B, L, 768]

        mask = (x_seq.sum(dim=-1) == 0)  # [L, B]

        x_trans = self.transformer_encoder(x_seq, src_key_padding_mask=mask)
        logits = self.head(x_trans, mask)           # [B, 26]

        return logits, fvec


class PretrainedBackbone(nn.Module):
    def __init__(self, timm_init_args, pretrained_path):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.new_head = copy.deepcopy(self.model.head)
        self.model.load_state_dict(torch.load(pretrained_path))
        self.model.head = nn.Identity()

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = self.new_head(x.detach())
        return x
    
    
class FusionBackbone2(FusionBackbone):
    def __init__(self, timm_init_args, pretrained_path=None):
        super().__init__(timm_init_args=timm_init_args, pretrained_path=pretrained_path)

        self.head_h = MLDecoder(num_classes=26, initial_num_features=768)
        self.head_t = MLDecoder(num_classes=26, initial_num_features=768)
        self.head_b = MLDecoder(num_classes=26, initial_num_features=768)

        if hasattr(self, "head"):
            del self.head

    def forward(self, x):
        b, s, _, _, _ = x.shape

        x = einops.rearrange(x, "b s c h w -> (b s) c h w")
        no_pad = torch.nonzero(x.sum(dim=(1, 2, 3)) != 0).squeeze(1)
        x_valid = x[no_pad]

                    
        with torch.no_grad():
            feats = self.model(x_valid)

        # stage2부터는 gradient 활성
        feats = feats.requires_grad_(True)

        x_enc = self.conv2d(feats)
        x_enc = self.pos_encoding(x_enc)

        pad_tokens = einops.repeat(
            self.padding_token,
            "1 c 1 1 -> (b s) c h w",
            b=b, s=s, h=x_enc.shape[2], w=x_enc.shape[3]
        ).type_as(x_enc)

        segment_embedding = einops.repeat(
            self.segment_embedding,
            "s c 1 1 -> (b s) c h w",
            b=b, h=x_enc.shape[2], w=x_enc.shape[3]
        ).type_as(x_enc)

        pad_tokens[no_pad] = x_enc + segment_embedding[no_pad]
        x_seq = pad_tokens

        x_seq = einops.rearrange(
            x_seq, "(b s) c h w -> b (s h w) c",
            b=b, s=s, h=x_enc.shape[2], w=x_enc.shape[3]
        )  # [B, L, 768]

        mask = (x_seq.sum(dim=-1) == 0)   # [B, L]

        x_trans = self.transformer_encoder(
            x_seq, src_key_padding_mask=mask
        )  # [B, L, 768]

        zh = self.head_h(x_trans, mask)
        zt = self.head_t(x_trans, mask)
        zb = self.head_b(x_trans, mask)

        return zh, zt, zb, x_trans, mask
    