import torch
import timm
import torch.nn as nn
import copy    
import einops
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
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8), num_layers=2)
        
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

        mask = (x_seq.sum(dim=-1) == 0).transpose(0, 1)  # [L, B]

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


class Backbone3Head(nn.Module):
    """
    - 기존 Backbone과 동일하게 timm 백본 + PositionalEncoding2D 사용
    - MLDecoder를 3개 사용:
        * head_bal  : 전체 클래스용 [B, 26]
        * head_head : head 클래스만 [B, |H|] → [B, 26]으로 scatter
        * head_tail : tail 클래스만 [B, |T|] → [B, 26]으로 scatter
    - forward()      → logits_head, logits_tail, logits_bal  (각 [B, 26])
    - forward_with_features() → logits_head, logits_tail, logits_bal, fvec  (fvec: [B, 768])
    """

    def __init__(self, timm_init_args):
        super().__init__()

        # ----- backbone -----
        self.model = timm.create_model(**timm_init_args)
        self.model.head = nn.Identity()           # feature map [B, 768, H, W] 유지
        self.pos_encoding = Summer(PositionalEncoding2D(768))

        num_classes = 26
        self.num_classes = num_classes

        # ----- 인덱스: 내부에서 선언 -----
        # head / medium / tail 분할 (앞에서 쓰던 것 그대로)
        head_idx   = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]  # 모델에서는 안 쓰지만 참고용으로 유지
        tail_idx   = [7, 11, 17, 18, 19, 21, 23, 25]

        # device 이동 자동으로 되게 buffer로 등록
        self.register_buffer("head_idx",   torch.tensor(head_idx,   dtype=torch.long))
        self.register_buffer("tail_idx",   torch.tensor(tail_idx,   dtype=torch.long))
        self.register_buffer("medium_idx", torch.tensor(medium_idx, dtype=torch.long))

        # ----- MLDecoder head 3개 -----
        self.head_bal  = MLDecoder(num_classes=num_classes,
                                   initial_num_features=768)
        self.head_head = MLDecoder(num_classes=len(head_idx),
                                   initial_num_features=768)
        self.head_tail = MLDecoder(num_classes=len(tail_idx),
                                   initial_num_features=768)

    def _scatter(self, part_logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        part_logits: [B, |idx|]
        idx        : [|idx|]  (전역 클래스 인덱스)
        return     : [B, num_classes]
        """
        B = part_logits.size(0)
        full = part_logits.new_zeros(B, self.num_classes)
        full.index_copy_(1, idx, part_logits)
        return full

    def forward(self, x):
        """
        x: [B, 3, H, W]
        return:
          logits_head: [B, 26]
          logits_tail: [B, 26]
          logits_bal : [B, 26]
        """
        feats = self.model(x)            # [B, 768, H, W]
        pos   = self.pos_encoding(feats) # [B, 768, H, W]

        # MLDecoder들은 [B, 768, H, W] → [B, num_classes]
        logits_bal   = self.head_bal(pos)          # [B, 26]
        logits_head_ = self.head_head(pos)         # [B, |H|]
        logits_tail_ = self.head_tail(pos)         # [B, |T|]

        # [B, |H|] / [B, |T|] → [B, 26]
        logits_head = self._scatter(logits_head_, self.head_idx)
        logits_tail = self._scatter(logits_tail_, self.tail_idx)

        return logits_head, logits_tail, logits_bal

    def forward_with_features(self, x):
        """
        x: [B, 3, H, W]
        return:
          logits_head: [B, 26]
          logits_tail: [B, 26]
          logits_bal : [B, 26]
          fvec       : [B, 768]  (global feature, SLP / RLC 통계 등에 사용 가능)
        """
        feats = self.model(x)                # [B, 768, H, W]
        fvec  = feats.mean(dim=[2, 3])       # [B, 768]

        pos   = self.pos_encoding(feats)
        logits_bal   = self.head_bal(pos)    # [B, 26]
        logits_head_ = self.head_head(pos)   # [B, |H|]
        logits_tail_ = self.head_tail(pos)   # [B, |T|]

        logits_head = self._scatter(logits_head_, self.head_idx)
        logits_tail = self._scatter(logits_tail_, self.tail_idx)

        return logits_head, logits_tail, logits_bal, fvec


class FusionBackbone3Head(nn.Module):
    """
    FusionBackbone 구조 + Backbone3Head의 3-head 구조 결합 버전
    - 입력: x [B, S, 3, H, W]
    - forward:
        return logits_head, logits_tail, logits_bal  (각 [B, 26])
    - forward_with_features:
        return logits_head, logits_tail, logits_bal, fvec (fvec: [B, 768])
    """

    def __init__(self, timm_init_args, pretrained_path=None):
        super().__init__()

        # ----- backbone (기존 FusionBackbone 그대로) -----
        self.model = timm.create_model(**timm_init_args)
        self.model.head = MLDecoder(num_classes=26, initial_num_features=768)
        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path))
        self.model.head = nn.Identity()

        self.conv2d = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(768))
        self.padding_token = nn.Parameter(torch.randn(1, 768, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(4, 768, 1, 1))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=2
        )

        num_classes = 26
        self.num_classes = num_classes

        # ----- head / medium / tail index (Backbone3Head 그대로) -----
        head_idx   = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]  # 사용은 안 하지만 유지
        tail_idx   = [7, 11, 17, 18, 19, 21, 23, 25]

        self.register_buffer("head_idx",   torch.tensor(head_idx,   dtype=torch.long))
        self.register_buffer("tail_idx",   torch.tensor(tail_idx,   dtype=torch.long))
        self.register_buffer("medium_idx", torch.tensor(medium_idx, dtype=torch.long))

        # ----- MLDecoder 3개 (FusionBackbone에서 쓰던 시그니처: (x, mask)) -----
        self.head_bal  = MLDecoder(num_classes=num_classes,
                                   initial_num_features=768)
        self.head_head = MLDecoder(num_classes=len(head_idx),
                                   initial_num_features=768)
        self.head_tail = MLDecoder(num_classes=len(tail_idx),
                                   initial_num_features=768)

    def _scatter(self, part_logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        part_logits: [B, |idx|]
        idx        : [|idx|]
        return     : [B, num_classes]
        """
        B = part_logits.size(0)
        full = part_logits.new_zeros(B, self.num_classes)
        full.index_copy_(1, idx, part_logits)
        return full

    def forward(self, x):
        """
        x: [B, S, 3, H, W]
        return:
          logits_head: [B, 26]
          logits_tail: [B, 26]
          logits_bal : [B, 26]
        """
        b, s, _, _, _ = x.shape

        # [B, S, C, H, W] → [B*S, C, H, W]
        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')

        # padding view (올 0) 제거
        flat_sum = x.sum(dim=(1, 2, 3))                   # [B*S]
        no_pad = torch.nonzero(flat_sum != 0).squeeze(1)  # 유효 view index
        x = x[no_pad]                                     # [N_valid, C, H, W]

        # ----- backbone -----
        with torch.no_grad():
            feats = self.model(x).detach()                # [N_valid, 768, H1, W1]

        # ----- conv + pos encoding -----
        x_enc = self.conv2d(feats)                        # [N_valid, 768, H2, W2]
        x_enc = self.pos_encoding(x_enc)

        # ----- pad token + segment embedding (기존 FusionBackbone 로직) -----
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
        x_seq = pad_tokens                                 # [B*S, 768, H2, W2]

        # [B*S, C, H2, W2] → [B, L, C]
        x_seq = einops.rearrange(
            x_seq, '(b s) c h w -> b (s h w) c',
            b=b, s=s, h=x_enc.shape[2], w=x_enc.shape[3]
        )                                                  # [B, L, 768]

        # transformer용 mask: [L, B] (True = padding)
        mask = (x_seq.sum(dim=-1) == 0).transpose(0, 1)

        # ----- transformer encoder -----
        x_trans = self.transformer_encoder(
            x_seq, src_key_padding_mask=mask
        )                                                  # [B, L, 768]

        # ----- 3-way MLDecoder head -----
        logits_bal   = self.head_bal(x_trans, mask)        # [B, 26]
        logits_head_ = self.head_head(x_trans, mask)       # [B, |H|]
        logits_tail_ = self.head_tail(x_trans, mask)       # [B, |T|]

        logits_head = self._scatter(logits_head_, self.head_idx)
        logits_tail = self._scatter(logits_tail_, self.tail_idx)

        return logits_head, logits_tail, logits_bal

    def forward_with_features(self, x):
        """
        x: [B, S, 3, H, W]
        return:
          logits_head: [B, 26]
          logits_tail: [B, 26]
          logits_bal : [B, 26]
          fvec       : [B, 768]  (view-wise global feature 평균)
        """
        b, s, _, _, _ = x.shape

        # [B,S,C,H,W] → [B*S, C, H, W]
        x_flat = einops.rearrange(x, 'b s c h w -> (b s) c h w')

        # padding view 체크
        flat_sum = x_flat.sum(dim=(1, 2, 3))               # [B*S]
        no_pad = torch.nonzero(flat_sum != 0).squeeze(1)   # [N_valid]
        x_valid = x_flat[no_pad]                           # [N_valid, C, H, W]

        # ----- backbone -----
        with torch.no_grad():
            feats = self.model(x_valid).detach()           # [N_valid, 768, H1, W1]

        # per-view global feature
        feat_per_view = feats.mean(dim=[2, 3])             # [N_valid, 768]

        BS = b * s
        D = feat_per_view.size(1)

        # B*S 전체 frame에 embedding 깔고 padding view는 0
        feat_all = feats.new_zeros((BS, D))                # [B*S, 768]
        feat_all[no_pad] = feat_per_view
        feat_all = feat_all.view(b, s, D)                  # [B, S, 768]

        # valid mask
        valid_mask = (flat_sum != 0).view(b, s).float().unsqueeze(-1)  # [B,S,1]
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)     # [B,1,1]
        fvec = (feat_all * valid_mask).sum(dim=1, keepdim=True) / denom  # [B,1,768]
        fvec = fvec.squeeze(1)                                         # [B,768]

        # ----- conv + pos encoding (classification path) -----
        x_enc = self.conv2d(feats)                         # [N_valid, 768, H2, W2]
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
        x_seq = pad_tokens                                 # [B*S, 768, H2, W2]

        x_seq = einops.rearrange(
            x_seq, '(b s) c h w -> b (s h w) c',
            b=b, s=s, h=x_enc.shape[2], w=x_enc.shape[3]
        )                                                  # [B, L, 768]

        mask = (x_seq.sum(dim=-1) == 0).transpose(0, 1)    # [L, B]

        x_trans = self.transformer_encoder(
            x_seq, src_key_padding_mask=mask
        )                                                  # [B, L, 768]

        logits_bal   = self.head_bal(x_trans, mask)        # [B, 26]
        logits_head_ = self.head_head(x_trans, mask)       # [B, |H|]
        logits_tail_ = self.head_tail(x_trans, mask)       # [B, |T|]

        logits_head = self._scatter(logits_head_, self.head_idx)
        logits_tail = self._scatter(logits_tail_, self.tail_idx)

        return logits_head, logits_tail, logits_bal, fvec


class FusionBackboneMultiView(nn.Module):
    def __init__(self, timm_init_args, pretrained_path=None, num_classes=26):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        # stage1에서 썼던 MLDecoder head 로드 후 backbone weight만 사용
        self.model.head = MLDecoder(num_classes=num_classes, initial_num_features=768)
        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        # backbone feature만 쓰도록 head 제거
        self.model.head = nn.Identity()

        self.conv2d = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(768))
        self.padding_token = nn.Parameter(torch.randn(1, 768, 1, 1))

        # 최대 view 개수(=segment embedding 개수)
        self.segment_embedding = nn.Parameter(torch.randn(4, 768, 1, 1))
        self.max_views = self.segment_embedding.shape[0]

        # ────────────── ML-Decoder 3개 ──────────────
        # 1) 멀티뷰 fusion용 (기존 head)
        self.head_fusion = MLDecoder(num_classes=num_classes, initial_num_features=768)
        # 2) PA view용 (view index 0)
        self.head_pa = MLDecoder(num_classes=num_classes, initial_num_features=768)
        # 3) LAT view용 (view index 1)
        self.head_lat = MLDecoder(num_classes=num_classes, initial_num_features=768)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=2
        )

    # 기존 forward는 fusion 결과만 내보내서 기존 코드와 호환
    def forward(self, x):
        logits_fusion, _, _ = self.forward_with_pa_lat_logits(x)
        return logits_fusion

    def forward_with_pa_lat_logits(self, x):
        """
        x: [B, S, 3, H, W]
        return:
          logits_fusion: [B, C]
          logits_pa    : [B, C]  (PA 없는 샘플은 무시하고 loss에서 필터링)
          logits_lat   : [B, C]  (LAT 없는 샘플은 무시하고 loss에서 필터링)
        """
        b, s, _, _, _ = x.shape

        # [B,S,C,H,W] → [B*S, C, H, W]
        x_flat = einops.rearrange(x, 'b s c h w -> (b s) c h w')      # [B*S,3,H,W]
        flat_sum = x_flat.sum(dim=(1, 2, 3))                         # [B*S]
        no_pad = torch.nonzero(flat_sum != 0).squeeze(1)             # 유효 view index (0이 아닌 애들)

        x_valid = x_flat[no_pad]                                     # [N_valid,3,H,W]

        # ----------------- 1) backbone feature -----------------
        # CheXFusion stage2처럼 backbone은 freeze 상태 유지
        with torch.no_grad():
            feats = self.model(x_valid).detach()                     # [N_valid,768,H1,W1]

        # conv + pos encoding (fusion용)
        x_enc = self.conv2d(feats)                                   # [N_valid,768,H2,W2]
        x_enc = self.pos_encoding(x_enc)

        # pad token + segment embedding
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
        x_seq = pad_tokens                                            # [B*S,768,H2,W2]

        # [B*S, C, H2, W2] → [B, L, C]
        x_seq = einops.rearrange(
            x_seq, '(b s) c h w -> b (s h w) c',
            b=b, s=s, h=x_enc.shape[2], w=x_enc.shape[3]
        )                                                             # [B, L, 768]

        # token mask (0이면 padding)
        mask_tokens = (x_seq.sum(dim=-1) == 0).transpose(0, 1)       # [L, B]

        # transformer + fusion head
        x_trans = self.transformer_encoder(x_seq, src_key_padding_mask=mask_tokens)
        logits_fusion = self.head_fusion(x_trans, mask_tokens)       # [B, C]

        # ----------------- 2) view별 global feature -----------------
        # per-view global feature: feats.mean(H,W)  [N_valid,768]
        feat_per_view = feats.mean(dim=[2, 3])                       # [N_valid, 768]

        BS = b * s
        D = feat_per_view.size(1)
        feat_all = feats.new_zeros((BS, D))                          # [B*S,768]
        feat_all[no_pad] = feat_per_view                             # 유효 view 위치만 채우기
        feat_all = feat_all.view(b, s, D)                            # [B,S,768]

        # 어떤 view가 padding인지 (image 기준)
        valid_mask = (flat_sum != 0).view(b, s)                      # [B,S]

        # ----------------- 3) PA / LAT 전용 head -----------------
        C = logits_fusion.size(1)
        logits_pa = logits_fusion.new_zeros(b, C)
        logits_lat = logits_fusion.new_zeros(b, C)

        # view 0 → PA라고 가정
        if s >= 1:
            fv_pa = feat_all[:, 0, :]                                # [B,768]
            fv_pa_seq = fv_pa.unsqueeze(1)                           # [B,1,768]
            # mask: [1,B], True = padding 위치
            pad_pa = (valid_mask[:, 0] == 0)                         # [B]
            mask_pa = pad_pa.unsqueeze(0)                            # [1,B]
            logits_pa = self.head_pa(fv_pa_seq, mask_pa)             # [B,C]

        # view 1 → LAT라고 가정
        if s >= 2:
            fv_lat = feat_all[:, 1, :]                               # [B,768]
            fv_lat_seq = fv_lat.unsqueeze(1)                         # [B,1,768]
            pad_lat = (valid_mask[:, 1] == 0)                        # [B]
            mask_lat = pad_lat.unsqueeze(0)                          # [1,B]
            logits_lat = self.head_lat(fv_lat_seq, mask_lat)         # [B,C]

        return logits_fusion, logits_pa, logits_lat

    def forward_with_features(self, x):
        """
        옛날 EstimatorCV 등과 호환 필요하면 여기서 fvec 다시 계산해서 리턴.
        지금은 fusion logits만 쓰고 있다면 굳이 안 써도 됨.
        """
        logits_fusion, logits_pa, logits_lat = self.forward_with_pa_lat_logits(x)
        # placeholder: fvec을 안 쓰고 있으면 0 텐서로만 채워도 됨
        fvec = logits_fusion.new_zeros(logits_fusion.size(0), 768)
        return logits_fusion, fvec