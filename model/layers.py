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
        """
        b, s, _, _, _ = x.shape

        # 1. Image Flatten & Valid Masking
        x_flat = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        flat_sum = x_flat.sum(dim=(1, 2, 3))
        no_pad = torch.nonzero(flat_sum != 0).squeeze(1) # 유효 인덱스

        x_valid = x_flat[no_pad]

        # 2. Backbone Feature Extraction
        with torch.no_grad():
            feats = self.model(x_valid).detach() # [N_valid, 768, H1, W1]

        # 3. Projection & Positional Encoding (Shared)
        # Fusion과 View Head 모두 동일한 '위치 정보가 담긴 Feature'를 쓰는 게 유리함
        x_enc = self.conv2d(feats)    # [N_valid, 768, H2, W2]
        x_enc = self.pos_encoding(x_enc)

        # ─────────── [Path A] Fusion Transformer 입력 준비 ───────────
        # (작성하신 코드와 동일 로직: Padding Token 채우기)
        pad_tokens = einops.repeat(
            self.padding_token, '1 c 1 1 -> (b s) c h w',
            b=b, s=s, h=x_enc.shape[2], w=x_enc.shape[3]
        ).type_as(x_enc)
        
        segment_embedding = einops.repeat(
            self.segment_embedding, 's c 1 1 -> (b s) c h w',
            b=b, h=x_enc.shape[2], w=x_enc.shape[3]
        ).type_as(x_enc)

        # 유효한 곳에 Feature + Seg Embedding 할당
        pad_tokens[no_pad] = x_enc + segment_embedding[no_pad]
        
        # [B*S, C, H, W] -> [B, S*H*W, C] (Sequence로 변환)
        x_seq_fusion = einops.rearrange(
            pad_tokens, '(b s) c h w -> b (s h w) c', b=b, s=s
        )

        # Masking (Padding 부분 0 처리)
        mask_tokens = (x_seq_fusion.sum(dim=-1) == 0).transpose(0, 1) # [L, B]

        # Fusion Forward
        x_trans = self.transformer_encoder(x_seq_fusion, src_key_padding_mask=mask_tokens)
        logits_fusion = self.head_fusion(x_trans, mask_tokens) # [B, num_classes]


        # ─────────── [Path B] View-specific Heads (수정된 부분) ───────────
        # 핵심: GAP를 쓰지 않고, Spatial Feature (H*W)를 그대로 가져옴
        
        # 전체 배치(B*S)에 대한 빈 텐서 준비 [B*S, 768, H2, W2]
        feat_all_spatial = pad_tokens.new_zeros(
            b * s, x_enc.shape[1], x_enc.shape[2], x_enc.shape[3]
        )
        # 유효한 Feature만 채워넣음 (Segment Embedding은 뺀 순수 Feature 추천)
        # 주의: x_enc는 Valid한 개수만큼만 있으므로 no_pad 위치에 scatter
        feat_all_spatial[no_pad] = x_enc 
        
        # [B, S, C, H, W] 형태로 복원
        feat_all_spatial = einops.rearrange(
            feat_all_spatial, '(b s) c h w -> b s c h w', b=b, s=s
        )

        C_cls = logits_fusion.size(1)
        logits_pa = logits_fusion.new_zeros(b, C_cls)
        logits_lat = logits_fusion.new_zeros(b, C_cls)

        # ── Head PA (View 0) ──
        if s >= 1:
            # View 0의 Spatial Feature 가져오기: [B, C, H, W]
            feat_pa = feat_all_spatial[:, 0, :, :, :] 
            
            # ML-Decoder 입력 형태: [B, L, C] = [B, H*W, 768]
            feat_pa_seq = einops.rearrange(feat_pa, 'b c h w -> b (h w) c')
            
            # Mask 생성 (이미지 전체가 0인 경우 Padding 처리)
            # [B, H*W] -> [1, B] (MLDecoder 구현체에 따라 mask shape 확인 필요)
            # 보통 MLDecoder는 key_padding_mask가 필요 없거나 [B, L]을 원함
            # 여기서는 feat 자체가 0이면 Attention 안하도록 처리되길 기대하거나,
            # 명시적으로 valid_mask를 확장해서 넣어줌
            
            # (view가 아예 없는 샘플에 대한 마스킹은 Loss단에서 처리하므로 여기선 Pass 가능)
            logits_pa = self.head_pa(feat_pa_seq) 

        # ── Head LAT (View 1) ──
        if s >= 2:
            feat_lat = feat_all_spatial[:, 1, :, :, :]
            feat_lat_seq = einops.rearrange(feat_lat, 'b c h w -> b (h w) c')
            logits_lat = self.head_lat(feat_lat_seq)

        return logits_fusion, logits_pa, logits_lat
    
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


class FusionBackboneMultiView2(nn.Module):
    def __init__(self, timm_init_args, pretrained_path=None, num_classes=26):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        # stage1에서 썼던 MLDecoder head 로드 후 backbone weight만 사용
        # (MLDecoder Class가 정의되어 있다고 가정)
        self.model.head = MLDecoder(num_classes=num_classes, initial_num_features=768)
        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        # backbone feature만 쓰도록 head 제거
        self.model.head = nn.Identity()

        self.conv2d = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1)
        # (PositionalEncoding2D, Summer Class가 정의되어 있다고 가정)
        self.pos_encoding = Summer(PositionalEncoding2D(768))
        self.padding_token = nn.Parameter(torch.randn(1, 768, 1, 1))

        # 최대 view 개수(=segment embedding 개수)
        self.segment_embedding = nn.Parameter(torch.randn(4, 768, 1, 1))
        self.max_views = self.segment_embedding.shape[0]

        # ────────────── ML-Decoder 3개 ──────────────
        # 1) 멀티뷰 fusion용 (기존 head)
        self.head_fusion = MLDecoder(num_classes=num_classes, initial_num_features=768)
        # 2) PA view용 (view index 0, 1)
        self.head_pa = MLDecoder(num_classes=num_classes, initial_num_features=768)
        # 3) LAT view용 (view index 2, 3)
        self.head_lat = MLDecoder(num_classes=num_classes, initial_num_features=768)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=2
        )

    def forward(self, x):
        # 기존 코드 호환을 위해 fusion 결과만 리턴하거나, 
        # 학습 때는 tuple로 리턴하는 것이 일반적입니다.
        return self.forward_with_pa_lat_logits(x)

    def forward_with_pa_lat_logits(self, x):
        """
        x: [B, S=4, 3, H, W] -> Dataset에서 [PA, PA, LAT, LAT] 순서로 줌
        """
        b, s, _, _, _ = x.shape

        # 1. Image Flatten & Valid Masking
        x_flat = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        flat_sum = x_flat.sum(dim=(1, 2, 3))
        # 패딩이 아닌 유효한 이미지 인덱스
        no_pad = torch.nonzero(flat_sum != 0).squeeze(1) 

        x_valid = x_flat[no_pad]

        # 2. Backbone Feature Extraction (유효한 것만)
        with torch.no_grad():
            feats = self.model(x_valid).detach() # [N_valid, 768, H1, W1]

        # 3. Projection & Positional Encoding
        x_enc = self.conv2d(feats)    
        x_enc = self.pos_encoding(x_enc) # [N_valid, 768, H2, W2]
        
        # Feature map size
        h_enc, w_enc = x_enc.shape[2], x_enc.shape[3]

        # 4. [B, S] 구조 복원 및 Segment Embedding
        # 전체를 Padding Token으로 초기화
        pad_tokens = einops.repeat(
            self.padding_token, '1 c 1 1 -> (b s) c h w',
            b=b, s=s, h=h_enc, w=w_enc
        ).type_as(x_enc)
        
        segment_embedding = einops.repeat(
            self.segment_embedding, 's c 1 1 -> (b s) c h w',
            b=b, h=h_enc, w=w_enc
        ).type_as(x_enc)

        # 유효한 위치에 Feature + SegEmb 할당
        pad_tokens[no_pad] = x_enc + segment_embedding[no_pad]
        
        # [B, S, C, H, W] 형태로 Reshape
        feats_all = einops.rearrange(
            pad_tokens, '(b s) c h w -> b s c h w', b=b, s=s
        )

        # ─────────── Mask 생성 로직 ───────────
        # 각 이미지(S)가 Padding인지 여부 확인 [B, S]
        # flat_sum은 [B*S] 형태
        valid_mask_bs = (flat_sum != 0).view(b, s) # True=Valid, False=Padding
        
        # ML-Decoder용 마스크 생성 함수 (True = Padding = Ignore)
        def create_mask(mask_sub_valid): # mask_sub_valid: [B, sub_S]
            # 이미지가 유효하지 않으면(False), 해당 이미지의 모든 픽셀(H*W)을 마스킹(True)
            is_padding = ~mask_sub_valid
            return einops.repeat(is_padding, 'b s -> b (s l)', l=h_enc*w_enc)

        # [함수] Feature를 Sequence로 변환 [B, S, C, H, W] -> [B, S*H*W, C]
        def feats_to_seq(f):
            return einops.rearrange(f, 'b s c h w -> b (s h w) c')


        # ─────────── [Path A] Fusion Forward ───────────
        # 모든 View (0~3) 사용
        x_seq_fusion = feats_to_seq(feats_all) # [B, 4*HW, C]
        mask_fusion = create_mask(valid_mask_bs) # [B, 4*HW]

        # Transformer (Cross-view Interaction)
        # Note: PyTorch Transformer는 (L, B, E) 혹은 batch_first=True면 (B, L, E)
        # 작성해주신 init에는 batch_first가 없으므로 (L, B, E)로 변환 필요할 수 있음
        # 하지만 TransformerEncoderLayer 기본값은 batch_first=False임.
        # 작성해주신 코드 흐름상 batch_first=False (L, B, E)를 가정하고 transpose 하는 듯 함.
        
        x_seq_fusion_t = x_seq_fusion.transpose(0, 1) # [L, B, C]
        
        # src_key_padding_mask는 [B, L] 형태여야 함
        x_trans = self.transformer_encoder(x_seq_fusion_t, src_key_padding_mask=mask_fusion)
        
        # MLDecoder 입력은 [B, L, C]를 선호하는 경우가 많음 (구현체 확인 필요)
        # 여기서는 작성해주신 기존 코드 스타일(transpose 안 된 상태로 넘김?)을 따르거나,
        # MLDecoder가 (L, B, C)를 받는지 (B, L, C)를 받는지에 따라 다름.
        # 보통 MLDecoder는 (B, L, C)를 받으므로 다시 transpose
        x_trans = x_trans.transpose(0, 1) # [B, L, C]
        
        logits_fusion = self.head_fusion(x_trans, mask_fusion)


        # ─────────── [Path B] Multi-Instance View Heads ───────────
        
        # 1. PA Head (Index 0, 1)
        # PA가 2장이면 2장 데이터를 다 씁니다 (Average Pooling 안함)
        feats_pa = feats_all[:, 0:2, ...]
        seq_pa = feats_to_seq(feats_pa) # [B, 2*HW, C]
        mask_pa = create_mask(valid_mask_bs[:, 0:2])
        
        logits_pa = self.head_pa(seq_pa, mask_pa)


        # 2. LAT Head (Index 2, 3)
        feats_lat = feats_all[:, 2:4, ...]
        seq_lat = feats_to_seq(feats_lat) # [B, 2*HW, C]
        mask_lat = create_mask(valid_mask_bs[:, 2:4])
        
        logits_lat = self.head_lat(seq_lat, mask_lat)

        return logits_fusion, logits_pa, logits_lat