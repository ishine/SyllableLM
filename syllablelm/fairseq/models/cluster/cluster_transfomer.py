import argparse
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import II

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from fairseq.modules import GradMultiply, LayerNorm


class DropPath(nn.Module):  # ALAN ADDITION BECAUSE TIMM IMPORT ERROR
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
                x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


class Mlp(nn.Module):  # ALAN ADDITION BECAUSE TIMM IMPORT ERROR
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    https://huggingface.co/spaces/Roll20/pet_score/blame/9e46325ff5d82df348bad5b4a235eac8410959b8/lib/timm/models/layers/mlp.py
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


@dataclass
class ValleAlanConfig(FairseqDataclass):
    ar_checkpoint_path: str = ""
    nar_checkpoint_path: str = ""

    # Feels smart to keep these?
    max_update: int = II("optimization.max_update")
    seed: int = II("common.seed")

    # ar_checkpoint


@register_model("valle_alan", dataclass=ValleAlanConfig)
class ValleAlanModel(BaseFairseqModel):
    def __init__(self, cfg: ValleAlanConfig):
        super().__init__()

        from fairseq.checkpoint_utils import load_model_ensemble
        self.ar_model = load_model_ensemble(cfg.ar_checkpoint_path)
        self.nar_model = load_model_ensemble(cfg.nar_checkpoint_path)

        self.cfg = cfg

    @classmethod
    def build_model(cls, cfg: ValleAlanConfig, task: None):
        """Build a new model instance."""

        model = ValleAlanModel(cfg)
        return model

    def sample(self):
        pass

    def forward(
            self,
            source,
            # text_list: list[Tensor],
            # proms_list: list[Tensor],
            # resps_list: list[Tensor],
    ):
        pass


# adapted from d2v2 alt attention
class ValleAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal

    def forward(self, x, padding_mask=None, causal_start=0):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)  # qkv x B x H x L x D
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])

        dtype = q.dtype

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        if self.is_causal:  # Assumes padded text + identical prompt length
            causal_mask = x.new_ones((N, N), dtype=torch.bool).tril_()
            causal_mask[:, :causal_start] = 1
            causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ValleFlashAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal

    def forward(self, x, padding_mask=None, causal_start=0, past=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)  # qkv x B x H x L x D
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])
        present = torch.stack([k, v])

        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pk, v], dim=-2)

        attn_mask = x.new_ones((B, self.num_heads, N, N), dtype=torch.bool)
        if padding_mask is not None and padding_mask.any():
            attn_mask.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), 0)

        if self.is_causal:  # Assumes padded text + identical prompt length
            causal_mask = x.new_ones((N, N), dtype=torch.bool).tril_()
            causal_mask[:, :causal_start] = 1
            causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(0)
            attn_mask.masked_fill_(causal_mask, 0)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop if self.training else 0)

        # def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        #     # Efficient implementation equivalent to the following:
        #     L, S = query.size(-2), key.size(-2)
        #     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        #     attn_bias = torch.zeros(L, S, dtype=query.dtype)
        #     if is_causal:
        #         assert attn_mask is None
        #         temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        #         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        #         attn_bias.to(query.dtype)
        #
        #     if attn_mask is not None:
        #         if attn_mask.dtype == torch.bool:
        #             attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        #         else:
        #             attn_bias += attn_mask
        #     attn_weight = query @ key.transpose(-2, -1) * scale_factor
        #     attn_weight += attn_bias
        #     attn_weight = torch.softmax(attn_weight, dim=-1)
        #     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        #     return attn_weight @ value
        #
        # x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop if self.training else 0)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, present


class ValleBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            mlp_drop=0.0,
            post_mlp_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_norm_first=True,
            is_causal=False,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first

        self.norm1 = norm_layer(dim)
        self.attn = ValleFlashAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

    def forward(self, x, padding_mask=None, causal_start=0, past=None):
        if self.layer_norm_first:
            attn, present = self.attn(self.norm1(x), padding_mask, causal_start, past)
            x = x + self.drop_path(attn)
            r = x
            x = self.mlp(self.norm2(x))
            x = r + self.drop_path(self.post_mlp_dropout(x))
        else:
            attn, present = self.attn(self.norm1(x), padding_mask, causal_start, past)
            x = x + self.drop_path(attn)
            x = self.norm1(x)
            r = x
            x = self.mlp(x)
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))

        return x, present



class ValleBlockAdaLNZero(ValleBlock):
    def __init__(
            self,
            dim,
            *args,
            **kwargs,
    ):
        super().__init__(dim, *args, **kwargs)

        self.adalnz_mlp = nn.Linear(dim, dim * 6)
        # todo zero init these

    def forward(self, x, padding_mask=None, causal_start=0, adalnz_embed=None):
        adalnz_y1, adalnz_b1, adalnz_a1, adalnz_y2, adalnz_b2, adalnz_a2 = (
            self.adalnz_mlp(adalnz_embed).chunk(6, dim=-1)
        )
        if self.layer_norm_first:
            x = x + self.drop_path(
                (1 + adalnz_a1) * self.attn(
                    self.norm1(x) * (1 + adalnz_y1) + adalnz_b1,
                    padding_mask,
                    causal_start,
                )[0]
            )
            r = x
            x = (1 + adalnz_a2) * self.mlp(
                self.norm2(x) * (1 + adalnz_y2) + adalnz_b2
            )
            x = r + self.drop_path(self.post_mlp_dropout(x))
        else:
            raise NotImplementedError()
            x = x + self.drop_path(self.attn(x, padding_mask, causal_start))
            x = self.norm1(x)
            r = x
            x = self.mlp(x)
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))

        return x, None


class ValleEncoder(nn.Module):
    def __init__(
            self,
            BlockType=ValleBlock,  # ValleBlock or ValleBlockAdaLNZero
            depth=12,
            use_simple_moe:bool=False,
            *args,
            **kwargs
    ):
        super().__init__()
        if use_simple_moe:
            self.blocks = nn.ModuleList(
                [BlockType(*args, **kwargs, ) if _ % 2 == 1 else ValleBlockEasyMoE(*args, **kwargs, ) for _ in range(depth)]
            )
        else:
            self.blocks = nn.ModuleList(
                [BlockType(*args, **kwargs, ) for _ in range(depth)]
            )

    def forward(self, x, padding_mask=None, **kwargs):
        hidden_states = [x]  # GIGA TODO???
        for idx, blk in enumerate(self.blocks):
            x, present = blk(x, padding_mask, **kwargs)
            hidden_states.append(x)
        return x, hidden_states

    def forward_cache(self, x, padding_mask=None, past=None, **kwargs):
        hidden_states = [x]  # GIGA TODO???
        past = None if past is None else torch.unbind(past, 0)
        presents = []
        for idx, blk in enumerate(self.blocks):
            x, present = blk(x, padding_mask, past=past, **kwargs)
            hidden_states.append(x)
            presents.append(present)
        return x, hidden_states, torch.stack(presents)


class ValleBlockEasyMoE(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            mlp_drop=0.0,
            post_mlp_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_norm_first=True,
            is_causal=False,
            *args,
            **kwargs,
    ):
        # if it predicts a mhubert token, it should be on part 2. Therefore, the second units should have a BOS token (Not necessary for Valle, kinda). If remove loss masking, whatever. Later problem
        super().__init__()

        self.layer_norm_first = layer_norm_first

        self.norm1 = norm_layer(dim)
        self.attn = ValleFlashAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)
        self.dumb = True

    def forward(self, x, padding_mask=None, causal_start=0, past=None):
        if self.layer_norm_first:
            attn, present = self.attn(self.norm1(x), padding_mask, causal_start, past)
            x = x + self.drop_path(attn)
            r = x
            x = self.norm2(x)

            if self.dumb:
                x1 = self.mlp1(x[:, :causal_start])
                x2 = self.mlp2(x[:, causal_start:])
                x = torch.cat([x1, x2], dim=1)
            else:
                x = self.mlp1(x)

            x = r + self.drop_path(self.post_mlp_dropout(x))
        else:
            attn, present = self.attn(self.norm1(x), padding_mask, causal_start, past)
            x = x + self.drop_path(attn)
            x = self.norm1(x)
            r = x

            x1 = self.mlp1(x[:, :causal_start])
            x2 = self.mlp2(x[:, causal_start:])
            x = torch.cat([x1, x2],dim=1)

            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))

        return x, present



def sin_pos_embed(embed_dim, max_length=1000):
    position = torch.arange(max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(1, max_length, embed_dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class ValleAlanBase(BaseFairseqModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # text input is 0 indexed
        self.num_text_tokens = cfg.num_text_tokens + 1
        self.num_codec_tokens = cfg.num_codec_tokens + 1

        self.phone_embed_weight = nn.Parameter(torch.empty(self.num_text_tokens, cfg.embed_dim).normal_(0, 0.02))
        self.encodec_embed_weight = nn.Parameter(torch.empty(self.num_codec_tokens, cfg.encodec_depth, cfg.embed_dim).normal_(0, 0.02))
        self.stop_token_idx = self.num_codec_tokens - 1

        self.num_encodec_same_speaker = cfg.encodec_depth if cfg.use_full_speech else 1

        self.register_buffer('pe', sin_pos_embed(cfg.embed_dim, 3000))

    def pos_embed(self, features, offset=0):
        # Features: BTC tensor
        return features + self.pe[:, offset:features.size(1) + offset]
