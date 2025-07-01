import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import II
import math

from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.data.data_utils import compute_mask_indices
from fairseq.modules import GradMultiply

from fairseq.models.valle.valle_alan import sin_pos_embed, ValleBlock, ValleEncoder, ValleAlanBase, ValleBlockMaskMoE


# from .configuration_opt import OPTConfig
# from .modeling_opt import OPTForCausalLM
#
@dataclass
class ClusterAlanInterleavedConfig(FairseqDataclass):
    # model_string: str = "facebook/opt-125m"
    # vocab_size_override: int = 1024
    vocab_size: int = field(
        default=1024 + 4096 + 3, metadata={"help": "(max) Size of vocab"}
    )

    depth: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    mlp_ratio: float = 4
    mlp_ratio_2: Optional[float] = None

    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.0
    drop_path: float = 0.0
    layerdrop: float = 0.0

    layer_norm_first = True
    max_position_embeddings = 2048

    use_ngram_noising: bool = field(
        default=False, metadata={"help": "Add noising schedule"}
    )
    mask_prob: float = 0.10
    mask_length: int = 5

    # Feels smart to keep these?
    max_update: int = II("optimization.max_update")
    seed: int = II("common.seed")

    use_learned_pos_embed: bool = True

    interleave_style: str = field(
        default='simple', metadata={"help": "simple: just one large vocab size, no moe"}
    )
    is_moe: bool = False
    moe_checkpoint: Optional[str] = None
    load_from_checkpoint_and_swap: Optional[str] = None
    load_from_checkpoint_source: Optional[str] = None
    load_from_checkpoint_target: Optional[str] = None


@register_model("cluster_alan_interleaved", dataclass=ClusterAlanInterleavedConfig)
class ClusterAlanModel(BaseFairseqModel):
    def __init__(self, cfg: ClusterAlanInterleavedConfig):
        super().__init__()

        # self.transformers_config = OPTConfig.from_pretrained(cfg.model_string)
        # self.transformers_config.vocab_size = cfg.vocab_size_override
        # self.model = OPTForCausalLM(self.transformers_config)

        self.cfg = cfg

        self.vocab_size = cfg.vocab_size + 1

        self.embed_weight = nn.Parameter(torch.empty(self.vocab_size, cfg.embed_dim).normal_(0, 1 / math.sqrt(cfg.embed_dim)))
        self.stop_token_idx = self.vocab_size

        if hasattr(cfg, 'use_learned_pos_embed') and cfg.use_learned_pos_embed:
            self.pe = nn.Parameter(torch.empty(1, 3000, cfg.embed_dim).normal_(0, 1 / math.sqrt(cfg.embed_dim)))
        else:
            self.register_buffer('pe', sin_pos_embed(cfg.embed_dim, 3000))

        self.encoder = ValleEncoder(
            BlockType=ValleBlockMaskMoE if self.cfg.is_moe else ValleBlock,
            depth=cfg.depth,
            dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop=cfg.encoder_dropout,
            attn_drop=cfg.attention_dropout,
            post_mlp_drop=cfg.post_mlp_drop,
            layer_norm_first=True,
            is_causal=True,
            mlp_ratio_2=cfg.mlp_ratio_2,
        )

        self.apply(init_bert_params)

        if self.cfg.moe_checkpoint is not None:
            assert self.cfg.is_moe
            state_dict = torch.load(self.cfg.moe_checkpoint)['model']
            remaining_keys = self.load_state_dict(state_dict, strict=False)
            assert len(remaining_keys.unexpected_keys) == 48
            for idx in range(cfg.depth):
                prefix = f'encoder.blocks.{idx}.mlp.'
                for expert in ['mlp1', 'mlp2']:
                    getattr(self.encoder.blocks[idx], expert).load_state_dict({
                        suffix: state_dict[prefix + suffix] for suffix in
                        ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
                    }, strict=True)

                for p in self.encoder.blocks[idx].mlp1.parameters():
                    p.param_group = "long_moe"

        if cfg.load_from_checkpoint_and_swap is not None:
            reducer_source = np.load(cfg.load_from_checkpoint_source)
            reducer_target = np.load(cfg.load_from_checkpoint_target)
            long_offset = 1024 + 1
            vocab_size_old = reducer_source.max() + 1
            vocab_size_new = reducer_target.max() + 1
            state_dict = torch.load(cfg.load_from_checkpoint_and_swap)
            old_embs = state_dict['model']['embed_weight']
            embs = old_embs[long_offset: long_offset + vocab_size_old][reducer_source]

            self.embed_weight.requires_grad_(False)

            new_embs = torch.zeros((vocab_size_new, cfg.embed_dim))
            new_embs.scatter_add_(0, torch.from_numpy(reducer_target).view(-1, 1).repeat(1, cfg.embed_dim), embs)
            ones = torch.ones((reducer_source.shape[0],))
            counts = torch.zeros((vocab_size_new,))
            counts.scatter_add_(0, torch.from_numpy(reducer_target), ones)
            new_embs = new_embs / counts.view(-1, 1)

            self.embed_weight[:long_offset] = old_embs[:long_offset]
            self.embed_weight[-3:] = old_embs[-3:]
            self.embed_weight[long_offset:-3] = new_embs
            self.embed_weight.requires_grad_(True)

            del state_dict['model']['embed_weight']
            self.load_state_dict(state_dict["model"], strict=False)

    @classmethod
    def build_model(cls, cfg: ClusterAlanInterleavedConfig, task: None):
        """Build a new model instance."""

        model = ClusterAlanModel(cfg)
        return model

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def pos_embed(self, features, offset=0):
        # Features: BTC tensor
        return features + self.pe[:, offset:features.size(1) + offset]

    # @torch.compile()
    def forward(
            self,
            interleaved_tokens,
            interleaved_padding_mask=None,
            short_tokens=None,
            long_tokens=None,
            predicts_short_indices=None,
            predicts_long_indices=None,
            is_long=None,
            features_only: bool = False,
    ):
        # prepend bos
        features = self.embed_weight[interleaved_tokens]
        features = self.pos_embed(features)  # giga todo update later

        x, hidden_states = self.encoder(features, padding_mask=interleaved_padding_mask, moe_mask=is_long)
        logits = self.embed_weight @ x.transpose(-1, -2)

        if features_only:
            return {"x": x, "padding_mask": interleaved_padding_mask, "hidden_states": hidden_states, "logits": logits}

        short_logits = logits.gather(-1, predicts_short_indices[:, None, :].expand(logits.size(0), logits.size(1), predicts_short_indices.size(-1)))
        short_loss = F.cross_entropy(short_logits, short_tokens, ignore_index=-100)

        if predicts_long_indices.size(-1) != 0:
            long_logits = logits.gather(-1, predicts_long_indices[:, None, :].expand(logits.size(0), logits.size(1), predicts_long_indices.size(-1)))
            long_loss = F.cross_entropy(long_logits, long_tokens, ignore_index=-100)
        else:
            long_loss = (short_loss * 0.).detach()

        # short_loss = GradMultiply.apply(short_loss, 1/2)  # Normalizes gradient norms at 100k steps, I think

        with torch.no_grad():
            stop_padded = short_tokens.clone()
            stop_padded[short_tokens == 1024] = -100
            short_loss_no_stop = F.cross_entropy(short_logits, stop_padded, ignore_index=-100)
            short_logits_no_stop = short_logits.clone()
            short_logits_no_stop[:, 1024] = -100
            short_loss_no_stop_v2 = F.cross_entropy(short_logits_no_stop, stop_padded, ignore_index=-100)

            # latter_half_idx = long_logits.size(-1) // 2
            # long_loss_latter_half = F.cross_entropy(long_logits[:, :, latter_half_idx:], long_tokens[:, latter_half_idx:], ignore_index=-100)

        result = {
            "logs": {"mhubert_25hz_no_stop": short_loss_no_stop, "mhubert_25hz_no_stop_v2": short_loss_no_stop_v2},
            "losses": {"mhubert_25hz_loss": short_loss, "sdhubert_loss": long_loss},
            "sample_size": 1,
        }

        return result
