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

from fairseq.models.valle.valle_alan import sin_pos_embed, ValleBlock, ValleEncoder, ValleAlanBase


# from .configuration_opt import OPTConfig
# from .modeling_opt import OPTForCausalLM
#
@dataclass
class ClusterAlanConfig(FairseqDataclass):
    # model_string: str = "facebook/opt-125m"
    # vocab_size_override: int = 1024
    vocab_size: int = field(
        default=1024, metadata={"help": "(max) Size of vocab"}
    )

    depth: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    mlp_ratio: float = 4

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


@register_model("cluster_alan", dataclass=ClusterAlanConfig)
class ClusterAlanModel(BaseFairseqModel):
    def __init__(self, cfg: ClusterAlanConfig):
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
            BlockType=ValleBlock,
            depth=cfg.depth,
            dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop=cfg.encoder_dropout,
            attn_drop=cfg.attention_dropout,
            post_mlp_drop=cfg.post_mlp_drop,
            layer_norm_first=True,
            is_causal=True,
        )

        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, cfg: ClusterAlanConfig, task: None):
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
            input_tokens: Tensor,
            padding_mask: Tensor = None,
            features_only: bool = False,
            use_stop: bool = True,  # True during training, false during inference. Use encodec stop rename todo
            verbose: bool = False,
    ):
        # prepend bos
        features = self.embed_weight[input_tokens]
        features = self.pos_embed(features)

        x, hidden_states = self.encoder(features, padding_mask)

        outs = x[:, :-1] if use_stop else x  # last token always stop (dataset), tiny perf hit
        logits = self.embed_weight @ outs.transpose(-1, -2)

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "layer_results": hidden_states, "logits": logits}

        y = input_tokens[:, 1:]
        nll_loss = F.cross_entropy(logits, y, ignore_index=-100, reduction='none' if verbose else 'mean')

        result = {
            "logs": {},
            "losses": {"nll_loss": nll_loss},
            "sample_size": 1
        }

        return result
