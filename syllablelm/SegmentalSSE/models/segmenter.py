# c.f. fairseq/fairseq/models/wav2vec/wav2vec2.py, with modification

#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from fairseq import utils
import fairseq

from fairseq.models import BaseFairseqModel
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange
from .wavlm_modules import TransformerEncoder_wavlm
from .crf_module import CRF




import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torchaudio
import time
import soundfile as sf
import sys
sys.path.append('/home/abaade/MAE-AST-Playground/')
sys.path.append('/home/abaade/MAE-AST-Playground/examples')
import examples
sys.path.append('/home/abaade/MAE-AST-Playground/examples/data2vec')
sys.path.append('/home/abaade/MAE-AST-Playground/examples/data2vec/models')

from fairseq import checkpoint_utils

import data2vec2


class Segmenter2(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.is_d2v2, self.is_hubert = False, False
        ckpt_path, self.is_hubert = "/data/scratch/abaade/hubert/hubert_base_ls960.pt", True
        # ckpt_path, self.is_d2v2 = "/data/scratch/abaade/data2vec2/base_libri.pt", True
        
        
        hidden_models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        hidden_model = hidden_models[0]
        hidden_model = hidden_model.eval().cuda().half()
        self.hidden_model = [hidden_model]
        
        #hubert_base_model.half()(torch.zeros((5,16000), device='cuda:0', dtype=torch.float16), 
         #                       padding_mask=torch.zeros((5,16000), device='cuda:0', dtype=torch.bool), 
          #                       features_only=True, mask=False, output_layer=10)['x']
                         
                         
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        model = models[0]
        self.model = model.cuda()
        
        #raw_d2v2_outs = model(torch.zeros((1,32000), device='cuda:0', dtype=torch.float), mode=None, mask=False, features_only=True, remove_extra_tokens=True)
        #print(raw_d2v2_outs)
        #assert False
        
        if self.is_d2v2:
            self.model.cfg.layerdrop = 0
        
        random_init = getattr(self.args, "random_init_last_x", None)
        if random_init is not None and random_init > 0:
            self.model.encoder.layers[-random_init:].apply(init_bert_params)
        
    
    def forward(self, source, padding_mask=None, tgt = None, decode=False, rescore=False, tgt_inner = None):        
        source = source.cuda()
        if padding_mask is not None:
            padding_mask = padding_mask.cuda()
        if tgt is not None:
            tgt = tgt.cuda()
        if tgt_inner is not None:
            tgt_inner = tgt_inner.cuda().int()
        else:
            tgt_inner = tgt.new_ones(tgt.shape).int()
        
        if self.is_hubert:
            model_results = self.model.forward(source, padding_mask=padding_mask, features_only=True, mask=False, output_layer=10)
            x = model_results['x']
        elif self.is_d2v2:
            model_results = self.model(source, padding_mask=padding_mask, mode=None, mask=False, features_only=True, remove_extra_tokens=True, out_layer=-2)
            x = model_results['x']
            
        x = x
        
        if padding_mask is None:
            loss_mask = x.new_ones(x.shape[:2], dtype=bool)
        else:
            loss_mask = ~model_results['padding_mask']

        with torch.no_grad():
            if self.is_d2v2:
                hidden_results = self.hidden_model[0](source.half(), padding_mask=padding_mask, mode=None, mask=False, features_only=True, remove_extra_tokens=True, out_layer=-2)
                y = hidden_results['x'].float()
            elif self.is_hubert:
                hidden_results = self.hidden_model[0].forward(source.half(), padding_mask=padding_mask, features_only=True, mask=False, output_layer=10)
                y = hidden_results['x'].float()
                        
            # 1 right after split location = what we want  0.02 -> 1 at idx 1
            indices = tgt.new_zeros(x.shape[:-1])
            indices[:, :min(x.size(1), tgt.size(1))] = tgt[:, :min(x.size(1), tgt.size(1))]
            tgt_inner = tgt_inner[:, :min(x.size(1), tgt_inner.size(1))]
                        
            indices = indices.cumsum(dim=-1).long()
            indices_expanded = indices.unsqueeze(-1).expand(-1,-1,y.size(-1))
            
            y[~loss_mask] = 0            
            sums = y.new_zeros(y.shape)
            sums.scatter_add_(1, indices_expanded, y * tgt_inner.unsqueeze(-1))
            
            ones = (loss_mask & tgt_inner).long()
            counts = y.new_zeros(y.shape[:-1], dtype=torch.long)
            counts.scatter_add_(1, indices, ones)
                        
            targets = sums.gather(1, indices_expanded)[loss_mask] / counts.gather(1, indices).unsqueeze(-1).clamp(min=1)[loss_mask]  # GIGA TODO
            
        result = {}
        result['crf_loss'] = F.mse_loss(x[loss_mask], targets)
        #print(result['crf_loss'].detach().cpu().item())
         
        #if decode:
        #    result['crf_infer'] = tgt.cpu()
        #    # todo run mincut on like 100 examples max. NVM not possible really yet.
        
        return result


class Segmenter(BaseFairseqModel):

    def __init__(self, args):
        super().__init__()
        self.args = args
        feature_enc_layers = eval(args.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, args.encoder_embed_dim)
            if self.embed != args.encoder_embed_dim and not args.quantize_input
            else None
        )

        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)

        self.feature_grad_mult = args.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = args.num_negatives
        self.cross_sample_negatives = args.cross_sample_negatives
        self.codebook_negatives = args.codebook_negatives
        self.negatives_from_everywhere = args.negatives_from_everywhere

        self.logit_temp = args.logit_temp

        self.diversity_weight = args.diversity_weight

        self.pos_conv = nn.Conv1d(
                self.args.encoder_embed_dim,
                self.args.encoder_embed_dim,
                kernel_size=args.conv_pos,
                padding=args.conv_pos // 2,
                groups=args.conv_pos_groups,
            )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.args.encoder_embed_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())
        if "WavLM" in self.args.load_weights_from:
            # wavlm specific parameters
            args.relative_position_embedding = True
            args.num_buckets = 320 
            args.max_distance = 800 
            args.gru_rel_pos = True
            self.encoder = TransformerEncoder_wavlm(args)
        else:
            self.encoder = TransformerEncoder(args)
        self.layer_norm = LayerNorm(self.embed)

        # self.proj = torch.nn.Linear(self.args.encoder_embed_dim, 1)
        self.proj = torch.nn.Linear(self.args.encoder_embed_dim, 2)
        if self.args.crf:
            self.crf = CRF(2, batch_first=True)

    def forward(self, source, padding_mask=None, tgt = None, decode=False, rescore=False):
        result = {}
        source = source.to(self.proj.weight.device)
        if padding_mask != None:
            padding_mask = padding_mask.to(self.proj.weight.device).bool()
        # return logits, lenghts
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
            if self.args.phase not in ["pretraining", "validate_seg"] and self.args.load_awe_weights_from == self.args.load_weights_from:
                result['conv_feats'] = features.transpose(1,2)
        
        
        features_pen = features.pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None: # resize padding mask according to Conv pooling ratio
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        x = features
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        features, layer_feats = self.encoder(x, padding_mask=padding_mask)

        logits = self.proj(features)
        lengths = (~padding_mask).sum(-1)
        
        #print(logits.shape)
        #print(tgt.shape)
        #print(tgt)
        
        result.update({"logits": logits, "lengths": lengths, "padding_mask": padding_mask})

        if self.args.crf:
            if self.training:
                assert tgt != None
                result['crf_loss'] = -1.0 * self.crf(emissions=logits, tags=tgt[:, :logits.shape[1]].long(), mask=~padding_mask, reduction='mean')
            if self.args.crf_infer or decode:
                result['crf_infer'] = self.crf.decode(emissions=logits, mask=~padding_mask) # a list of list
                if rescore:
                    result['crf_llh'] = self.crf(emissions=logits, tags=result['crf_infer'], mask=~padding_mask, reduction='none') 
                    
        return result

    def get_extra_losses(self, net_output):
        pen = []

        if "features_pen" in net_output and self.feature_grad_mult > 0:
            pen.append(net_output["features_pen"])

        return pen
        
    def carefully_load_state_dict(self, states, load_all=False):
        """
        1) Take care of DataParallel/nn.Module state_dict
        2) Show keys that are not loaded due to size mismatch or not found in model
        """
        random_init_last_x = getattr(self.args, "random_init_last_x", None)
        freeze_first_x = getattr(self.args, "freeze_first_x", None)
        if random_init_last_x != None and not load_all:
            cut = self.args.encoder_layers - random_init_last_x
            assert cut >= 0
            random_init_names = [f'encoder.layers.{i}.' for i in range(cut, self.args.encoder_layers)]
            print(f"randomly reinitialize the weights start with the following: {random_init_names}\n")
        new_states = self.state_dict()
        loaded_keys = []
        for k, v in states.items():
            k = k[7:] if k.startswith('module') else k
            k = k[22:] if k.startswith('w2v_encoder.w2v_model') else k
            k = k[11:] if k.startswith('w2v2_model') else k
            k = k[8:] if k.startswith('encoder.pos_conv') else k
            k = k[22:] if k.startswith('conv1_trm1_conv2_trm2') else k
            k = k.replace("audio_encoder.", "")
            if random_init_last_x != None and not load_all:
                for names in random_init_names:
                    if k.startswith(names):
                        v = torch.tensor([0.0]).to(v.device) # make it so that the size doesn't match
                        break
            if k in new_states and new_states[k].size() == v.size():
                new_states[k] = v
                loaded_keys.append(k)
            else:
                print('Ignoring %s due to not existing or size mismatch' % k)

        non_loaded_keys = set(new_states.keys()) - set(loaded_keys)
        if non_loaded_keys:
            print('\nModel states that do not exist in the seed_dir:')
            for k in sorted(non_loaded_keys):
                print('  %s' % k)
        
        self.load_state_dict(new_states)
        print("")
        if freeze_first_x != None:
            freeze_names =  [f'encoder.layers.{i}.' for i in range(freeze_first_x)]
            for n, p in self.named_parameters():
                for fn in freeze_names:
                    if n.startswith(fn):
                        p.requires_grad = False
                        print(f"disable gradient of weights: {n}")
                        break

    def get_last_selfattention(self, source, tgt_layer=None, padding_mask = None):
        features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is not None: # resize padding mask according to Conv pooling ratio
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        x = self.dropout_input(features)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv
        # ############################################
        if self.args.use_audio_cls_token:
            x = torch.cat([self.cls_token.repeat(x.shape[0],1,1), x], dim=1)
            if padding_mask != None:
                cls_token_padding_mask = torch.zeros((padding_mask.shape[0],1)).to(padding_mask)
                padding_mask = torch.cat([cls_token_padding_mask, padding_mask], dim=1)
        # ############################################
        _, attn_weights = self.encoder.extract_features(x, padding_mask=padding_mask, need_head_weights=True, tgt_layer=tgt_layer)
        return attn_weights            


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        first_conv = True
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
        
        if first_conv:
            in_d = 1
        else:
            in_d = 768
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

# class ConvFeatureExtractionModel(nn.Module):
#     def __init__(
#         self,
#         conv_layers: List[Tuple[int, int, int]],
#         dropout: float = 0.0,
#         mode: str = "default",
#         conv_bias: bool = False,
#     ):
#         super().__init__()

#         assert mode in {"default", "layer_norm"}

#         def block(
#             n_in,
#             n_out,
#             k,
#             stride,
#             is_layer_norm=False,
#             is_group_norm=False,
#             conv_bias=False,
#         ):
#             def make_conv():
#                 conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
#                 nn.init.kaiming_normal_(conv.weight)
#                 return conv

#             assert (
#                 is_layer_norm and is_group_norm
#             ) == False, "layer norm and group norm are exclusive"

#             if is_layer_norm:
#                 return nn.Sequential(
#                     make_conv(),
#                     nn.Dropout(p=dropout),
#                     nn.Sequential(
#                         TransposeLast(),
#                         Fp32LayerNorm(dim, elementwise_affine=True),
#                         TransposeLast(),
#                     ),
#                     nn.GELU(),
#                 )
#             elif is_group_norm:
#                 return nn.Sequential(
#                     make_conv(),
#                     nn.Dropout(p=dropout),
#                     Fp32GroupNorm(dim, dim, affine=True),
#                     nn.GELU(),
#                 )
#             else:
#                 return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

#         in_d = 1
#         self.conv_layers = nn.ModuleList()
#         for i, cl in enumerate(conv_layers):
#             assert len(cl) == 3, "invalid conv definition: " + str(cl)
#             (dim, k, stride) = cl

#             self.conv_layers.append(
#                 block(
#                     in_d,
#                     dim,
#                     k,
#                     stride,
#                     is_layer_norm=mode == "layer_norm",
#                     is_group_norm=mode == "default" and i == 0,
#                     conv_bias=conv_bias,
#                 )
#             )
#             in_d = dim

#     def forward(self, x):

#         # BxT -> BxCxT
#         x = x.unsqueeze(1)

#         for conv in self.conv_layers:
#             x = conv(x)

#         return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer_use = args.layer_use
        assert args.layer_use < args.encoder_layers, f"w2v2 only has {args.encoder_layers} layers, but you want layer feat from layer {args.layer_use+1}"
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, superb=False, tgt_layer=None, need_attention_weights=False, pre_feats=False):
        x = x.to(self.layer_norm.bias.data.dtype)
        if superb:
            assert not self.layer_norm_first
            all_feats = self.extract_features(x, padding_mask = padding_mask, all_hidden_states=True)
            return all_feats
        # print("1386", x.dtype)
        x, layer_feats = self.extract_features(x, padding_mask = padding_mask, tgt_layer = tgt_layer, need_head_weights=need_attention_weights, pre_feats=pre_feats)

        if self.layer_norm_first:
            x = self.layer_norm(x)
            layer_feats = self.layer_norm(layer_feats)

        return x, layer_feats

    def extract_features(self, x, padding_mask=None, need_head_weights=False, tgt_layer=None, all_hidden_states=False, pre_feats=False):
        if tgt_layer == None:
            layer_use = self.layer_use
            stop_pass = False
        else:
            layer_use = tgt_layer
            stop_pass = True

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            # print(i)
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask,need_head_weights=need_head_weights)
                layer_results.append(x.transpose(0,1))
                if i == layer_use:
                    layer_feats = x.transpose(0, 1)
                    if need_head_weights:
                        if len(z.shape) == 3:
                            z = z.unsqueeze(0)
                        attn_weights = z # [bsz, num_heads, tgt_len, src_len]
                        if pre_feats:
                            return layer_results[-2], attn_weights
                        else:
                            return layer_feats, attn_weights
                    # if need_head_weights:
                        # # print(z.shape)
                        # if len(z.shape) == 3:
                        #     z = z.unsqueeze(0)
                        # attn_weights = z # [bsz, num_heads, tgt_len, src_len]
                        # # cls_attn_weights = attn_weights[:,:,0,1:] # [bsz,n_heads,src_len-1]
                        # # return cls_attn_weights
                        # return attn_weights
                    # else
                    if stop_pass:
                        return layer_feats, layer_feats
        if all_hidden_states:
            return layer_results
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_feats

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            temp = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
            )
            if len(temp) == 3:
                x, attn, _  = temp
            elif len(temp) == 2:
                x, attn = temp
            else:
                print(f"length of self_attn should be either 2 or 3, but it's {len(temp)}")
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            temp = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights
            )
            if len(temp) == 3:
                x, attn, _  = temp
            elif len(temp) == 2:
                x, attn = temp
            else:
                print(f"length of self_attn should be either 2 or 3, but it's {len(temp)}")

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


