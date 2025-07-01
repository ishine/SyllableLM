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
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from fairseq import utils

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
class Embedder(BaseFairseqModel):

    def __init__(self, args):
        super().__init__()
        self.args = args
        awe_args = argparse.Namespace(**vars(args))

        awe_args.encoder_layers = args.awe_encoder_layers
        self.awe = AWE(awe_args)

        # the combiner
        combiner_args = argparse.Namespace(**vars(args))
        combiner_args.encoder_layers = self.args.combiner_encoder_layers
        self.proj = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.pos_embed = SinusoidalPositionalEncoding(d_model=self.args.encoder_embed_dim, max_len=1000)
        self.layer_norm = nn.LayerNorm(self.args.encoder_embed_dim)
        self.combiner = TransformerEncoder(combiner_args)
 
        if self.args.use_audio_cls_token:
            self.cls_token = torch.nn.Parameter(torch.randn((1, 1, args.encoder_embed_dim)), requires_grad=True)

    def forward(self, source, seg):
        assert len(source) > 1, "currently it doesn't support batch size == 1!"
        source = source.to(self.awe.layer_norm.weight.device)
        # print(source.shape)
        len_fact = 1 if (len(source.shape) == 3 and source.shape[2] == 512) else self.args.downsample_factor # 320 by default

        in_awe = []
        in_length = []
        processed_seg = []
        # print(seg)
        for cur_feat, cur_seg in zip(source, seg):
            # print(cur_seg, flush=True)
            # print(cur_feat)
            # print(cur_feat.shape)
            # print(cur_seg)
            temp = [cur_feat[int(l*len_fact):int(r*len_fact)] for l,r in cur_seg]
            cur_in_length = [(r-l)*len_fact for l,r in cur_seg]
            cur_processed_seg = cur_seg
            if len(temp) > self.args.max_num_words:
                if sum(cur_in_length) > len(cur_in_length): # not all segments are of length 1
                    temp = [f for f, l in zip(temp, cur_in_length) if l > 1]
                    cur_in_length = [cl for cl in cur_in_length if cl > 1]
                    cur_processed_seg = [cps for cps, l in zip(cur_processed_seg, cur_in_length) if l>1]
                if len(temp) > self.args.max_num_words: # still more than mex_num_words, we will randomly drop (but keep the order correct!)
                    ind = list(range(len(temp)))
                    random.shuffle(ind)
                    ind_use = np.sort(ind[:self.args.max_num_words]).tolist()
                    temp = [temp[i] for i in ind_use]
                    cur_in_length = [cur_in_length[i] for i in ind_use]
                    cur_processed_seg = [cur_processed_seg[i] for i in ind_use]
            in_awe = in_awe + temp
            in_length = in_length + cur_in_length
            processed_seg.append(cur_processed_seg)
        
        # print("in_awe element shape: ", [t.shape for t in in_awe])

        
        in_length = torch.LongTensor(in_length).to(source.device)
        in_awe = torch.nn.utils.rnn.pad_sequence(in_awe, batch_first=True) # TODO pad to self.args.seg_cap might drastically increase memory consumption
        padding_mask = torch.arange(in_awe.shape[1]).unsqueeze(0).to(source.device) >= in_length.unsqueeze(1)

        out_awe = self.awe(in_awe, padding_mask, skip_cnn=True if (len(source.shape) == 3 and source.shape[2] == 512) else False)
        awe = []
        start = 0
        out_length = (~out_awe["padding_mask"]).sum(-1) # shouldn't use in_length because input might be raw waveform
        # print("embedder out_awe['feature'] shape: ", out_awe['features'].shape)
        # print("processed_seg:", processed_seg)
        for cur_seg in processed_seg:
            end = start + len(cur_seg)
            # print(start, end)
            cur_feat = out_awe['features'][start:end] # t_words, t_frame, d
            cur_len = out_length[start:end] # t_words
            temp = torch.stack([feat[:l].mean(0) for feat, l in zip(cur_feat, cur_len)], dim=0) # [t_words, d]
            assert temp.shape[0] == cur_len.shape[0] and temp.shape[1] == cur_feat.shape[2], f"temp shape: {temp.shape}"
            assert len(temp.shape) == 2 and temp.shape[-1] == 768, temp.shape
            awe.append(temp)
            start = end

        awe = torch.nn.utils.rnn.pad_sequence(awe, batch_first=True) # [B, t_words, d]
        in_combiner_length = torch.LongTensor([len(cur_seg) for cur_seg in processed_seg]).to(in_awe.device) # need to use filtered seg
        padding_mask = torch.arange(awe.shape[1]).unsqueeze(0).to(source.device) >= in_combiner_length.unsqueeze(1)

        x = self.proj(awe)
        x = self.pos_embed(x, padding_mask) + x
        x = self.layer_norm(x)

        if self.args.use_audio_cls_token:
            x = torch.cat([self.cls_token.repeat(x.shape[0],1,1), x], dim=1)
            cls_token_padding_mask = torch.zeros((padding_mask.shape[0],1)).to(padding_mask)
            padding_mask = torch.cat([cls_token_padding_mask, padding_mask], dim=1)
        
        combiner_feat, _ = self.combiner(x, padding_mask=padding_mask)
        if self.args.use_audio_cls_token:
            cls_token = combiner_feat[:,0]
            features = combiner_feat[:,1:]
            if padding_mask != None:
                padding_mask = padding_mask[:,1:]
        else:
            cls_token = None
            features = combiner_feat

        meanpool_feature = torch.stack([f[:ilen].mean(0) for f, ilen in zip(features, in_combiner_length)], dim=0)
        # print(f"from embedder line 129: cls_token.shape: {cls_token.shape if cls_token != None else None}")
        return {"cls_token": cls_token, "meanpool_feature": meanpool_feature}
        



                



class AWE(BaseFairseqModel):

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


        final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim

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

        if "WavLM" in self.args.load_awe_weights_from:
            # wavlm specific parameters
            args.relative_position_embedding = True
            args.num_buckets = 320 
            args.max_distance = 800 
            args.gru_rel_pos = True
            args.expand_attention_head_size=-1
            self.encoder = TransformerEncoder_wavlm(args)
        else:
            self.encoder = TransformerEncoder(args)
        self.layer_norm = LayerNorm(self.embed)



    def forward(self, source, padding_mask=None, skip_cnn=False):
        source = source.to(self.layer_norm.weight.device)
        if padding_mask != None:
            padding_mask = padding_mask.to(self.layer_norm.weight.device)

        if skip_cnn == False:
            if self.feature_grad_mult > 0:
                features = self.feature_extractor(source)
                if self.feature_grad_mult != 1.0:
                    features = GradMultiply.apply(features, self.feature_grad_mult)
            else:
                with torch.no_grad():
                    features = self.feature_extractor(source)
            features = features.transpose(1, 2)
        else:
            assert source.shape[2] == 512
            features = source

        
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
        x += x_conv
        # if self.args.use_audio_cls_token:
        #     x = torch.cat([self.cls_token.repeat(x.shape[0],1,1), x], dim=1)
        #     if padding_mask != None:
        #         cls_token_padding_mask = torch.zeros((padding_mask.shape[0],1)).to(padding_mask)
        #         padding_mask = torch.cat([cls_token_padding_mask, padding_mask], dim=1)

        # if need_attention_weights:
        #     features, attn_weights = self.encoder(x, padding_mask=padding_mask, tgt_layer=tgt_layer, need_attention_weights=True, pre_feats=pre_feats)
        #     return {"features": features, "attn_weights": attn_weights, "padding_mask": padding_mask}

        features, layer_feats = self.encoder(x, padding_mask=padding_mask)
        # if self.args.use_audio_cls_token:
        #     cls_token = layer_feats[:,0]
        #     features = features[:,1:]
        #     layer_feats = layer_feats[:,1:]
        #     if padding_mask != None:
        #         padding_mask = padding_mask[:,1:]
        # else:
        #     cls_token = None

        result = {"features": features, "padding_mask": padding_mask}

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
        print("load pretrained weights for AWE model")
        # random_init_last_x = getattr(self.args, "random_init_last_x", None)
        # freeze_first_x = getattr(self.args, "freeze_first_x", None)
        # if random_init_last_x != None and not load_all:
        #     cut = self.args.encoder_layers - random_init_last_x
        #     assert cut >= 0
        #     random_init_names = [f'encoder.layers.{i}.' for i in range(cut, self.args.encoder_layers)]
        #     print(f"randomly reinitialize the weights start with the following: {random_init_names}\n")
        new_states = self.state_dict()
        loaded_keys = []
        for k, v in states.items():
            k = k[7:] if k.startswith('module') else k
            k = k[22:] if k.startswith('w2v_encoder.w2v_model') else k
            k = k[11:] if k.startswith('w2v2_model') else k
            k = k[8:] if k.startswith('encoder.pos_conv') else k
            k = k[22:] if k.startswith('conv1_trm1_conv2_trm2') else k
            k = k.replace("audio_encoder.", "")
            # if random_init_last_x != None and not load_all:
            #     for names in random_init_names:
            #         if k.startswith(names):
            #             v = torch.tensor([0.0]).to(v.device) # make it so that the size doesn't match
            #             break
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
        # if freeze_first_x != None:
        #     freeze_names =  [f'encoder.layers.{i}.' for i in range(freeze_first_x)]
        #     for n, p in self.named_parameters():
        #         for fn in freeze_names:
        #             if n.startswith(fn):
        #                 p.requires_grad = False
        #                 print(f"disable gradient of weights: {n}")
        #                 break

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



class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.layer_use = args.layer_use
        # assert args.layer_use < args.encoder_layers, f"w2v2 only has {args.encoder_layers} layers, but you want layer feat from layer {args.layer_use+1}"
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
        stop_pass = False

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            # print(i)
            # dropout_probability = np.random.random()
            # if not self.training or (dropout_probability > self.layerdrop):
            if True:
                x, z = layer(x, self_attn_padding_mask=padding_mask,need_head_weights=need_head_weights)
                layer_results.append(x.transpose(0,1))
                # if i == layer_use:
                #     layer_feats = x.transpose(0, 1)
                #     if need_head_weights:
                #         if len(z.shape) == 3:
                #             z = z.unsqueeze(0)
                #         attn_weights = z # [bsz, num_heads, tgt_len, src_len]
                #         if pre_feats:
                #             return layer_results[-2], attn_weights
                #         else:
                #             return layer_feats, attn_weights
                #     # if need_head_weights:
                #         # # print(z.shape)
                #         # if len(z.shape) == 3:
                #         #     z = z.unsqueeze(0)
                #         # attn_weights = z # [bsz, num_heads, tgt_len, src_len]
                #         # # cls_attn_weights = attn_weights[:,:,0,1:] # [bsz,n_heads,src_len-1]
                #         # # return cls_attn_weights
                #         # return attn_weights
                #     # else
                #     if stop_pass:
                #         return layer_feats, layer_feats
        if all_hidden_states:
            return layer_results
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, x

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


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 480000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, padding_mask):
        """
        Args:
            x: Tensor, shape [bsz, seq_len, embedding_dim]
        """
        if padding_mask == None:
            return self.pe[:, :x.shape[1]]
        pe = self.pe[:, :x.shape[1]].repeat((padding_mask.shape[0], 1, 1))
        pe[padding_mask] = 0.
        return pe