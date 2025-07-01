# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union

import numpy as np
from collections import defaultdict
import random

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset

import torchaudio

logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep):
    inds = []
    token_paths, token_lengths = [], []
    with open(manifest_path) as f:
        f.readline()
        ind = 0
        for line in f:
            inds.append(ind)
            items = line.strip().split()
            token_paths.append(items[0])
            token_lengths.append(int(items[1]))
            ind += 1

    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(inds)}"
        )
    )
    return inds, token_paths, token_lengths


class ClusterAlanDataset(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            shuffle: bool = True,
            vocab_size: int = 1024,
            reducer_path: Optional[str] = None,
            tsv_replace_source: Optional[str] = None,
            tsv_replace_target: Optional[str] = None,
            bpe_path: Optional[str] = None,
    ):
        self.inds, self.token_paths, self.token_lengths = load_audio(manifest_path, max_keep_sample_size, min_keep_sample_size)

        self.sample_rate = sample_rate
        self.shuffle = shuffle

        self.vocab_size = vocab_size

        self.reducer = None
        if reducer_path is not None:
            self.reducer = torch.from_numpy(np.load(reducer_path))
        self.tsv_replace_source = tsv_replace_source
        self.tsv_replace_target = tsv_replace_target

        self.bpe_tokenizer = None
        if bpe_path is not None:
            from tokenizers import Tokenizer
            self.bpe_tokenizer = Tokenizer.from_file(bpe_path)

    def get_audio(self, index):
        token_path = self.token_paths[index]
        if self.tsv_replace_source is not None and self.tsv_replace_target is not None:  # todo rename to split?
            token_path = token_path.replace(self.tsv_replace_source, self.tsv_replace_target)

        if token_path.endswith('.pt'):
            input_tokens = torch.load(token_path)
        else:
            # input_tokens = torch.from_numpy(np.load(self.token_paths[index])).view(1, -1)
            input_tokens = torch.from_numpy(np.load(token_path))[0]

        if self.reducer is not None:
            input_tokens = self.reducer[input_tokens]
            # TODO: RE-RLE(?)
        if self.bpe_tokenizer is not None:
            input_tokens = torch.tensor(self.bpe_tokenizer.encode(''.join([chr(a+5000) for a in input_tokens])).ids)

        return input_tokens

    def __getitem__(self, index):
        # I don't remember what receives this? Collater? Yup. Batch with source and net input, check!
        # We use model criterion, so the output should support getting called w/ model(**sample["net_input"])

        # giga todo for mixed cluster
        input_tokens = self.get_audio(index)
        # input_tokens = torch.clamp(self.get_audio(index), max=self.vocab_size-1)[0]  # syllable

        return {
            "id": index,
            'input_tokens': input_tokens,
        }

    def __len__(self):
        return len(self.inds)

    def collater(self, samples):
        def collate_pad(tensors, pad_to_max=True, pad_value=-100, stop_value=None, start_value=None, end_stop=False):  # input is b list of (t, c)
            max_tensor_len = max([t.size(0) for t in tensors])
            max_tensor_len += 1 if stop_value is not None else 0
            b, t = len(tensors), max_tensor_len

            source = tensors[0].new_full((b, t), pad_value)
            padding_mask = tensors[0].new_zeros((b, t))

            for idx, tensor in enumerate(tensors):
                source[idx, :tensor.size(0)] = tensor
                padding_mask[idx, tensor.size(0):] = True
                if stop_value is not None:
                    if end_stop:
                        source[idx, -1] = stop_value
                        padding_mask[idx, -1] = False
                    else:
                        source[idx, tensor.size(0)] = stop_value

            if start_value is not None:
                source = torch.cat([source.new_full((b, 1), start_value), source], dim=1)
                padding_mask = torch.cat([padding_mask.new_zeros((b, 1)), padding_mask], dim=1)

            return source, padding_mask

        if len(samples) == 0:
            print('WHAT THE HELL BATCH LOADER')
            samples = [self.__getitem__(i) for i in range(64)]

        collated_input, padding_mask = collate_pad(
            [s['input_tokens'] for s in samples],
            stop_value=self.vocab_size,
            start_value=self.vocab_size,
            end_stop=True,
        )

        net_input = {
            "input_tokens": collated_input,
            "padding_mask": padding_mask,
        }

        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):  # todo fix
        return self.token_lengths[index]

    def ordered_indices(self):
        if self.shuffle:  # Shuffle only breaks length ties
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(np.array(self.token_lengths))
        return np.lexsort(order)[::-1]  # Last key least important
