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


def load_audio(manifest_path):
    inds = []
    token_paths_short, token_lengths_short, token_paths_long, token_lengths_long = [], [], [], []
    with open(manifest_path) as f:
        f.readline()
        ind = 0
        for line in f:
            inds.append(ind)
            items = line.strip().split()
            token_paths_short.append(items[0])
            token_lengths_short.append(int(items[1]))
            token_paths_long.append(items[2])
            token_lengths_long.append(int(items[3]))
            ind += 1

    logger.info(
        (
            f"loaded {len(inds)}"
        )
    )
    return inds, token_paths_short, token_lengths_short, token_paths_long, token_lengths_long


class ClusterAlanDatasetInterleaved(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            shuffle: bool = True,
            vocab_size_short: int = 1024,
            vocab_size_long: int = 4096,
            simple: bool = True,
            interleave_strat: int = 1,
            long_dropout: float = 0.0,
            loss_threshold: float = 0.35,
            reducer_path: Optional[str] = None,
            tsv_replace_source: Optional[str] = None,
            tsv_replace_target: Optional[str] = None,
            interleave_long_grouping: Optional[int] = 1,
    ):
        if manifest_path is not None:
            self.inds, self.token_paths_short, self.token_lengths_short, self.token_paths_long, self.token_lengths_long = load_audio(manifest_path)

        self.shuffle = shuffle
        self.time_bias = 0

        self.vocab_size_short = vocab_size_short
        self.vocab_size_long = vocab_size_long

        self.pad_idx = vocab_size_short + vocab_size_long + 2

        self.simple = simple
        self.interleave_strat = interleave_strat
        self.long_dropout = long_dropout
        self.loss_threshold = loss_threshold
        if long_dropout != 0.0:
            assert interleave_strat != 1

        self.reducer = None
        if reducer_path is not None:
            self.reducer = torch.from_numpy(np.load(reducer_path))
        self.tsv_replace_source = tsv_replace_source
        self.tsv_replace_target = tsv_replace_target
        self.interleave_long_grouping = interleave_long_grouping

    def get_audio(self, token_path_short, token_path_long, loss_path_short=None):
        input_losses_short = None
        if token_path_short.endswith('.pt'):
            input_tokens_short = torch.load(token_path_short)
        else:
            input_tokens_short = torch.from_numpy(np.load(token_path_short))

        if token_path_long.endswith('.pt'):
            input_tokens_long = torch.load(token_path_long).T
        else:
            if self.interleave_strat >= 4:
                input_tokens_long = torch.from_numpy(np.load(token_path_long.replace(self.tsv_replace_source, self.tsv_replace_target)))
            else:
                input_tokens_long = torch.from_numpy(np.load(token_path_long)).T

        if self.reducer is not None:
            input_tokens_long = torch.cat([self.reducer[input_tokens_long[:1]], input_tokens_long[1:]], dim=0)

        return input_tokens_short, input_tokens_long, input_losses_short

    def interleave1(self, input_tokens_short, input_tokens_long, rate_short=2, rate_long=1):
        # io spec exists, short has durations at end. Long has durations later.
        # in_tokens_short = 25hz. 0 is tokens, 1 is durations
        # in_tokens_long = 50hz. 0 is tokens, 1 is starts, 2 is ends.

        # mhubert mhubert_stop sdhubert sdhubert_stop mhubert
        # 0 1 1 0 0 = routing function

        # best idea is to insert after last token included in previous range.

        long_offset = self.vocab_size_short + 1
        stop_token_short = self.vocab_size_short
        stop_token_long = self.vocab_size_long + long_offset

        idx_0 = 0  # index of current mhubert unit
        time_0 = rate_short * input_tokens_short[1, 0]  # start time of current mhubert unit
        idx_1 = 0  # index of current sdhubert unit
        time_1 = 0  # long time, end of last sdhubert unit

        tokens_interleaved = []
        predicts_long = []
        predicts_short = []
        tokens_long = []
        tokens_short = []

        while idx_0 < input_tokens_short.size(-1) or idx_1 < input_tokens_long.size(-1):
            if (idx_0 >= input_tokens_short.size(-1)) or (time_1 - self.time_bias < time_0 and idx_1 < input_tokens_long.size(-1)):
                cur_long_token = input_tokens_long[0, idx_1] + long_offset

                if len(tokens_interleaved) > 0:
                    predicts_short.append(len(tokens_interleaved) - 1)
                    tokens_short.append(stop_token_short)
                predicts_long.append(len(tokens_interleaved))
                tokens_long.append(cur_long_token)

                tokens_interleaved.extend([stop_token_short, cur_long_token, stop_token_long])
                if idx_1 != input_tokens_long.size(-1) - 1:
                    time_1 = input_tokens_long[1, idx_1 + 1] - max(3, 1 + rate_long * (input_tokens_long[1, idx_1 + 1] - input_tokens_long[2, idx_1]))
                idx_1 += 1
            else:
                cur_short_token = input_tokens_short[0, idx_0]

                assert len(tokens_interleaved) != 0
                predicts_short.append(len(tokens_interleaved) - 1)
                tokens_short.append(cur_short_token)

                tokens_interleaved.append(cur_short_token)
                if idx_0 + 1 < input_tokens_short.size(-1):
                    time_0 += rate_short * input_tokens_short[1, idx_0 + 1]
                idx_0 += 1

        return (
            torch.tensor(tokens_interleaved),
            torch.tensor(tokens_short),
            torch.tensor(tokens_long),
            torch.tensor(predicts_short),
            torch.tensor(predicts_long),
            None,
        )

    def interleave2(self, input_tokens_short, input_tokens_long, input_losses_short, rate_short=2, rate_long=1):
        # this one is much more willing to predict mhubert before sdhubert. Therefore, it may be best with dropout / MOE
        # breaks ties on middle of short unit (bc of durations), where long SDH goes before short unit if SDH start <= MH middle
        # -3 At least
        # Prev_end-1 At least
        # -7 At most
        # Todo: walk forward if low loss is a connecting bridge
        # Todo: dropout
        # Todo

        # best idea is to insert after last token included in previous range.

        long_offset = self.vocab_size_short + 1
        stop_token_short = self.vocab_size_short
        stop_token_long = self.vocab_size_long + long_offset

        idx_0 = 0  # index of current mhubert unit
        time_0 = 0  # start time of current mhubert unit
        idx_1 = 0  # index of current sdhubert unit
        time_1 = 0  # long time, end of last sdhubert unit

        tokens_interleaved = []
        predicts_long = []
        predicts_short = []
        tokens_long = []
        tokens_short = []
        moe_mask = []  # is_long (stop tokens exceptions)

        def insert_short_token():
            nonlocal idx_0, time_0
            cur_short_token = input_tokens_short[0, idx_0]

            assert len(tokens_interleaved) != 0
            predicts_short.append(len(tokens_interleaved) - 1)
            tokens_short.append(cur_short_token)
            tokens_interleaved.append(cur_short_token)
            moe_mask.append(False)

            if idx_0 < input_tokens_short.size(-1):
                time_0 += rate_short * input_tokens_short[1, idx_0]
            idx_0 += 1

        def insert_long_token():
            nonlocal idx_1, time_1
            cur_long_token = input_tokens_long[0, idx_1] + long_offset

            if len(tokens_interleaved) > 0:
                predicts_short.append(len(tokens_interleaved) - 1)
                tokens_short.append(stop_token_short)
            predicts_long.append(len(tokens_interleaved))
            tokens_long.append(cur_long_token)

            if torch.rand((1,)) < self.long_dropout:
                moe_mask.extend([True, False])
                tokens_interleaved.extend([stop_token_short, stop_token_long])
            else:
                moe_mask.extend([True, True, False])
                tokens_interleaved.extend([stop_token_short, cur_long_token, stop_token_long])
            time_1 = input_tokens_long[2, idx_1]
            idx_1 += 1

        insert_long_token()
        while idx_0 < input_tokens_short.size(-1) or idx_1 < input_tokens_long.size(-1):
            if idx_0 >= input_tokens_short.size(-1):
                insert_long_token()
                continue
            if idx_1 >= input_tokens_long.size(-1):
                insert_short_token()
                continue

            end_last_long = time_1
            start_cur_long = input_tokens_long[1, idx_1]
            long_insert_time = max(
                0,
                start_cur_long - 7,
                min(end_last_long - 1, start_cur_long - 3)
            )
            short_insert_time = time_0 + rate_short * input_tokens_short[1, idx_0] / 2

            if short_insert_time >= start_cur_long:  # don't go too far
                insert_long_token()
            elif short_insert_time <= long_insert_time:
                insert_short_token()
            elif time_0 > 100 and time_1 > 100 and input_losses_short[idx_0] < -np.log(
                    self.loss_threshold):  # at least a 35% chance on this token to be predicted, >2sec in
                insert_short_token()
            else:
                insert_long_token()

        return (
            torch.tensor(tokens_interleaved),
            torch.tensor(tokens_short),
            torch.tensor(tokens_long),
            torch.tensor(predicts_short),
            torch.tensor(predicts_long),
            torch.tensor(moe_mask),
        )

    def interleave3(self, input_tokens_short, input_tokens_long, rate_short=2, rate_long=1):
        # this one is much more willing to predict mhubert before sdhubert. Therefore, it may be best with dropout / MOE
        # breaks ties on middle of short unit (bc of durations), where long SDH goes before short unit if SDH start <= MH middle
        # -3 At least
        # Prev_end-1 At least
        # -7 At most
        # Todo: walk forward if low loss is a connecting bridge
        # Todo: dropout
        # Todo

        # best idea is to insert after last token included in previous range.

        long_offset = self.vocab_size_short + 1
        stop_token_short = self.vocab_size_short
        stop_token_long = self.vocab_size_long + long_offset

        idx_0 = 0  # index of current mhubert unit
        time_0 = 0  # start time of current mhubert unit
        idx_1 = 0  # index of current sdhubert unit
        time_1 = 0  # long time, end of last sdhubert unit

        tokens_interleaved = []
        predicts_long = []
        predicts_short = []
        tokens_long = []
        tokens_short = []
        moe_mask = []  # is_long (stop tokens exceptions)

        def insert_short_token():
            nonlocal idx_0, time_0
            cur_short_token = input_tokens_short[0, idx_0]

            assert len(tokens_interleaved) != 0
            predicts_short.append(len(tokens_interleaved) - 1)
            tokens_short.append(cur_short_token)
            tokens_interleaved.append(cur_short_token)
            moe_mask.append(False)

            if idx_0 < input_tokens_short.size(-1):
                time_0 += rate_short * input_tokens_short[1, idx_0]
            idx_0 += 1

        def insert_long_token(is_span_start=True, is_span_end=True):
            nonlocal idx_1, time_1
            cur_long_token = input_tokens_long[0, idx_1] + long_offset

            if len(tokens_interleaved) > 0:
                predicts_short.append(len(tokens_interleaved) - 1)
                tokens_short.append(stop_token_short)
            predicts_long.append(len(tokens_interleaved))
            tokens_long.append(cur_long_token)

            if torch.rand((1,)) < self.long_dropout:
                raise NotImplementedError()
                moe_mask.extend([True, False])
                tokens_interleaved.extend([stop_token_short, stop_token_long])
            else:
                if is_span_start:
                    moe_mask.append(True)
                    tokens_interleaved.append(stop_token_short)
                moe_mask.append(True)
                tokens_interleaved.append(cur_long_token)
                if is_span_end:
                    moe_mask.append(False)
                    tokens_interleaved.append(stop_token_long)
            time_1 = input_tokens_long[2, idx_1]
            idx_1 += 1

        insert_long_token()
        while idx_0 < input_tokens_short.size(-1) or idx_1 < input_tokens_long.size(-1):
            if idx_0 >= input_tokens_short.size(-1):
                insert_long_token()
                continue
            if idx_1 >= input_tokens_long.size(-1):
                insert_short_token()
                continue

            end_last_long = time_1
            start_cur_long = input_tokens_long[1, idx_1]
            long_insert_time = max(
                0,
                start_cur_long - 3,
                min(end_last_long - 1, start_cur_long - 3)
            )
            short_insert_time = time_0 + rate_short * input_tokens_short[1, idx_0] / 2

            if short_insert_time <= long_insert_time:
                insert_short_token()
            else:
                insert_long_token()

        return (
            torch.tensor(tokens_interleaved),
            torch.tensor(tokens_short),
            torch.tensor(tokens_long),
            torch.tensor(predicts_short),
            torch.tensor(predicts_long),
            torch.tensor(moe_mask),
        )

    def interleave4(self, input_tokens_short, input_tokens_long, rate_short=2, rate_long=1):
        # this one is much more willing to predict mhubert before sdhubert. Therefore, it may be best with dropout / MOE
        # breaks ties on middle of short unit (bc of durations), where long SDH goes before short unit if SDH start <= MH middle
        # -3 At least
        # Prev_end-1 At least
        # -7 At most
        # Todo: walk forward if low loss is a connecting bridge
        # Todo: dropout
        # Todo

        # best idea is to insert after last token included in previous range.

        long_offset = self.vocab_size_short + 1
        stop_token_short = self.vocab_size_short
        stop_token_long = self.vocab_size_long + long_offset

        idx_0 = 0  # index of current mhubert unit
        time_0 = 0  # start time of current mhubert unit
        idx_1 = 0  # index of current sdhubert unit
        time_1 = 0  # long time, end of last sdhubert unit

        tokens_interleaved = []
        predicts_long = []
        predicts_short = []
        tokens_long = []
        tokens_short = []
        moe_mask = []  # is_long (stop tokens exceptions)

        def insert_short_token():
            nonlocal idx_0, time_0
            cur_short_token = input_tokens_short[0, idx_0]

            assert len(tokens_interleaved) != 0
            predicts_short.append(len(tokens_interleaved) - 1)
            tokens_short.append(cur_short_token)
            tokens_interleaved.append(cur_short_token)
            moe_mask.append(False)

            if idx_0 < input_tokens_short.size(-1):
                time_0 += rate_short * input_tokens_short[1, idx_0]
            idx_0 += 1

        def insert_long_token(is_span_start=True, is_span_end=True):
            nonlocal idx_1, time_1
            cur_long_token = input_tokens_long[0, idx_1] + long_offset

            if is_span_start:
                if len(tokens_interleaved) > 0:
                    predicts_short.append(len(tokens_interleaved) - 1)
                    tokens_short.append(stop_token_short)
                moe_mask.append(True)
                tokens_interleaved.append(stop_token_short)

            predicts_long.append(len(tokens_interleaved) - 1)
            tokens_long.append(cur_long_token)
            moe_mask.append(True)
            tokens_interleaved.append(cur_long_token)

            if is_span_end:
                moe_mask.append(False)
                tokens_interleaved.append(stop_token_long)

            time_1 = input_tokens_long[2, idx_1]
            idx_1 += 1

        def insert_long_tokens():
            num_to_insert = min(self.interleave_long_grouping, input_tokens_long.size(-1) - idx_1)
            for i in range(num_to_insert):
                is_span_start = i == 0
                is_span_end = i == num_to_insert - 1
                insert_long_token(is_span_start, is_span_end)

        insert_long_tokens()
        while idx_0 < input_tokens_short.size(-1) or idx_1 < input_tokens_long.size(-1):
            if idx_0 >= input_tokens_short.size(-1):
                insert_long_tokens()  # should not get called todo just remove?
                continue
            if idx_1 >= input_tokens_long.size(-1):
                insert_short_token()
                continue

            end_last_long = time_1
            start_cur_long = input_tokens_long[1, idx_1]
            long_insert_time = max(
                0,
                start_cur_long - 5,
            )
            short_insert_time = time_0 + rate_short * input_tokens_short[1, idx_0] / 2

            if short_insert_time <= long_insert_time:
                insert_short_token()
            else:
                insert_long_tokens()

        return (
            torch.tensor(tokens_interleaved),
            torch.tensor(tokens_short),
            torch.tensor(tokens_long),
            torch.tensor(predicts_short),
            torch.tensor(predicts_long),
            torch.tensor(moe_mask),
        )

    def interleave5(self, input_tokens_short, input_tokens_long, rate_short=2, rate_long=1):
        # ELLA-V

        # best idea is to insert after last token included in previous range.

        long_offset = self.vocab_size_short + 1
        stop_token_short = self.vocab_size_short
        stop_token_long = self.vocab_size_long + long_offset

        idx_0 = 0  # index of current mhubert unit
        time_0 = 0  # start time of current mhubert unit
        idx_1 = 0  # index of current sdhubert unit
        time_1 = 0  # long time, end of last sdhubert unit

        tokens_interleaved = []
        predicts_long = []
        predicts_short = []
        tokens_long = []
        tokens_short = []
        moe_mask = []  # is_long (stop tokens exceptions)

        def insert_short_token():
            nonlocal idx_0, time_0
            cur_short_token = input_tokens_short[0, idx_0]

            assert len(tokens_interleaved) != 0
            predicts_short.append(len(tokens_interleaved) - 1)
            tokens_short.append(cur_short_token)
            tokens_interleaved.append(cur_short_token)
            moe_mask.append(False)

            if idx_0 < input_tokens_short.size(-1):
                time_0 += rate_short * input_tokens_short[1, idx_0]
            idx_0 += 1

        def insert_long_token(is_span_start=True, is_span_end=True):
            nonlocal idx_1, time_1
            cur_long_token = input_tokens_long[0, idx_1] + long_offset

            if is_span_start:
                if len(tokens_interleaved) > 0:
                    predicts_short.append(len(tokens_interleaved) - 1)
                    tokens_short.append(stop_token_short)
                # moe_mask.append(True)
                # tokens_interleaved.append(stop_token_short)

            # predicts_long.append(len(tokens_interleaved) - 1)
            # tokens_long.append(cur_long_token)
            moe_mask.append(True)
            tokens_interleaved.append(cur_long_token)

            if is_span_end:
                moe_mask.append(False)
                tokens_interleaved.append(stop_token_long)

            time_1 = input_tokens_long[2, idx_1]
            idx_1 += 1

        def insert_long_tokens():
            num_to_insert = min(self.interleave_long_grouping, input_tokens_long.size(-1) - idx_1)
            for i in range(num_to_insert):
                is_span_start = i == 0
                is_span_end = i == num_to_insert - 1
                insert_long_token(is_span_start, is_span_end)

        insert_long_tokens()
        while idx_0 < input_tokens_short.size(-1) or idx_1 < input_tokens_long.size(-1):
            if idx_0 >= input_tokens_short.size(-1):
                insert_long_tokens()  # should not get called todo just remove?
                continue
            if idx_1 >= input_tokens_long.size(-1):
                insert_short_token()
                continue

            end_last_long = time_1
            start_cur_long = input_tokens_long[1, idx_1]
            long_insert_time = max(
                0,
                start_cur_long - 5,
            )
            short_insert_time = time_0 + rate_short * input_tokens_short[1, idx_0] / 2

            if short_insert_time <= long_insert_time:
                insert_short_token()
            else:
                insert_long_tokens()

        return (
            torch.tensor(tokens_interleaved),
            torch.tensor(tokens_short),
            torch.tensor(tokens_long),
            torch.tensor(predicts_short),
            torch.tensor(predicts_long),
            torch.tensor(moe_mask),
        )

    def interleave(self, input_tokens_short, input_tokens_long, input_losses_short=None, rate_short=2, rate_long=1):
        if self.interleave_strat == 1:
            return self.interleave1(input_tokens_short, input_tokens_long, rate_short, rate_long)
        elif self.interleave_strat == 2:
            return self.interleave2(input_tokens_short, input_tokens_long, input_losses_short, rate_short, rate_long)
        elif self.interleave_strat == 3:
            return self.interleave3(input_tokens_short, input_tokens_long, rate_short, rate_long)
        elif self.interleave_strat == 4:
            return self.interleave4(input_tokens_short, input_tokens_long, rate_short, rate_long)
        elif self.interleave_strat == 5:
            return self.interleave5(input_tokens_short, input_tokens_long, rate_short, rate_long)
        else:
            assert False, "Interleave strat"

    def __getitem__(self, index):
        # I don't remember what receives this? Collater? Yup. Batch with source and net input, check!
        # We use model criterion, so the output should support getting called w/ model(**sample["net_input"])

        # giga todo for mixed cluster
        input_tokens_short, input_tokens_long, input_losses_short = self.get_audio(self.token_paths_short[index],
                                                                                   self.token_paths_long[index])  # giga todo include durations?
        tokens_interleaved, tokens_short, tokens_long, predicts_short, predicts_long, is_long = self.interleave(input_tokens_short, input_tokens_long,
                                                                                                                input_losses_short)

        return {
            "id": index,
            "tokens_interleaved": tokens_interleaved,
            "tokens_short": tokens_short,
            "tokens_long": tokens_long,
            "predicts_short": predicts_short,
            "predicts_long": predicts_long,
            "is_long": is_long,
        }

    def __len__(self):
        return len(self.inds)

    def collater(self, samples):
        ignore_index = -100

        if len(samples) == 0:
            print('WHAT THE HELL BATCH LOADER')
            samples = [self.__getitem__(i) for i in range(64)]

        collated_interleaved = torch.nn.utils.rnn.pad_sequence([s['tokens_interleaved'] for s in samples], batch_first=True, padding_value=self.pad_idx)
        padding_mask_interleaved = (collated_interleaved == self.pad_idx)
        collated_short = torch.nn.utils.rnn.pad_sequence([s['tokens_short'] for s in samples], batch_first=True, padding_value=ignore_index)
        collated_long = torch.nn.utils.rnn.pad_sequence([s['tokens_long'] for s in samples], batch_first=True, padding_value=ignore_index)
        collated_pred_short = torch.nn.utils.rnn.pad_sequence([s['predicts_short'] for s in samples], batch_first=True, padding_value=0)
        collated_pred_long = torch.nn.utils.rnn.pad_sequence([s['predicts_long'] for s in samples], batch_first=True, padding_value=0)
        collated_is_long = None
        if 'is_long' in samples[0]:
            collated_is_long = torch.nn.utils.rnn.pad_sequence([s['is_long'] for s in samples], batch_first=True, padding_value=1).bool()

        net_input = {
            "interleaved_tokens": collated_interleaved,
            "interleaved_padding_mask": padding_mask_interleaved,
            "short_tokens": collated_short,
            "long_tokens": collated_long,
            "predicts_short_indices": collated_pred_short,
            "predicts_long_indices": collated_pred_long,
            "is_long": collated_is_long,
        }

        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):  # todo fix
        return self.token_lengths_short[index] + self.token_lengths_long[index] * 3

    def ordered_indices(self):
        if self.shuffle:  # Shuffle only breaks length ties
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(np.array(self.token_lengths_short) + np.array(self.token_lengths_long) * 3)
        return np.lexsort(order)[::-1]  # Last key least important
