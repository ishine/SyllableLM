# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.data.audio.cluster_alan_dataset_interleaved import ClusterAlanDatasetInterleaved
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class ClusterAlanInterleavedPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
                    "sampled to this rate"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    min_keep_phone_size: Optional[int] = field(
        default=50,
        metadata={"help": "min sample size to crop phones to for batching"},
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )

    feature_type: Optional[str] = field(
        default='wav',
        metadata={"help": "choose from ['wav', 'spectrogram', 'fbank', 'mfcc', 'phase_spectrogram']"}
    )

    encodec_rate: Optional[int] = field(
        default=75,
        metadata={"help": "75 for encodec, 50 for hubert"},
    )

    full_prompt_type: Optional[str] = field(
        default='norm',
        metadata={"help": "norm, phn_ctc, mhubert"},
    )
    pred_prompt_type: Optional[str] = field(
        default='encodec',
        metadata={"help": "encodec, mhubert"},
    )

    vocab_size_short: Optional[int] = 1024
    vocab_size_long: Optional[int] = 4096

    interleave_strat: int = 1
    long_dropout: float = 0.0

    reducer_path: Optional[str] = None
    tsv_replace_source: Optional[str] = None
    tsv_replace_target: Optional[str] = None
    interleave_long_grouping: Optional[int] = 1


@register_task("cluster_alan_interleaved_pretraining", dataclass=ClusterAlanInterleavedPretrainingConfig)
class ClusterAlanInterleavedPretrainingTask(FairseqTask):
    def __init__(
            self,
            cfg: ClusterAlanInterleavedPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"ClusterAlanPretrainingConfig Config {cfg}")

        self.cfg = cfg

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def dictionaries(self) -> List[Dictionary]:
        return None

    @classmethod
    def setup_task(
            cls, cfg: ClusterAlanInterleavedPretrainingConfig, **kwargs
    ) -> "ClusterAlanInterleavedPretrainingTask":
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"

        self.datasets[split] = ClusterAlanDatasetInterleaved(
            manifest,
            vocab_size_short=self.cfg.vocab_size_short,
            vocab_size_long=self.cfg.vocab_size_long,
            simple=True,
            interleave_strat=self.cfg.interleave_strat,
            long_dropout=self.cfg.long_dropout if split == 'train' else 0.0,
            reducer_path=self.cfg.reducer_path,
            tsv_replace_source=self.cfg.tsv_replace_source,
            tsv_replace_target=self.cfg.tsv_replace_target,
            interleave_long_grouping=self.cfg.interleave_long_grouping,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
