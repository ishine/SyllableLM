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
from fairseq.data.audio.cluster_alan_dataset import ClusterAlanDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)

@dataclass
class ClusterAlanPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
                    "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
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

    reducer_path: Optional[str] = None
    tsv_replace_source: Optional[str] = None
    tsv_replace_target: Optional[str] = None

    bpe_path: Optional[str] = None

    vocab_size: Optional[int] = 1024


@register_task("cluster_alan_pretraining", dataclass=ClusterAlanPretrainingConfig)
class ClusterAlanPretrainingTask(FairseqTask):
    def __init__(
            self,
            cfg: ClusterAlanPretrainingConfig,
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
            cls, cfg: ClusterAlanPretrainingConfig, **kwargs
    ) -> "ClusterAlanPretrainingTask":
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"

        self.datasets[split] = ClusterAlanDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            vocab_size=self.cfg.vocab_size,
            reducer_path=self.cfg.reducer_path,
            tsv_replace_source=self.cfg.tsv_replace_source,
            tsv_replace_target=self.cfg.tsv_replace_target,
            bpe_path=self.cfg.bpe_path,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
