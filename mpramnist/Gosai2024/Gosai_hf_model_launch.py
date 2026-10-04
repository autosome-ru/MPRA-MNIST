"""
Fine-tune one Hugging Face DNA model on Gosai2024.

Flags
-----
--model          backbone; default moderngena-large
--filtration     own or original; default own
--device         CUDA index; default 0
--num_workers    DataLoader workers; default 0
--root           data directory; default <repo>/data
--result_dir     TSV path
--model_dir      checkpoint directory
--lr             learning-rate override
--wd             weight-decay override
--batch_size     batch size; default 256
--epoch_num      epochs; default 10
--max_length     tokenizer length override
--head_dropout   head dropout for Caduceus and DNABERT-2

The description continues below.

This file must stay at ``MPRA-MNIST/mpramnist/Gosai2024/`` and must be
started with that directory as the working directory. Python puts the
script directory on ``sys.path``, which is not the repository root, so
``find_mpramnist_repo`` adds the root (two levels up) before any
``mpramnist`` import. Do not set ``PYTHONPATH`` for that import. Do not
move the file: the root is derived from ``__file__``, not from the shell's
current directory.

Environment
-----------
Each backbone has incompatible dependencies. Create a fresh environment,
clone that model's repository, and install its dependency file. Then
install MPRA-MNIST's own ``requirements.txt`` into the same environment.
Do not share one environment across backbones.

moderngena-base and moderngena-large
    git clone https://github.com/AIRI-Institute/GENA_LM.git
    cd GENA_LM
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

ntv3
    git clone https://github.com/instadeepai/nucleotide-transformer.git
    cd nucleotide-transformer
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    If the clone has no requirements.txt, the upstream install is ``pip install .``.
    The checkpoint is gated. Export ``HF_TOKEN`` (or
    ``HUGGING_FACE_HUB_TOKEN``) in the same shell before launching.

caduceus
    git clone https://github.com/kuleshov-group/caduceus.git
    cd caduceus
    The repository has no requirements.txt. Create the published conda env:
    conda env create -f caduceus_env.yml
    conda activate caduceus_env

dnabert2
    The DNABERT-2 repository pins Python 3.8, torch 1.13, and
    transformers 4.29 for its own finetune/train.py. This script does not
    import that repository and does not start a 3.8 interpreter. It loads
    zhihan1996/DNABERT-2-117M inside the MPRA-MNIST process
    (Python 3.10 or newer) and trains the three-output head defined below.
    ``--head_dropout`` (default 0.1) is dropout on that head only. The
    backbone's own dropout stays whatever the checkpoint config sets.
    Do not install the Python 3.8 DNABERT requirements and the MPRA-MNIST
    requirements into the same environment.

    The checkpoint's remote code was written for transformers 4.28.
    ``AutoModel.from_pretrained`` fails on recent transformers (verified on
    5.x) with "Tensor on device meta is not on the expected device cpu":
    the model is built on the meta device, and the remote ALiBi code makes
    its slope tensor with ``torch.Tensor(list)``, which stays on CPU. This
    script therefore does not call ``from_pretrained`` for DNABERT-2. See
    ``load_dnabert2_backbone`` for the load path and the two other
    compatibility fixes it applies. Tested with transformers 4.57.6 and
    5.18.0.

In every environment above, also run:
    pip install -r /path/to/MPRA-MNIST/requirements.txt

One launch, three cell lines
----------------------------
Gosai2024 stores three log2FC activities per sequence. Every launch
trains HepG2, K562, and SKNSH together, in that order. ``--root`` is
the data directory (default ``<repo>/data``). The dataset class locates
its table under that directory.
The sequence flanks are ``GosaiDataset.LEFT_FLANK`` and
``GosaiDataset.RIGHT_FLANK`` in ``dataset.py``.

Labels are the ``*_log2FC`` columns. Output 0 is HepG2, output 1 is
K562, output 2 is SKNSH.

Regression head
---------------
The pool reduces the token states to one vector. The regression head is
dropout (Caduceus and DNABERT-2 only) followed by
``nn.Linear(hidden_size, 3)``. One shared vector produces all three
outputs. Logits stay shaped ``[batch, 3]`` and are not flattened. Huber
loss (delta 1.0) is the mean over the batch and over the three outputs.
Pearson is computed on each column separately, so a HepG2 prediction is
never correlated with a K562 label.

Chromosome split, 10 epochs
----------------------------
There is no k-fold. One launch trains for ``--epoch_num`` epochs
(default 10) on the fixed chromosome split:

    train    chromosomes 1-6, 8-12, 14-18, 20, 22, Y
    val      chromosomes 19, 21, X
    test     chromosomes 7, 13

    for epoch in 1..10:
        train on the train chromosomes
        validate on the val chromosomes
        append one TSV row with record=epoch
        if avg_val_pearson is the best so far, write best.pt
    after epoch 10:
        load best.pt
        score the test chromosomes with those weights only
        append one TSV row with record=test

The TSV has the epoch rows and then the test row. There is no fold column.

Per-cell columns are ``train_pearson_HepG2``, ``train_pearson_K562``,
``train_pearson_SKNSH``, and the same pattern for val and test. The mean
of the three is ``avg_train_pearson``, ``avg_val_pearson``, or
``avg_test_pearson``. ``avg_train_pearson`` comes from the training
forwards (dropout is on for Caduceus). ``val_loss`` and
``avg_val_pearson`` are computed in eval mode. The checkpoint is chosen
by ``avg_val_pearson``. A NaN on any cell makes that average NaN and
does not replace the saved weights. The test row is not written into an
epoch row.

Filtration (``--filtration``)
-----------------------------
Default ``own``. Quality filters use:

    stderr_threshold = 1.0
    std_multiple_cut = 6.0
    up_cutoff_move   = 3.0
    duplication_cutoff = 0.5

``duplication_cutoff`` is passed for train, val, and test. Rows whose
maximum activity across the three lines is above 0.5 are copied once.
``use_original_reverse_complement`` stays false. Reverse-complement is
only the training string transform below.

``own`` does not pad sequences. Training adds
``GosaiDataset.LEFT_FLANK`` and ``GosaiDataset.RIGHT_FLANK``,
center-crops to 600 bp, then reverse-complements with probability 0.5
(0 for Caduceus, whose pool already averages the reverse complement).
Validation and test add the same two flanks and center-crop to 600 bp.

``--filtration original`` uses the dataset's own padding to 600 bp with
those same two flanks. This launcher then does not add flanks and does
not crop. Training still reverse-complements with the probability
above. Validation and test receive the padded string unchanged.

The default result and checkpoint paths include the filtration name, so
an ``own`` run and an ``original`` run do not write the same files.

Outputs, relative to the working directory
-------------------------------------------
    ./gosai_{model}_{filtration}.tsv
        Metrics. Pass ``--result_dir`` to change the path. Reusing a TSV
        whose columns differ from ``METRIC_COLUMNS`` aborts, so a new run
        with a new schema needs a new path. Each split stores one Pearson
        per cell line and one ``avg_*`` mean of those three.
    ./gosai_{model}_{filtration}_models/best.pt
        ``state_dict`` of the epoch with the highest ``avg_val_pearson``.
        Pass ``--model_dir`` to change the folder. The test uses this
        file, not the weights left in memory after the last epoch.

Models (``--model``)
--------------------
    moderngena-base     AIRI-Institute/moderngena-base
    moderngena-large    AIRI-Institute/moderngena-large
    ntv3                InstaDeepAI/NTv3_650M_pre
    caduceus            kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16
    dnabert2            zhihan1996/DNABERT-2-117M

Launch
------
``--model`` selects the backbone (default ``moderngena-large``).
``--filtration`` is ``own`` (default) or ``original``. The three cell
lines are always HepG2, K562, and SKNSH. Labels are log2FC.

Leaving ``--batch_size``, ``--lr``, ``--wd``, ``--max_length``, and
``--head_dropout`` unset uses the values in ``MODEL_SPECS``. The shared
batch size is 256. Learning rate is 1e-4 except Caduceus (3e-4). Weight
decay is 0.02 except Caduceus (0.05). Tokenizer length is 602 except
Caduceus (600, the center-crop length). ``--head_dropout`` sets dropout
on the regression head for Caduceus (default 0.3, applied in the pool
and again before the linear map) and DNABERT-2 (default 0.1, applied
only before the linear map). It is ignored for ModernGENA and NTv3.
``--epoch_num`` defaults to 10. ``--device`` is the CUDA index (default
0). ``--root`` defaults to ``<repository>/data``.

Example, ``own`` filtration::

    cd /path/to/MPRA-MNIST/mpramnist/Gosai2024
    python Gosai_hf_model_launch.py --model caduceus

Dataset-side padding, ``original`` filtration::

    python Gosai_hf_model_launch.py --model caduceus --filtration original

For NTv3, export the token in that same shell first::

    export HF_TOKEN=hf_...
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional


def find_mpramnist_repo() -> str:
    """Return the MPRA-MNIST repository root from this file's location.

    Layout this function depends on::

        <repo>/mpramnist/Gosai2024/Gosai_hf_model_launch.py

    ``<repo>`` is two directories above the script. The check is the
    presence of ``mpramnist/__init__.py`` under that root. Call this
    before ``import mpramnist``. If the file is copied elsewhere, the
    check fails on purpose: do not point it at an arbitrary checkout.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(script_dir, "..", ".."))
    package_init = os.path.join(repo, "mpramnist", "__init__.py")
    if os.path.isfile(package_init):
        return repo
    raise SystemExit(
        "This script must stay at MPRA-MNIST/mpramnist/Gosai2024/"
        "Gosai_hf_model_launch.py.\n"
        f"This file is: {os.path.abspath(__file__)}\n"
        f"Expected package init: {package_init}"
    )


# Insert the repository root before importing mpramnist. The script's own
# directory is already on sys.path, but that directory is the Gosai2024
# package, not the root, so `import mpramnist` would fail without this.
# DEFAULT_DATA_ROOT is <repo>/data. Override it with --root only when the
# tables live somewhere else. The dataset class resolves its own table
# under that directory.
MPRA_MNIST_REPO = find_mpramnist_repo()
if MPRA_MNIST_REPO not in sys.path:
    sys.path.insert(0, MPRA_MNIST_REPO)
DEFAULT_DATA_ROOT = os.path.join(MPRA_MNIST_REPO, "data")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

import mpramnist.transforms as t
from mpramnist.Gosai2024.dataset import GosaiDataset


# =============================================================================
# Model catalogue
# =============================================================================
# MODEL_CHOICES is the closed set accepted by --model. MODEL_SPECS holds the
# hyperparameters that are correct for each backbone. To add a backbone:
# append its CLI name to MODEL_CHOICES, add a ModelSpec, and teach
# build_tokenizer_and_model which wrapper to use. Do not reuse another
# model's flags (Caduceus must not take an attention mask or raw-sequence
# reverse-complement augmentation; NTv3 must be loaded as a masked LM).

MODEL_CHOICES = [
    "moderngena-base",
    "moderngena-large",
    "ntv3",
    "caduceus",
    "dnabert2",
]

# Gosai predicts one log2FC per cell line. Head column order is
# HepG2, K562, SKNSH. Both regression heads use this width.
CELL_TYPES = ["HepG2", "K562", "SKNSH"]
NUM_OUTPUTS = len(CELL_TYPES)
SEQ_LEN = 600
FILTRATION = "own"
STDERR_THRESHOLD = 1.0
STD_MULTIPLE_CUT = 6.0
UP_CUTOFF_MOVE = 3.0
DUPLICATION_CUTOFF = 0.5

# Column order of the results TSV. For each of train, val, and test the
# file stores one Pearson per cell line and then avg_{split}_pearson, the
# mean of those three. The model-selection score is avg_val_pearson.
# record=epoch rows fill the training metrics and leave best_epoch and the
# test fields empty. record=test rows are written only after best.pt has
# been loaded. There is no fold column. ensure_results_file refuses to
# append to a file whose header is not exactly this list.
def _pearson_columns(split_name: str) -> list[str]:
    per_cell = [f"{split_name}_pearson_{cell}" for cell in CELL_TYPES]
    return per_cell + [f"avg_{split_name}_pearson"]


METRIC_COLUMNS = [
    "model",
    "cell_types",
    "record",
    "epoch",
    "train_loss",
    "val_loss",
    *_pearson_columns("train"),
    *_pearson_columns("val"),
    "best_epoch",
    "test_loss",
    *_pearson_columns("test"),
]


@dataclass(frozen=True)
class ModelSpec:
    """Frozen hyperparameters for one Hugging Face DNA backbone.

    ``resolve_spec`` copies the entry from ``MODEL_SPECS`` and replaces
    only the fields the CLI is allowed to override: ``max_length``, ``lr``,
    ``weight_decay``, ``batch_size``, and ``head_dropout``. Flags that
    change the architecture (mask, AMP, masked-LM loading, cuDNN) stay on
    the catalogue entry. Do not override those from the command line.
    """

    hub_id: str
    # Tokenizer truncation length. A Gosai sequence is center-cropped to
    # 600 bp. Caduceus uses 600, the crop length. The other tokenizers use
    # 602 so a 600 bp string plus the usual special tokens is not truncated.
    max_length: int
    # AdamW peak learning rate, weight decay, and gradient-clip norm taken
    # from the source training scripts. Caduceus uses 3e-4 / 0.05 / 0.5;
    # the others use 1e-4 / 0.02 / 1.0.
    lr: float
    weight_decay: float
    max_grad_norm: float
    # Shared default is 256. Pass --batch_size only to override one launch.
    batch_size: int
    # Probability of reverse-complementing the DNA string on the training
    # transform. Caduceus-PS already averages forward and reverse-complement
    # states inside the pool, so this must stay 0 for Caduceus. The other
    # models use 0.5. Evaluation always uses probability 0.
    rc_train_prob: float
    # Caduceus-PS has no attention mask. Leave this False for Caduceus so
    # the collate does not pass a mask into the backbone.
    uses_attention_mask: bool
    # NTv3 publishes a masked-LM checkpoint. Set True to load
    # AutoModelForMaskedLM, drop the LM head, and read hidden_states[-1].
    load_as_masked_lm: bool
    # bfloat16 autocast on CUDA. Caduceus was trained in fp32; leave False
    # or the pool and the head see a dtype the source run did not use.
    use_amp: bool
    # Request FlashAttention-2 only when the flash_attn package is installed.
    # Used by ModernGENA. A missing package falls back to the default attention.
    try_flash_attn: bool
    # Set only for Caduceus-PS. The backbone concatenates forward and
    # reverse-complement states, so the pool's input width is 512, not
    # config.hidden_size. None selects the generic token-regression wrapper.
    pooling_hidden_size: Optional[int] = None
    # Dropout on the regression head. Used by Caduceus (pool and linear
    # head) and, when uses_head_dropout is True, by DNABERT-2 (linear head
    # only). --head_dropout overrides this for those two models. ModernGENA
    # and NTv3 keep 0 and ignore the flag.
    head_dropout: float = 0.0
    # True only for DNABERT-2. TokenRegressionModel then applies
    # head_dropout to the pooled vector. Caduceus does not use this flag;
    # its dropout is applied because pooling_hidden_size is set.
    uses_head_dropout: bool = False
    # True only for DNABERT-2. build_tokenizer_and_model then calls
    # load_dnabert2_backbone instead of AutoModel.from_pretrained, because
    # the remote code cannot be built on the meta device used by recent
    # transformers releases.
    load_bert_config: bool = False
    # NTv3 was unstable with cuDNN on some GPUs. True disables cuDNN for
    # the whole process; do not set it on the other models.
    disable_cudnn: bool = False
    # NTv3 is a U-Net with 7 stride-2 stages: the token length must be a
    # multiple of 2**7 = 128, otherwise skip connections get different
    # lengths ("size of tensor a (8) must match tensor b (9)"). The
    # tokenizer pads each batch up to this multiple; padded positions
    # have attention_mask 0. None for the other models.
    pad_to_multiple_of: Optional[int] = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "moderngena-base": ModelSpec(
        hub_id="AIRI-Institute/moderngena-base",
        max_length=602,
        lr=1e-4,
        weight_decay=0.02,
        max_grad_norm=1.0,
        batch_size=256,
        rc_train_prob=0.5,
        uses_attention_mask=True,
        load_as_masked_lm=False,
        use_amp=True,
        try_flash_attn=True,
    ),
    "moderngena-large": ModelSpec(
        hub_id="AIRI-Institute/moderngena-large",
        max_length=602,
        lr=1e-4,
        weight_decay=0.02,
        max_grad_norm=1.0,
        batch_size=256,
        rc_train_prob=0.5,
        uses_attention_mask=True,
        load_as_masked_lm=False,
        use_amp=True,
        try_flash_attn=True,
    ),
    "ntv3": ModelSpec(
        hub_id="InstaDeepAI/NTv3_650M_pre",
        max_length=602,
        lr=1e-4,
        weight_decay=0.02,
        max_grad_norm=1.0,
        batch_size=256,
        rc_train_prob=0.5,
        uses_attention_mask=True,
        load_as_masked_lm=True,
        use_amp=True,
        try_flash_attn=False,
        disable_cudnn=True,
        pad_to_multiple_of=128,
    ),
    "caduceus": ModelSpec(
        hub_id="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        max_length=600,
        lr=3e-4,
        weight_decay=0.05,
        max_grad_norm=0.5,
        batch_size=256,
        rc_train_prob=0.0,
        uses_attention_mask=False,
        load_as_masked_lm=False,
        use_amp=False,
        try_flash_attn=False,
        pooling_hidden_size=512,
        head_dropout=0.3,
    ),
    "dnabert2": ModelSpec(
        hub_id="zhihan1996/DNABERT-2-117M",
        max_length=602,
        lr=1e-4,
        weight_decay=0.02,
        max_grad_norm=1.0,
        batch_size=256,
        rc_train_prob=0.5,
        uses_attention_mask=True,
        load_as_masked_lm=False,
        use_amp=True,
        try_flash_attn=False,
        head_dropout=0.1,
        uses_head_dropout=True,
        load_bert_config=True,
    ),
}


# =============================================================================
# CLI arguments
# =============================================================================
# Every launch trains HepG2, K562, and SKNSH together on the chromosome
# split. --filtration is own (default) or original. The result and
# checkpoint paths default to the working directory and include the model
# name and the filtration, so the two filtrations do not overwrite each
# other. A second launch that appends to an existing TSV continues that file.


def parse_args() -> argparse.Namespace:
    """Parse the launch command.

    Every launch trains all three cell lines on the chromosome split.
    ``--filtration`` is ``own`` or ``original``. Hyperparameters left as
    None are filled from ``MODEL_SPECS`` in ``resolve_spec``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a Hugging Face DNA model on Gosai2024 (three cell "
            "lines, three log2FC outputs) with the chromosome train/val/test split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help=(
            "TSV path. Ten epoch rows, then the test row. "
            "Default: ./gosai_{model}_{filtration}.tsv"
        ),
    )
    general.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=(
            "Directory for best.pt (highest avg_val_pearson). "
            "Default: ./gosai_{model}_{filtration}_models"
        ),
    )
    general.add_argument("--device", type=int, default=0, help="CUDA device index.")
    general.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 if tokenization happens in collate_fn on macOS/Windows.",
    )
    general.add_argument(
        "--model",
        type=str,
        default="moderngena-large",
        choices=MODEL_CHOICES,
        help="Hugging Face DNA backbone to fine-tune.",
    )

    dataset_args = parser.add_argument_group("dataset", "Dataset settings")
    dataset_args.add_argument(
        "--root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Data directory. Default is <MPRA-MNIST repo>/data, resolved "
            "from this file's location. The dataset class finds its table there."
        ),
    )
    dataset_args.add_argument(
        "--filtration",
        type=str,
        default=FILTRATION,
        choices=("own", "original"),
        help=(
            "Gosai filtration. 'own' (default) filters with the stderr and "
            "activity cutoffs below and leaves padding to this launcher "
            "(LEFT_FLANK, RIGHT_FLANK, center crop 600). 'original' pads "
            "each sequence to 600 bp inside the dataset with those same "
            "flanks; this launcher does not add flanks or crop."
        ),
    )

    trainer_args = parser.add_argument_group("trainer", "Training hyperparameters")
    trainer_args.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the model-specific learning rate.",
    )
    trainer_args.add_argument(
        "--wd",
        type=float,
        default=None,
        help="Override the model-specific AdamW weight decay.",
    )
    trainer_args.add_argument(
        "--batch_size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Mini-batch size. Default for every model is 256."
        ),
    )
    trainer_args.add_argument("--epoch_num", type=int, default=10, help="Epochs on the train chromosomes.")
    trainer_args.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Override tokenizer max_length.",
    )
    trainer_args.add_argument(
        "--head_dropout",
        type=float,
        default=None,
        help=(
            "Dropout on the regression head for Caduceus (default 0.3) and "
            "DNABERT-2 (default 0.1). Ignored for ModernGENA and NTv3."
        ),
    )

    args = parser.parse_args()
    if args.result_dir is None:
        args.result_dir = f"./gosai_{args.model}_{args.filtration}.tsv"
    if args.model_dir is None:
        args.model_dir = f"./gosai_{args.model}_{args.filtration}_models"
    return args


def resolve_spec(args: argparse.Namespace) -> ModelSpec:
    """Return the catalogue spec with the CLI overrides applied.

    Only max length, learning rate, weight decay, batch size, and head
    dropout can change here. ``--head_dropout`` is applied for Caduceus and
    DNABERT-2. For ModernGENA and NTv3 the catalogue value 0 is kept even
    if the flag is passed. Architecture flags are copied unchanged so a
    launch cannot silently turn on an attention mask or AMP for a model
    that was not trained that way.
    """
    spec = MODEL_SPECS[args.model]
    if args.head_dropout is not None and (spec.uses_head_dropout or spec.pooling_hidden_size is not None):
        head_dropout = args.head_dropout
    else:
        head_dropout = spec.head_dropout
    return ModelSpec(
        hub_id=spec.hub_id,
        max_length=args.max_length if args.max_length is not None else spec.max_length,
        lr=args.lr if args.lr is not None else spec.lr,
        weight_decay=args.wd if args.wd is not None else spec.weight_decay,
        max_grad_norm=spec.max_grad_norm,
        batch_size=args.batch_size if args.batch_size is not None else spec.batch_size,
        rc_train_prob=spec.rc_train_prob,
        uses_attention_mask=spec.uses_attention_mask,
        load_as_masked_lm=spec.load_as_masked_lm,
        use_amp=spec.use_amp,
        try_flash_attn=spec.try_flash_attn,
        pooling_hidden_size=spec.pooling_hidden_size,
        head_dropout=head_dropout,
        uses_head_dropout=spec.uses_head_dropout,
        load_bert_config=spec.load_bert_config,
        disable_cudnn=spec.disable_cudnn,
        pad_to_multiple_of=spec.pad_to_multiple_of,
    )


# =============================================================================
# Pooling heads
# =============================================================================
# The backbone returns one vector per token. The pool reduces that sequence
# to one vector. The regression head is a linear map from that vector to
# three log2FC activities, in CELL_TYPES order: HepG2, K562, SKNSH. The three
# outputs share the pooled vector; they are not three separate pools.
# MaskedAttentionPooling is for models that pad and expose an attention mask.
# CaduceusAttentionPooling is only for Caduceus-PS: it has no pad mask, and
# its attention logits are averaged with the reverse-complement axis so the
# pool stays RC-invariant. Do not swap the two pools.


class MaskedAttentionPooling(nn.Module):
    """Attention pool that ignores pad tokens.

    Scores are computed in fp32. Softmax in bfloat16 overflows on the pad
    fill value and produces NaNs; the upcast is required when AMP is on.
    Pass the backbone attention mask. Positions with mask 0 receive a large
    negative score and drop out of the weighted sum.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_fp32 = last_hidden_state.float()
        scores = self.attention(hidden_fp32).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(hidden_fp32 * weights, dim=1)


class CaduceusAttentionPooling(nn.Module):
    """RC-invariant attention pool for Caduceus-PS.

    There is no attention mask. The module scores the sequence and its
    token-axis flip, averages those logits, and pools the forward states.
    Dropout inside the score network is active only while the module is in
    train mode. Call ``model.eval()`` before validation and test; otherwise
    the reported Pearson is not the Pearson of the saved weights.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_fp32 = last_hidden_state.float()
        hidden_rc = torch.flip(hidden_fp32, dims=[1])
        logits_fw = self.attention(hidden_fp32)
        logits_rc = self.attention(hidden_rc)
        combined = (logits_fw + torch.flip(logits_rc, dims=[1])) / 2.0
        weights = F.softmax(combined.squeeze(-1), dim=1).unsqueeze(-1)
        return torch.sum(hidden_fp32 * weights, dim=1)


# =============================================================================
# Regression wrappers
# =============================================================================


class TokenRegressionModel(nn.Module):
    """Masked-pool regression wrapper for ModernGENA, NTv3, and DNABERT-2.

    Use ``use_hidden_states=True`` only for NTv3, whose forward returns
    ``hidden_states`` rather than ``last_hidden_state``. After the masked
    attention pool, ``nn.Linear(hidden_size, 3)`` writes one scalar per
    cell line. Column 0 is HepG2, column 1 is K562, column 2 is SKNSH.
    Labels must be ``[batch, 3]`` in that same order. Huber loss
    (delta 1.0) is the mean of every element of that matrix. The returned
    logits stay ``[batch, 3]``. Do not flatten them: Pearson is computed
    per column, and ``avg_val_pearson`` is the mean of the three column
    correlations.

    ``head_dropout`` is applied to the pooled vector before the linear map.
    Pass the DNABERT-2 value from ``ModelSpec``. Leave it at 0 for
    ModernGENA and NTv3. ``nn.Dropout`` is inactive after ``model.eval()``,
    so validation and test Pearson do not include this noise.

    DNABERT-2's remote ``BertModel`` returns a plain tuple
    ``(sequence_output, pooled_output)`` instead of a ``ModelOutput``. The
    first element is the padded ``[batch, tokens, hidden]`` tensor, so it
    is used as the token states. ``return_dict`` is ignored by that model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        use_hidden_states: bool = False,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_hidden_states = use_hidden_states
        self.pool = MaskedAttentionPooling(hidden_size)
        self.regression_head = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(hidden_size, NUM_OUTPUTS),
        ).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if self.use_hidden_states:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            token_states = outputs.hidden_states[-1]
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            if isinstance(outputs, (tuple, list)):
                token_states = outputs[0]
            else:
                token_states = outputs.last_hidden_state

        pooled = self.pool(token_states, attention_mask)
        logits = self.regression_head(pooled)

        loss = None
        if labels is not None:
            loss = nn.HuberLoss(delta=1.0)(logits, labels.float())
        return {"loss": loss, "logits": logits}


class CaduceusRegressionModel(nn.Module):
    """Caduceus-PS wrapper: RC-invariant pool, dropout, three Huber outputs.

    Do not pass ``attention_mask``. Extra keyword arguments are ignored so
    the shared training loop can unpack a batch that sometimes contains a
    mask. ``hidden_size`` must be 512 (``pooling_hidden_size``), because
    Caduceus-PS concatenates both directions. The linear head maps that
    512-d pooled vector to three outputs in ``CELL_TYPES`` order. Dropout
    is applied inside the pool and again on the pooled vector; both are
    inactive after ``model.eval()``.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int = 512, head_dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.pool = CaduceusAttentionPooling(hidden_size, dropout=head_dropout)
        # Dropout on the pooled vector is a no-op after model.eval(), so the
        # checkpoint selected by validation Pearson is the one used at test.
        self.regression_head = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(hidden_size, NUM_OUTPUTS),
        ).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_unused: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, return_dict=True)
        pooled = self.pool(outputs.last_hidden_state)
        logits = self.regression_head(pooled)

        loss = None
        if labels is not None:
            loss = nn.HuberLoss(delta=1.0)(logits, labels.float())
        return {"loss": loss, "logits": logits}


# =============================================================================
# Hugging Face loading
# =============================================================================


def hub_kwargs() -> dict[str, Any]:
    """Keyword arguments shared by every ``from_pretrained`` call.

    ``trust_remote_code=True`` is required: these checkpoints ship custom
    modeling code. A token is added only when ``HF_TOKEN`` or
    ``HUGGING_FACE_HUB_TOKEN`` is already set. NTv3 will fail to download
    without one. Do not hard-code a token in this file.
    """
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    kwargs.update(hub_token_kwargs())
    return kwargs


def hub_token_kwargs() -> dict[str, Any]:
    """Hub token only, for calls that do not accept ``trust_remote_code``.

    ``BertConfig.from_pretrained``, ``get_class_from_dynamic_module``, and
    ``hf_hub_download`` take a token but warn about or reject
    ``trust_remote_code``. Returns an empty dict when no token is set;
    DNABERT-2 is public, so that is enough for it.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"token": token} if token else {}


def load_dnabert2_backbone(hub_id: str) -> nn.Module:
    """Build DNABERT-2 on CPU and load its pretrained encoder weights.

    Use this instead of ``AutoModel.from_pretrained`` for DNABERT-2. It
    fixes three incompatibilities between the 2023 remote code and current
    transformers / torch:

    1. Meta-device construction. Recent transformers builds every model
       inside a meta-device context. The remote ``BertEncoder`` computes
       its ALiBi bias in ``__init__`` from ``torch.Tensor(list)``, which
       ignores that context and lands on CPU, so the multiply with a meta
       tensor raises. Here the class is fetched with
       ``get_class_from_dynamic_module`` and instantiated directly on CPU,
       then the weights are loaded from ``pytorch_model.bin``.
    2. Triton kernel. The checkpoint ships ``flash_attn_triton.py`` written
       for a 2022 Triton pre-release. Linux torch wheels install a modern
       Triton, the import succeeds, and the kernel fails at the first
       bfloat16 forward. Setting the module's ``flash_attn_qkvpacked_func``
       to None makes the remote attention use its own PyTorch path, which
       is numerically the same attention with ALiBi.
    3. Output type. The remote ``BertModel`` returns a tuple, handled in
       ``TokenRegressionModel.forward``.

    The checkpoint is a masked-LM state dict: encoder keys start with
    ``bert.`` and the LM head keys start with ``cls.``. The prefix is
    stripped and the LM head is dropped. The model is built without the
    unused pooler. Any missing or unexpected encoder key aborts the run,
    so a silently half-loaded backbone cannot train.
    """
    from huggingface_hub import hf_hub_download
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.bert.configuration_bert import BertConfig

    token_kw = hub_token_kwargs()
    config = BertConfig.from_pretrained(hub_id, **token_kw)
    model_cls = get_class_from_dynamic_module("bert_layers.BertModel", hub_id, **token_kw)
    sys.modules[model_cls.__module__].flash_attn_qkvpacked_func = None

    with torch.device("cpu"):
        backbone = model_cls(config, add_pooling_layer=False)

    weights_path = hf_hub_download(hub_id, "pytorch_model.bin", **token_kw)
    checkpoint = _load_state_dict(weights_path, torch.device("cpu"))
    encoder_state = {
        key[len("bert."):] if key.startswith("bert.") else key: value
        for key, value in checkpoint.items()
        if not key.startswith("cls.")
    }
    result = backbone.load_state_dict(encoder_state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(
            f"DNABERT-2 checkpoint {weights_path} does not match the remote model.\n"
            f"missing keys: {result.missing_keys}\n"
            f"unexpected keys: {result.unexpected_keys}"
        )
    return backbone


def build_tokenizer_and_model(spec: ModelSpec) -> tuple[Any, nn.Module]:
    """Download one backbone and wrap it with the matching three-output head.

    The wrapper is ``CaduceusRegressionModel`` when ``pooling_hidden_size``
    is set, otherwise ``TokenRegressionModel``. Both end in
    ``Linear(hidden, 3)``.

    Called once per launch so training starts from the published
    checkpoint. The branch
    is selected by ``ModelSpec``, not by the CLI name:

    * ``load_as_masked_lm`` — NTv3. Drop the LM head and read the last
      hidden state.
    * ``pooling_hidden_size is not None`` — Caduceus-PS. Turn the KV cache
      off and use the 512-d RC-concat width.
    * otherwise — ``AutoModel`` plus ``TokenRegressionModel``. ModernGENA
      gets ``head_dropout=0``. DNABERT-2 (``uses_head_dropout``) gets
      dropout on the pooled vector. DNABERT-2 is loaded by
      ``load_dnabert2_backbone``, not by ``from_pretrained``. Neither path
      imports the DNABERT-2 GitHub repository.

    FlashAttention-2 is requested only when ``try_flash_attn`` is set and
    the package imports. A missing package is not an error.
    """
    load_kw = hub_kwargs()
    if spec.try_flash_attn and importlib.util.find_spec("flash_attn") is not None:
        load_kw["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(spec.hub_id, **load_kw)

    if spec.load_as_masked_lm:
        # NTv3 is published as a masked LM. The encoder used for regression
        # is `.model` (fallback `.base_model`). The LM head is discarded.
        full = AutoModelForMaskedLM.from_pretrained(spec.hub_id, **load_kw)
        backbone = full.model if hasattr(full, "model") else full.base_model
        hidden = getattr(getattr(backbone, "config", None), "hidden_size", 1536)
        model = TokenRegressionModel(backbone, hidden_size=hidden, use_hidden_states=True)
        return tokenizer, model

    if spec.pooling_hidden_size is not None:
        # Caduceus-PS is not used for generation here. use_cache=False avoids
        # storing a KV cache on every forward. The pool width is 512.
        config = AutoConfig.from_pretrained(spec.hub_id, **load_kw)
        config.use_cache = False
        backbone = AutoModel.from_pretrained(spec.hub_id, config=config, **load_kw)
        model = CaduceusRegressionModel(
            backbone,
            hidden_size=spec.pooling_hidden_size,
            head_dropout=spec.head_dropout,
        )
        return tokenizer, model

    if spec.load_bert_config:
        backbone = load_dnabert2_backbone(spec.hub_id)
    else:
        backbone = AutoModel.from_pretrained(spec.hub_id, **load_kw)
    hidden = backbone.config.hidden_size
    # DNABERT-2 is this branch with uses_head_dropout. The official 3.8
    # finetune script is not called; only the Hub encoder is wrapped.
    model = TokenRegressionModel(
        backbone,
        hidden_size=hidden,
        use_hidden_states=False,
        head_dropout=spec.head_dropout if spec.uses_head_dropout else 0.0,
    )
    return tokenizer, model


def apply_runtime_quirks(spec: ModelSpec) -> None:
    """Apply process-wide settings that belong to one backbone.

    Currently this only disables cuDNN for NTv3. Call it once, after
    ``resolve_spec`` and before any CUDA work. It is process-global: do
    not run a second model in the same process afterwards.
    """
    if spec.disable_cudnn:
        torch.backends.cudnn.enabled = False


# =============================================================================
# Collate + transforms
# =============================================================================


def make_collate_fn(tokenizer: Any, spec: ModelSpec) -> Callable:
    """Build the DataLoader collate that tokenizes DNA strings.

    ``GosaiDataset`` yields ``(sequence, target)`` where the target
    is a length-3 log2FC vector (HepG2, K562, SKNSH). The sequence may be a string
    or a list of characters after the flank transforms; both are joined
    into one string. Sequences are padded and truncated to
    ``spec.max_length``. Labels are stacked to ``[batch, 3]``. The
    attention mask is included only when ``spec.uses_attention_mask`` is
    true, because Caduceus-PS errors if it receives one. Keep
    ``num_workers`` at 0 unless you have checked that this tokenizer is
    safe to call from worker processes.
    """

    def collate_fn(batch: list[tuple[Any, Any]]) -> dict[str, torch.Tensor]:
        sequences = ["".join(item[0]) if isinstance(item[0], list) else str(item[0]) for item in batch]
        labels = torch.stack(
            [torch.as_tensor(item[1], dtype=torch.float32).reshape(-1) for item in batch]
        )
        if labels.ndim != 2 or labels.shape[1] != NUM_OUTPUTS:
            raise RuntimeError(
                f"Gosai labels must be [batch, {NUM_OUTPUTS}], got {tuple(labels.shape)}."
            )
        encoded = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=spec.max_length,
            pad_to_multiple_of=spec.pad_to_multiple_of,
            return_tensors="pt",
        )
        out = {"input_ids": encoded["input_ids"], "labels": labels}
        if spec.uses_attention_mask:
            if "attention_mask" in encoded:
                out["attention_mask"] = encoded["attention_mask"]
            else:
                out["attention_mask"] = torch.ones_like(encoded["input_ids"])
        return out

    return collate_fn


def build_string_transforms(
    rc_train_prob: float,
    filtration: str,
) -> dict[str, Optional[t.Compose]]:
    """Return train and eval transforms that leave sequences as strings.

    These backbones tokenize strings. ``Seq2Tensor`` is not applied.

    ``filtration='own'``: the dataset does not pad. Training adds
    ``GosaiDataset.LEFT_FLANK`` and ``GosaiDataset.RIGHT_FLANK``,
    center-crops to 600 bp, then reverse-complements with
    ``rc_train_prob``. Pass 0 for Caduceus and 0.5 for the other models.
    Evaluation adds the same two flanks and center-crops to 600 bp.

    ``filtration='original'``: the dataset already pads each sequence to
    600 bp with those two flanks. Training only reverse-complements.
    Evaluation is ``None``, so validation and test keep the padded string.
    """
    if filtration == "original":
        return {
            "train": t.Compose([t.ReverseComplement(rc_train_prob)]),
            "eval": None,
        }
    if filtration != "own":
        raise ValueError(f"filtration must be 'own' or 'original', got {filtration!r}.")

    left_flank = GosaiDataset.LEFT_FLANK
    right_flank = GosaiDataset.RIGHT_FLANK
    train_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(SEQ_LEN),
            t.ReverseComplement(rc_train_prob),
        ]
    )
    eval_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.CenterCrop(SEQ_LEN),
        ]
    )
    return {"train": train_transform, "eval": eval_transform}


# =============================================================================
# Train / validate / test
# =============================================================================
# train_one_epoch updates weights and reports the training-mode metrics.
# evaluate_loader does not update weights. Call evaluate_loader on the
# validation chromosomes at the end of each epoch, and on the test
# chromosomes only after
# the best state dict has been loaded. Pearson for model selection is always
# the validation one.





def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move every collate tensor onto ``device``. Labels stay in the batch."""
    return {key: value.to(device) for key, value in batch.items()}


def _amp_ctx(spec: ModelSpec, device: torch.device):
    """Autocast context for one forward.

    bfloat16 on CUDA when ``spec.use_amp`` is set. Otherwise autocast is
    disabled, including for Caduceus, which stays in fp32. Do not switch
    this to float16: the attention pool is only guarded for bfloat16 via
    the fp32 upcast inside the pool.
    """
    if spec.use_amp and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.amp.autocast(device_type=device.type, enabled=False)


@dataclass(frozen=True)
class PearsonScores:
    """Per-cell Pearson correlations and their mean.

    ``per_cell`` follows ``CELL_TYPES`` and is written to
    ``{split}_pearson_{cell}``. ``mean`` is written to
    ``avg_{split}_pearson``. ``mean`` is NaN when any cell is NaN, so a
    missing cell cannot select a checkpoint.
    """

    per_cell: np.ndarray
    mean: float

    def format(self) -> str:
        parts = [
            f"{cell}={value:.4f}" for cell, value in zip(CELL_TYPES, self.per_cell)
        ]
        return " ".join(parts) + f" mean={self.mean:.4f}"


def _empty_scores() -> PearsonScores:
    return PearsonScores(per_cell=np.full(NUM_OUTPUTS, np.nan), mean=float("nan"))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    spec: ModelSpec,
    device: torch.device,
) -> tuple[float, PearsonScores]:
    """Train for one epoch.

    The model is set to train mode, so Caduceus dropout is on. Returns the
    mean Huber loss over optimizer steps and the per-cell Pearson
    correlations of those same training forwards. This Pearson is not
    comparable to validation Pearson: augmentation and dropout are active.
    Use ``avg_train_pearson`` as the reported training correlation, not as
    a model-selection score. Selection uses ``avg_val_pearson``.
    """
    model.train()
    running = 0.0
    preds, labels = [], []
    bar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(bar):
        optimizer.zero_grad(set_to_none=True)
        batch = _move_batch(batch, device)
        with _amp_ctx(spec, device):
            outputs = model(**batch)
            loss = outputs["loss"].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=spec.max_grad_norm)
        optimizer.step()
        running += loss.item()
        preds.append(outputs["logits"].detach().float().cpu())
        labels.append(batch["labels"].detach().float().cpu())
        bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running / (step + 1):.4f}")
    mean_loss = running / max(len(loader), 1)
    if not preds:
        return mean_loss, _empty_scores()
    train_scores = pearson_per_output(torch.cat(preds).numpy(), torch.cat(labels).numpy())
    return mean_loss, train_scores


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    spec: ModelSpec,
    device: torch.device,
) -> tuple[float, PearsonScores]:
    """Score a loader in eval mode.

    Dropout is off. Returns mean Huber loss and per-cell Pearson
    correlations. The caller writes those correlations to
    ``{split}_pearson_{cell}`` and their mean to ``avg_{split}_pearson``.
    Use this for the validation chromosomes at the end of every epoch and,
    after the best checkpoint has been loaded, for the test chromosomes. An empty
    loader returns NaN rather than raising, so a bad split is visible in
    the TSV.
    """
    model.eval()
    running = 0.0
    steps = 0
    preds, labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _move_batch(batch, device)
        with _amp_ctx(spec, device):
            outputs = model(**batch)
            loss = outputs["loss"].mean()
        running += loss.item()
        steps += 1
        preds.append(outputs["logits"].float().cpu())
        labels.append(batch["labels"].float().cpu())
    if steps == 0:
        return float("nan"), _empty_scores()
    mean_loss = running / steps
    pearson = pearson_per_output(torch.cat(preds).numpy(), torch.cat(labels).numpy())
    return mean_loss, pearson


def pearson_from_arrays(preds: np.ndarray, labels: np.ndarray) -> float:
    """Pearson r of one output against its labels, ignoring NaN labels.

    Returns NaN when fewer than two finite labels remain or when either
    vector is constant. Callers must treat NaN as "no score": it must not
    replace the best checkpoint.
    """
    valid = ~np.isnan(labels)
    if valid.sum() <= 1:
        return float("nan")
    try:
        r, _ = pearsonr(preds[valid], labels[valid])
        return float(r)
    except ValueError:
        return float("nan")


def pearson_per_output(preds: np.ndarray, labels: np.ndarray) -> PearsonScores:
    """Pearson r of each cell-line column, then the mean of the three.

    ``preds`` and ``labels`` are ``[n_sequences, 3]`` in ``CELL_TYPES``
    order. Columns are not pooled into one correlation: HepG2 is never
    mixed with K562. The mean, stored as ``avg_{split}_pearson``, is NaN
    if any column is NaN.
    """
    if preds.ndim != 2 or preds.shape[1] != NUM_OUTPUTS:
        raise RuntimeError(
            f"Predictions must be [n, {NUM_OUTPUTS}], got {tuple(preds.shape)}."
        )
    per_cell = np.array(
        [pearson_from_arrays(preds[:, i], labels[:, i]) for i in range(NUM_OUTPUTS)],
        dtype=np.float64,
    )
    mean = float("nan") if np.isnan(per_cell).any() else float(per_cell.mean())
    return PearsonScores(per_cell=per_cell, mean=mean)


# =============================================================================
# Chromosome split, all three cell lines
# =============================================================================
# main() calls run_split once. Training is --epoch_num epochs (default 10)
# on the chromosome split. There is no k-fold.


def _dataset(
    split: str,
    transform: Optional[t.Compose],
    root: str,
    filtration: str,
) -> GosaiDataset:
    """One Gosai chromosome split for HepG2, K562, and SKNSH.

    ``filtration`` is ``own`` or ``original``. ``duplication_cutoff`` is
    passed for every split, including val and test. ``root`` is the data
    directory; the dataset class finds its table there.
    """
    if filtration not in ("own", "original"):
        raise ValueError(f"filtration must be 'own' or 'original', got {filtration!r}.")
    stderr_columns = [cell + "_lfcSE" for cell in CELL_TYPES]
    return GosaiDataset(
        split=split,
        filtration=filtration,
        cell_types=list(CELL_TYPES),
        stderr_columns=stderr_columns,
        stderr_threshold=STDERR_THRESHOLD,
        std_multiple_cut=STD_MULTIPLE_CUT,
        up_cutoff_move=UP_CUTOFF_MOVE,
        duplication_cutoff=DUPLICATION_CUTOFF,
        use_original_reverse_complement=False,
        transform=transform,
        root=root,
    )


def _score_fields(prefix: str, scores: Optional[PearsonScores]) -> dict[str, float]:
    """TSV cells for one split: one Pearson per cell line, then ``avg_`` mean."""
    avg_name = f"avg_{prefix}_pearson"
    if scores is None:
        fields = {f"{prefix}_pearson_{cell}": np.nan for cell in CELL_TYPES}
        fields[avg_name] = np.nan
        return fields
    fields = {
        f"{prefix}_pearson_{cell}": float(value)
        for cell, value in zip(CELL_TYPES, scores.per_cell)
    }
    fields[avg_name] = scores.mean
    return fields


def run_split(
    args: argparse.Namespace,
    spec: ModelSpec,
    transforms: dict[str, Optional[t.Compose]],
    device: torch.device,
) -> PearsonScores:
    """Train one model on the train chromosomes and test on chromosomes 7 and 13.

    Validation chromosomes are 19, 21, and X. They are never used as the
    test set, and the test chromosomes are never used to choose the
    checkpoint. All three cell lines are trained together.

    After each epoch the validation loader is scored and one ``record=epoch``
    row is appended. The state dict is saved only when ``avg_val_pearson``
    strictly improves. After the last epoch that file is loaded and the
    test chromosomes are scored; the test metrics go on the following
    ``record=test`` row. The weights sitting in memory after the last
    epoch are not the test weights unless that epoch happened to be the best.

    Returns the test scores of the loaded checkpoint. The mean is NaN if
    no finite validation Pearson was observed and therefore no file was
    written.
    """
    cell_label = ",".join(CELL_TYPES)

    tokenizer, model = build_tokenizer_and_model(spec)
    model = model.to(device)
    collate_fn = make_collate_fn(tokenizer, spec)

    train_loader = DataLoader(
        _dataset("train", transforms["train"], args.root, args.filtration),
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        _dataset("val", transforms["eval"], args.root, args.filtration),
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        _dataset("test", transforms["eval"], args.root, args.filtration),
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    # Caduceus source runs use AdamW eps 1e-6; the other backbones use 1e-8.
    # The cosine schedule decays the learning rate across this run's epochs
    # and is stepped once per epoch, after validation, so epoch 1 still sees
    # the peak learning rate.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        eps=1e-6 if args.model == "caduceus" else 1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch_num, eta_min=5e-6)

    os.makedirs(args.model_dir, exist_ok=True)
    ckpt_path = os.path.join(args.model_dir, "best.pt")
    best_val = -np.inf
    best_epoch = None

    for epoch in range(args.epoch_num):
        epoch_num = epoch + 1
        print(
            f"\n=== {args.model} | {cell_label} | "
            f"epoch {epoch_num}/{args.epoch_num} ==="
        )
        train_loss, train_scores = train_one_epoch(model, train_loader, optimizer, spec, device)
        val_loss, val_scores = evaluate_loader(model, val_loader, spec, device)
        print(
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}\n"
            f"  train {train_scores.format()}\n"
            f"  val   {val_scores.format()}"
        )
        append_epoch_row(
            args.result_dir,
            {
                "model": args.model,
                "cell_types": cell_label,
                "record": "epoch",
                "epoch": epoch_num,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **_score_fields("train", train_scores),
                **_score_fields("val", val_scores),
                "best_epoch": np.nan,
                "test_loss": np.nan,
                **_score_fields("test", None),
            },
        )

        # Selection metric is avg_val_pearson only, not validation loss
        # and not avg_train_pearson. A NaN on any cell makes the average
        # NaN and must not overwrite the file.
        if not np.isnan(val_scores.mean) and val_scores.mean > best_val:
            best_val = val_scores.mean
            best_epoch = epoch_num
            torch.save(model.state_dict(), ckpt_path)
            print(f"new best avg_val_pearson={best_val:.4f} -> {ckpt_path}")

        scheduler.step()

    # Test only the saved epoch. Reloading the file also drops dropout state
    # and any in-memory weights from the final epoch when it was not best.
    if best_epoch is None or not os.path.exists(ckpt_path):
        print(f"=== TEST {cell_label}: no checkpoint, test skipped ===")
        test_scores = _empty_scores()
    else:
        model.load_state_dict(_load_state_dict(ckpt_path, torch.device("cpu")))
        test_loss, test_scores = evaluate_loader(model, test_loader, spec, device)
        print(
            f"=== TEST {cell_label}: loaded {ckpt_path} "
            f"(epoch {best_epoch}) test_loss={test_loss:.4f} "
            f"{test_scores.format()} ==="
        )
        append_epoch_row(
            args.result_dir,
            {
                "model": args.model,
                "cell_types": cell_label,
                "record": "test",
                "epoch": np.nan,
                "train_loss": np.nan,
                "val_loss": np.nan,
                **_score_fields("train", None),
                **_score_fields("val", None),
                "best_epoch": best_epoch,
                "test_loss": test_loss,
                **_score_fields("test", test_scores),
            },
        )

    del model, optimizer, scheduler, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_scores


# =============================================================================
# Main
# =============================================================================


def ensure_results_file(path: str) -> None:
    """Create the TSV with ``METRIC_COLUMNS``, or accept an existing one.

    An existing file is usable only when its header matches exactly. A
    mismatch means the file was written with a different column list.
    Pass a new ``--result_dir`` instead of appending incompatible rows.
    New rows are appended and then ``order_results`` rewrites the file.
    The on-disk order is epoch 1..N, then the test row.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=METRIC_COLUMNS).to_csv(path, sep="\t", index=False)
        return
    existing = pd.read_csv(path, sep="\t", nrows=0)
    if list(existing.columns) != METRIC_COLUMNS:
        raise SystemExit(
            f"{path} has columns {list(existing.columns)}, expected {METRIC_COLUMNS}. "
            "Pass a new --result_dir."
        )


def append_epoch_row(path: str, row: dict[str, Any]) -> None:
    """Append one metrics row, then sort the whole TSV.

    Called at the end of every epoch and once after the test, so a killed
    run keeps the rows already finished. Keys must be ``METRIC_COLUMNS``.
    Empty metric cells are NaN; pandas writes them as blank fields.
    Sorting after the append keeps epoch 1 before epoch 2 and puts the
    test row after the last epoch.
    """
    pd.DataFrame([row], columns=METRIC_COLUMNS).to_csv(
        path, sep="\t", index=False, mode="a", header=False
    )
    order_results(path)


def order_results(path: str) -> pd.DataFrame:
    """Rewrite ``path`` so epochs are in ascending order.

    ``record=epoch`` rows come first, epoch 1 then 2 and so on. The
    ``record=test`` row follows the last epoch. The sort is stable.
    """
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        return frame
    frame = frame.assign(
        _record=frame["record"].map({"epoch": 0, "test": 1}).fillna(2),
        _epoch=pd.to_numeric(frame["epoch"], errors="coerce"),
    )
    frame = frame.sort_values(
        by=["_record", "_epoch"],
        ascending=True,
        na_position="last",
        kind="mergesort",
    )
    frame = frame.drop(columns=["_record", "_epoch"])
    frame.to_csv(path, sep="\t", index=False)
    return frame


def _load_state_dict(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a ``state_dict`` written by ``torch.save(model.state_dict(), path)``.

    ``weights_only=True`` is used on PyTorch versions that support it, so
    the checkpoint cannot execute arbitrary code. Older PyTorch raises
    ``TypeError`` on that argument; the fallback is the historical loader.
    The file must be a tensor dict, not a checkpoint with an optimizer.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    """Train HepG2, K562, and SKNSH on the Gosai chromosome split.

    Order of work: resolve the model spec, prepare the TSV and checkpoint
    directory, then ``run_split``. The printed table at the end is the TSV
    as written, including rows from an earlier launch if ``--result_dir``
    already existed.
    """
    args = parse_args()
    spec = resolve_spec(args)
    apply_runtime_quirks(spec)
    ensure_results_file(args.result_dir)
    os.makedirs(args.model_dir, exist_ok=True)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA is not available; running on CPU (this will be very slow).")
    else:
        device = torch.device(f"cuda:{args.device}")

    transforms = build_string_transforms(spec.rc_train_prob, args.filtration)

    print("=" * 60)
    print("LAUNCH PARAMETERS")
    print("=" * 60)
    print(f"  model          : {args.model}")
    print(f"  hub_id         : {spec.hub_id}")
    print(f"  outputs        : {NUM_OUTPUTS} ({', '.join(CELL_TYPES)})")
    print("  split          : train chr 1-6,8-12,14-18,20,22,Y")
    print("                   val chr 19,21,X")
    print("                   test chr 7,13")
    print("  labels         : log2FC")
    print(f"  filtration     : {args.filtration}")
    print(f"  batch_size     : {spec.batch_size}")
    print(f"  epoch_num      : {args.epoch_num}")
    print(f"  lr             : {spec.lr}")
    print(f"  weight_decay   : {spec.weight_decay}")
    print(f"  max_length     : {spec.max_length}")
    print(f"  head_dropout   : {spec.head_dropout}")
    print(f"  device         : {device}")
    print(f"  root           : {args.root}")
    print(f"  result_dir     : {args.result_dir}")
    print(f"  model_dir      : {args.model_dir}")
    print("=" * 60)

    print(
        f"\n### chromosome split — model: {args.model} — "
        f"cells: {', '.join(CELL_TYPES)} ###"
    )
    test_scores = run_split(args, spec, transforms, device)

    results = pd.read_csv(args.result_dir, sep="\t")
    print("\n" + "=" * 60)
    print(results.to_string(index=False))
    print("=" * 60)
    print(
        "avg_test_pearson: "
        f"{None if np.isnan(test_scores.mean) else round(test_scores.mean, 4)}"
    )
    for index, cell in enumerate(CELL_TYPES):
        value = test_scores.per_cell[index]
        printed = None if np.isnan(value) else round(float(value), 4)
        print(f"test Pearson {cell}: {printed}")
    print(f"checkpoints: {args.model_dir}")


if __name__ == "__main__":
    main()
