"""
Fine-tune one Hugging Face DNA model on deAlmeida2022.

Flags
-----
--model                              backbone; default moderngena-large
--device                             CUDA index; default 0
--num_workers                        DataLoader workers; default 0
--root                               data directory; default <repo>/data
--use_original_reverse_complement    double the train split with reverse complements; default off
--result_dir                         results folder; default ./results/{model}
--model_dir                          checkpoint folder; default the same results folder
--lr                                 learning-rate override
--wd                                 weight-decay override
--batch_size                         batch size; default 256
--runs                               independent training runs; default 5
--seed                               seed of run 1; run k uses seed+k-1; default 0
--epoch_num                          epochs per run; default 10
--max_length                         tokenizer length override
--head_dropout                       head dropout for Caduceus and DNABERT-2
--scheduler                          epoch_cosine (default), step_cosine, step_linear,
                                     onecycle, wsd, constant, plateau
--warmup_ratio                       linear warmup fraction (not epoch_cosine/onecycle); default 0
--decay_ratio                        wsd final decay fraction; default 0.2 with wsd
--head_lr_mult                       pool/head LR = lr * this; default 1
--no_decay_bias_norm                 no weight decay on biases and norm weights
--backbone_dropout                   ModernGENA backbone dropout; default checkpoint
--ema_decay                          EMA weights for val, best.pt, test; default off
--rc_tta                             average val/test over forward and RC; default off

The description continues below.

This file must stay at ``MPRA-MNIST/mpramnist/deAlmeida2022/`` and must be
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
    (Python 3.10 or newer) and trains the two-output head defined below.
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

One launch, two promoter activities
------------------------------------
deAlmeida2022 stores two log2 activities per 249 bp sequence, measured
in Drosophila S2 cells. Every launch trains both together, in this
order: ``Dev_log2`` (developmental promoter), ``Hk_log2`` (housekeeping
promoter). ``--root`` is the data directory (default ``<repo>/data``).
The dataset class locates its table under that directory.

Output 0 is ``Dev_log2``. Output 1 is ``Hk_log2``.

Regression head
---------------
The pool reduces the token states to one vector. The regression head is
dropout (Caduceus and DNABERT-2 only) followed by
``nn.Linear(hidden_size, 2)``. One shared vector produces both outputs.
Logits stay shaped ``[batch, 2]`` and are not flattened. Huber loss
(delta 1.0) is the mean over the batch and over the two outputs.
Pearson is computed on each column separately, so a developmental
prediction is never correlated with a housekeeping label.

Split, 10 epochs
----------------
There is no k-fold. One launch repeats training ``--runs`` times
(default 5). Each run has its own folder and seed, and trains for
``--epoch_num`` epochs
(default 10) on the dataset split column:

    train    native sequences
    val      first half of chromosome 2R
    test     second half of chromosome 2R

    for epoch in 1..10:
        train on split=train
        validate on split=val
        append one row to epochs.tsv
        if avg_val_pearson is the best so far, write best.pt
    after the last epoch:
        load best.pt
        score split=test with those weights only
        write one row to test.tsv

Epoch rows go to ``epochs.tsv`` and the test row to ``test.tsv``, both in
the run folder. There is no fold column.

Per-target columns are ``train_pearson_Dev_log2`` and
``train_pearson_Hk_log2``, and the same pattern for val and test. The
mean of the two is ``avg_train_pearson``, ``avg_val_pearson``, or
``avg_test_pearson``. ``avg_train_pearson`` comes from the training
forwards (dropout is on for Caduceus). ``val_loss`` and
``avg_val_pearson`` are computed in eval mode. The checkpoint is chosen
by ``avg_val_pearson``. A NaN on either target makes that average NaN
and does not replace the saved weights.

Sequences
---------
Each sequence is already 249 bp. This launcher does not add flanks and
does not crop.

``--use_original_reverse_complement`` defaults to false. The dataset
then returns each training sequence once. Training reverse-complements
the string with probability 0.5 (0 for Caduceus, whose pool already
averages the reverse complement).

Pass ``--use_original_reverse_complement`` to set the dataset flag to
true on the train split only. The loader appends the reverse complement
of every training sequence and copies its two labels, so the train table
is twice as long. Val and test stay unaugmented. In that mode the train
transform uses ``ReverseComplement(0)``: the complement is already in
the table, and a second random flip is not applied. Caduceus still
averages forward and reverse-complement states inside its pool.

Validation and test always call ``ReverseComplement(0)``, which leaves
the sequence unchanged.

Outputs, in the working directory
----------------------------------
    ./results/{model}/run1/
    ./results/{model}/run2/
        One folder per run, ``run1`` through ``run{--runs}``, as in the
        Fromel2025 launcher. With ``--use_original_reverse_complement``
        the default parent is ``./results/{model}_origrc``, so the two
        modes do not share files. Inside each folder:
            epochs.tsv
                One row per epoch: train and validation loss, one Pearson
                per promoter activity and their ``avg_*`` mean.
            test.tsv
                The test row of best.pt: best epoch, test loss, one
                Pearson per promoter activity and ``avg_test_pearson``.
            best.pt
                ``state_dict`` of the epoch with the highest
                ``avg_val_pearson`` in that run. ``test.tsv`` is scored
                with this file, not the weights left in memory after the
                last epoch.
            seed.txt
                The integer seed used for that run. Run 1 uses ``--seed``
                (default 0). Run k uses ``--seed + k - 1``.

Pass ``--result_dir`` to use another parent folder (``--model_dir``
defaults to the same folder). A TSV whose header differs from the one
this launch would write aborts the launch; pass a new folder.

Opt-in fine-tuning options
--------------------------
Every option below defaults to the behaviour this launcher had before the
option existed, so an old command reproduces its old TSV and best.pt. The
code is the "Opt-in fine-tuning options" section, copy-identical across
the six Hugging Face launchers.

``--scheduler``
    ``epoch_cosine`` (default) is ``CosineAnnealingLR(T_max=epoch_num,
    eta_min=5e-6)`` stepped once per epoch after validation. The others
    count ``T = epoch_num * len(train_loader)`` optimizer steps,
    ``W = round(warmup_ratio * T)`` warmup steps, and scale every param
    group from its own LR ``lr_g`` (so ``--head_lr_mult`` holds). Warmup is
    ``lr_g * (s + 1) / W`` for step ``s < W``; ``p`` is the fraction of the
    post-warmup steps taken, ``eta = 5e-6``:

    step_cosine  ``eta + (lr_g - eta) * 0.5 * (1 + cos(pi * p))``
    step_linear  ``eta + (lr_g - eta) * (1 - p)`` (the usual HF schedule)
    constant     ``lr_g`` after the warmup (pairs well with ``--ema_decay``)
    wsd          ``lr_g`` until the last ``D = round(decay_ratio * T)``
                 steps, then a cosine from ``lr_g`` to ``eta`` over them
    onecycle     ``OneCycleLR(max_lr=lr_g, total_steps=T, pct_start=0.3,
                 three_phase=False, cycle_momentum=False)`` per step, as in
                 the MPRA-MNIST CNN trainers: cosine up from ``lr_g / 25``
                 over 30 % of the steps, then cosine down to ``lr_g / 25e4``.
                 ``--warmup_ratio`` must be 0.
    plateau      ``ReduceLROnPlateau(mode="max", factor=0.5, patience=1,
                 min_lr=5e-6)``, stepped once per epoch on the best.pt
                 selection score (with ``--rc_tta`` the strand-averaged
                 one); NaN epochs are skipped. The optional warmup runs per
                 step first; plateau steps are ignored until it is done.

    The scheduler, ``T``, ``W``, and ``D`` are printed when the optimizer
    is built (once per run, act, or fold).
``--warmup_ratio``
    Default 0. A value above 0 with ``epoch_cosine`` or ``onecycle``
    aborts the launch.
``--decay_ratio``
    ``wsd`` only (default 0.2 there); any other scheduler with it aborts,
    and so does ``warmup_ratio + decay_ratio > 1``.
``--head_lr_mult``
    Default 1. Parameters under ``model.backbone`` keep ``lr``; the pool and
    the regression head use ``lr * head_lr_mult``. Both schedulers scale every
    group from its own starting LR. At 1, and without
    ``--no_decay_bias_norm``, AdamW receives the single old parameter group.
``--no_decay_bias_norm``
    Off by default. Biases and 1-D weights of norm modules (LayerNorm,
    RMSNorm) get weight decay 0, in the backbone group and in the head group.
``--backbone_dropout``
    Default: the checkpoint config. ModernGENA only: sets
    ``embedding_dropout``, ``attention_dropout``, and ``mlp_dropout`` of its
    ``ModernBertConfig`` before ``from_pretrained``. For NTv3, Caduceus, and
    DNABERT-2 a notice says the flag is ignored.
``--ema_decay``
    Off by default. Keeps an fp32 exponential moving average of the trainable
    weights on the training device, updated after every optimizer step with
    ``d_t = min(ema_decay, (1 + t) / (10 + t))``. Validation, the choice of
    best.pt, the saved best.pt, and the test use the EMA weights; training
    loss and training Pearson stay from the live weights. Memory: 4 bytes per
    parameter for the run, plus 4 bytes per parameter while validating.
    Choose it from the optimizer steps per epoch: the average spans roughly
    ``1 / (1 - ema_decay)`` steps, so 0.99-0.999 for small datasets and
    0.999-0.9999 for epochs of tens of thousands of steps.
``--rc_tta``
    Off by default. Validation and test predictions are the mean of the
    forward string and its reverse complement: ``mpramnist.transforms.
    reverse_complement`` is applied in the collate to the string the
    evaluation transform produced (flanks and crops included), and both
    strands are scored in the same batch, so rows and order are identical.
    ``val_loss`` and ``test_loss`` become the mean of the two strands' losses.
    The checkpoint is then chosen by the strand-averaged validation Pearson,
    so a run with ``--rc_tta`` may select a different epoch than one without.
    Training is unchanged.

Models (``--model``)
--------------------
    moderngena-base     AIRI-Institute/moderngena-base
    moderngena-large    AIRI-Institute/moderngena-large
    ntv3                InstaDeepAI/NTv3_650M_pre
    caduceus            kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16
    dnabert2            zhihan1996/DNABERT-2-117M

Launch
------
``--model`` selects the backbone (default ``moderngena-large``). Both
promoter activities are always ``Dev_log2`` and ``Hk_log2``.
``--use_original_reverse_complement`` is off unless the flag is passed.

Leaving ``--batch_size``, ``--lr``, ``--wd``, ``--max_length``, and
``--head_dropout`` unset uses the values in ``MODEL_SPECS``. The shared
batch size is 256. Learning rate is 1e-4 except Caduceus (3e-4). Weight
decay is 0.02 except Caduceus (0.05). Tokenizer length is 256 except
Caduceus (249, the sequence length). ``--head_dropout`` sets dropout
on the regression head for Caduceus (default 0.3, applied in the pool
and again before the linear map) and DNABERT-2 (default 0.1, applied
only before the linear map). It is ignored for ModernGENA and NTv3.
``--epoch_num`` defaults to 10. ``--device`` is the CUDA index (default
0). ``--root`` defaults to ``<repository>/data``.

Example::

    cd /path/to/MPRA-MNIST/mpramnist/deAlmeida2022
    python deAlmeida_hf_model_launch.py --model caduceus

Train split doubled by the dataset reverse complement::

    python deAlmeida_hf_model_launch.py --model caduceus --use_original_reverse_complement

For NTv3, export the token in that same shell first::

    export HF_TOKEN=hf_...
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional


def find_mpramnist_repo() -> str:
    """Return the MPRA-MNIST repository root from this file's location.

    Layout this function depends on::

        <repo>/mpramnist/deAlmeida2022/deAlmeida_hf_model_launch.py

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
        "This script must stay at MPRA-MNIST/mpramnist/deAlmeida2022/"
        "deAlmeida_hf_model_launch.py.\n"
        f"This file is: {os.path.abspath(__file__)}\n"
        f"Expected package init: {package_init}"
    )


# Insert the repository root before importing mpramnist. The script's own
# directory is already on sys.path, but that directory is the deAlmeida2022
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

import mpramnist.transforms as t
from mpramnist.deAlmeida2022.dataset import deAlmeidaDataset


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

# deAlmeida predicts one log2 activity per promoter type. Head column
# order is Dev_log2, then Hk_log2. Both regression heads use this width.
CELL_TYPES = ["Dev_log2", "Hk_log2"]
NUM_OUTPUTS = len(CELL_TYPES)

# Each run folder holds epochs.tsv (one row per epoch) and test.tsv (the
# test row of best.pt), as in the Fromel2025 launcher. For each split the
# file stores one Pearson per output and then avg_{split}_pearson, the mean
# of those. The model-selection score is avg_val_pearson. append_row
# refuses to append to a file whose header is not exactly the list below.
def _pearson_columns(split_name: str) -> list[str]:
    per_cell = [f"{split_name}_pearson_{cell}" for cell in CELL_TYPES]
    return per_cell + [f"avg_{split_name}_pearson"]


EPOCH_COLUMNS = [
    "model",
    "cell_types",
    "epoch",
    "train_loss",
    "val_loss",
    *_pearson_columns("train"),
    *_pearson_columns("val"),
]
TEST_COLUMNS = [
    "model",
    "cell_types",
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
    # Tokenizer truncation length. A deAlmeida sequence is 249 bp.
    # Caduceus uses 249, the sequence length. The other tokenizers use
    # 256 so a 249 bp string plus the usual special tokens is not truncated.
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
        max_length=256,
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
        max_length=256,
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
        max_length=256,
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
        max_length=249,
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
        max_length=256,
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
# Every launch trains Dev_log2 and Hk_log2 together on the dataset split.
# --use_original_reverse_complement defaults to off. The result and
# checkpoint paths include the model name, and add _origrc when that flag
# is on, so the two modes do not share a file. A second launch that
# appends to an existing TSV continues that file.


def parse_args() -> argparse.Namespace:
    """Parse the launch command.

    Every launch trains both promoter activities on the train/val/test
    split. Hyperparameters left as None are filled from ``MODEL_SPECS``
    in ``resolve_spec``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a Hugging Face DNA model on deAlmeida2022 (two "
            "promoter activities, two log2 outputs) for 10 epochs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help=(
            "Results folder. Each run writes run{k}/epochs.tsv, run{k}/test.tsv, "
            "run{k}/seed.txt, and run{k}/best.pt. "
            "Default: ./results/{model}, or ./results/{model}_origrc when --use_original_reverse_complement is set."
        ),
    )
    general.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=(
            "Parent folder for best.pt. Each run uses run{k}/best.pt inside it. "
            "Default is the same folder as --result_dir."
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
        "--use_original_reverse_complement",
        action="store_true",
        help=(
            "On the train split only, append each sequence's reverse "
            "complement and copy its Dev_log2 and Hk_log2 labels. Default "
            "is off. Val and test are unchanged. When this flag is set, "
            "the train transform does not reverse-complement again."
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
    trainer_args.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Independent training runs. Each run gets its own folder.",
    )
    trainer_args.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed of run 1. Run k uses this value plus k-1. "
            "The seed is written to seed.txt in that run's folder."
        ),
    )
    trainer_args.add_argument("--epoch_num", type=int, default=10, help="Epochs in each run.")
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

    add_finetune_args(parser)

    args = parser.parse_args()
    check_finetune_args(parser, args)
    tag = "_origrc" if args.use_original_reverse_complement else ""
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")
    if args.result_dir is None:
        args.result_dir = os.path.join("results", f"{args.model}{tag}")
    if args.model_dir is None:
        args.model_dir = args.result_dir
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
# two log2 activities, in CELL_TYPES order: Dev_log2, Hk_log2. The two
# outputs share the pooled vector; they are not two separate pools.
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
    attention pool, ``nn.Linear(hidden_size, 2)`` writes one scalar per
    promoter activity. Column 0 is Dev_log2, column 1 is Hk_log2.
    Labels must be ``[batch, 2]`` in that same order. Huber loss
    (delta 1.0) is the mean of every element of that matrix. The returned
    logits stay ``[batch, 2]``. Do not flatten them: Pearson is computed
    per column, and ``avg_val_pearson`` is the mean of the two column
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
    """Caduceus-PS wrapper: RC-invariant pool, dropout, two Huber outputs.

    Do not pass ``attention_mask``. Extra keyword arguments are ignored so
    the shared training loop can unpack a batch that sometimes contains a
    mask. ``hidden_size`` must be 512 (``pooling_hidden_size``), because
    Caduceus-PS concatenates both directions. The linear head maps that
    512-d pooled vector to two outputs in ``CELL_TYPES`` order. Dropout
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


def build_tokenizer_and_model(
    spec: ModelSpec,
    backbone_dropout: Optional[float] = None,
) -> tuple[Any, nn.Module]:
    """Download one backbone and wrap it with the matching two-output head.

    The wrapper is ``CaduceusRegressionModel`` when ``pooling_hidden_size``
    is set, otherwise ``TokenRegressionModel``. Both end in
    ``Linear(hidden, 2)``.

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
    elif backbone_dropout is not None:
        # --backbone_dropout (ModernGENA only): edit the ModernBERT config
        # before the weights are loaded. Without the flag the checkpoint
        # config is used unchanged, as before.
        config = AutoConfig.from_pretrained(spec.hub_id, **hub_kwargs())
        apply_backbone_dropout(config, backbone_dropout)
        backbone = AutoModel.from_pretrained(spec.hub_id, config=config, **load_kw)
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


def make_collate_fn(tokenizer: Any, spec: ModelSpec, add_rc: bool = False) -> Callable:
    """Build the DataLoader collate that tokenizes DNA strings.

    ``deAlmeidaDataset`` yields ``(sequence, target)`` where the target
    is a length-2 vector (Dev_log2, Hk_log2). The sequence may be a string
    or a list of characters after the flank transforms; both are joined
    into one string. Sequences are padded and truncated to
    ``spec.max_length``. Labels are stacked to ``[batch, 2]``. The
    attention mask is included only when ``spec.uses_attention_mask`` is
    true, because Caduceus-PS errors if it receives one. Keep
    ``num_workers`` at 0 unless you have checked that this tokenizer is
    safe to call from worker processes.

    ``add_rc=True`` is for evaluation loaders under ``--rc_tta``: the batch
    also gets ``rc_input_ids`` (and ``rc_attention_mask``), the tokens of
    ``mpramnist.transforms.reverse_complement`` of each string after the
    evaluation transform. ``split_rc_batch`` separates the two strands.
    """

    def collate_fn(batch: list[tuple[Any, Any]]) -> dict[str, torch.Tensor]:
        sequences = ["".join(item[0]) if isinstance(item[0], list) else str(item[0]) for item in batch]
        labels = torch.stack(
            [torch.as_tensor(item[1], dtype=torch.float32).reshape(-1) for item in batch]
        )
        if labels.ndim != 2 or labels.shape[1] != NUM_OUTPUTS:
            raise RuntimeError(
                f"deAlmeida labels must be [batch, {NUM_OUTPUTS}], got {tuple(labels.shape)}."
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
        if add_rc:
            rc_encoded = tokenizer(
                [t.reverse_complement(sequence) for sequence in sequences],
                padding=True,
                truncation=True,
                max_length=spec.max_length,
                pad_to_multiple_of=spec.pad_to_multiple_of,
                return_tensors="pt",
            )
            out["rc_input_ids"] = rc_encoded["input_ids"]
            if spec.uses_attention_mask:
                if "attention_mask" in rc_encoded:
                    out["rc_attention_mask"] = rc_encoded["attention_mask"]
                else:
                    out["rc_attention_mask"] = torch.ones_like(rc_encoded["input_ids"])
        return out

    return collate_fn


def build_string_transforms(
    rc_train_prob: float,
    use_original_reverse_complement: bool,
) -> dict[str, t.Compose]:
    """Return train and eval transforms that leave sequences as strings.

    These backbones tokenize strings. ``Seq2Tensor`` is not applied.
    Sequences are already 249 bp, so this function does not add flanks
    and does not crop.

    Training reverse-complements with ``rc_train_prob`` (0 for Caduceus,
    0.5 for the other models). If ``use_original_reverse_complement`` is
    true, the train probability is forced to 0: the dataset already
    stored both strands. Evaluation always calls ``ReverseComplement(0)``,
    which keeps the original string.
    """
    train_prob = 0.0 if use_original_reverse_complement else rc_train_prob
    return {
        "train": t.Compose([t.ReverseComplement(train_prob)]),
        "eval": t.Compose([t.ReverseComplement(0)]),
    }


# Help text of --rc_tta, the one per-launcher line of the fine-tuning options.
RC_TTA_HELP = (
    "Average validation and test predictions over the forward string and its "
    "reverse complement; checkpoint selection then uses the averaged val score."
)


# =============================================================================
# Opt-in fine-tuning options
# =============================================================================
# Everything in this section is copy-identical across the six *_hf_model_launch
# scripts. Every flag defaults to the behaviour these launchers had before the
# flags existed: one AdamW group with weight decay on every parameter,
# CosineAnnealingLR(T_max=epoch_num, eta_min=5e-6) stepped once per epoch after
# validation, no warmup, checkpoint backbone dropout, no EMA, forward-strand
# evaluation. With all defaults the optimizer, the scheduler, the loaders, and
# the random-number stream are built exactly as before, so an old command
# reproduces its old TSV.

SCHEDULER_CHOICES = (
    "epoch_cosine",
    "step_cosine",
    "step_linear",
    "onecycle",
    "wsd",
    "constant",
    "plateau",
)
WSD_DEFAULT_DECAY_RATIO = 0.2
COSINE_ETA_MIN = 5e-6

# --backbone_dropout applies only to these backbones. ModernGENA is a ModernBERT
# checkpoint; MODERNBERT_DROPOUT_FIELDS are the ModernBertConfig dropout fields
# (transformers 4.48+ and 5.x). classifier_dropout is not listed: the HF
# classifier head is not used. NTv3 (remote U-Net code), Caduceus-PS, and
# DNABERT-2 (custom loader) are not listed, so the flag is ignored for them
# with a printed notice.
BACKBONE_DROPOUT_MODELS = ("moderngena-base", "moderngena-large")
MODERNBERT_DROPOUT_FIELDS = ("embedding_dropout", "attention_dropout", "mlp_dropout")


def add_finetune_args(parser: argparse.ArgumentParser) -> None:
    """Add the opt-in fine-tuning flags. Every default is the old behaviour."""
    group = parser.add_argument_group(
        "fine-tuning options",
        "Opt-in. Every default reproduces runs made before these flags existed.",
    )
    group.add_argument(
        "--scheduler",
        type=str,
        default="epoch_cosine",
        choices=SCHEDULER_CHOICES,
        help=(
            "epoch_cosine: CosineAnnealingLR(T_max=epoch_num, eta_min=5e-6) "
            "stepped once per epoch after validation (the old behaviour). "
            "Per optimizer step over epoch_num * len(train_loader) steps, after an "
            "optional linear warmup (--warmup_ratio): step_cosine (cosine to 5e-6), "
            "step_linear (linear to 5e-6), constant (peak LR), wsd (peak LR, then "
            "cosine to 5e-6 over the last --decay_ratio of steps), onecycle "
            "(OneCycleLR, pct_start=0.3, no --warmup_ratio). plateau: "
            "ReduceLROnPlateau(max, factor=0.5, patience=1, min_lr=5e-6) on the "
            "best.pt validation score once per epoch, after the optional warmup."
        ),
    )
    group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of all optimizer steps spent in linear LR warmup. "
            "Not allowed with epoch_cosine or onecycle."
        ),
    )
    group.add_argument(
        "--decay_ratio",
        type=float,
        default=None,
        help=(
            "wsd only: fraction of all optimizer steps in the final decay to "
            "5e-6. Default with wsd: 0.2."
        ),
    )
    group.add_argument(
        "--head_lr_mult",
        type=float,
        default=1.0,
        help=(
            "Learning rate of the pool and the regression head is lr * this. "
            "Parameters under model.backbone keep lr."
        ),
    )
    group.add_argument(
        "--no_decay_bias_norm",
        action="store_true",
        help="Exclude biases and 1-D norm weights (LayerNorm/RMSNorm) from weight decay.",
    )
    group.add_argument(
        "--backbone_dropout",
        type=float,
        default=None,
        help=(
            "Set ModernGENA embedding/attention/MLP dropout before loading. "
            "Default: keep the checkpoint config. Ignored for NTv3, Caduceus, DNABERT-2."
        ),
    )
    group.add_argument(
        "--ema_decay",
        type=float,
        default=None,
        help=(
            "Keep an fp32 EMA of the trainable weights; validation, best.pt, "
            "and the test use it. Default: off."
        ),
    )
    group.add_argument(
        "--rc_tta",
        action="store_true",
        help=RC_TTA_HELP,
    )


def check_finetune_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject flag values and combinations that cannot do what they say."""
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error(f"--warmup_ratio must be in [0, 1), got {args.warmup_ratio}.")
    if args.warmup_ratio > 0 and args.scheduler == "epoch_cosine":
        parser.error(
            "--warmup_ratio > 0 needs a step-level --scheduler; epoch_cosine "
            "steps once per epoch and has no warmup."
        )
    if args.warmup_ratio > 0 and args.scheduler == "onecycle":
        parser.error(
            "--warmup_ratio must be 0 with --scheduler onecycle; its warmup is "
            "pct_start=0.3."
        )
    if args.decay_ratio is not None and args.scheduler != "wsd":
        parser.error("--decay_ratio is only valid with --scheduler wsd.")
    if args.scheduler == "wsd":
        if args.decay_ratio is None:
            args.decay_ratio = WSD_DEFAULT_DECAY_RATIO
        if not 0.0 < args.decay_ratio <= 1.0:
            parser.error(f"--decay_ratio must be in (0, 1], got {args.decay_ratio}.")
        if args.warmup_ratio + args.decay_ratio > 1.0:
            parser.error(
                f"--warmup_ratio + --decay_ratio must be <= 1, got "
                f"{args.warmup_ratio} + {args.decay_ratio}."
            )
    if args.head_lr_mult <= 0:
        parser.error(f"--head_lr_mult must be > 0, got {args.head_lr_mult}.")
    if args.backbone_dropout is not None and not 0.0 <= args.backbone_dropout < 1.0:
        parser.error(f"--backbone_dropout must be in [0, 1), got {args.backbone_dropout}.")
    if args.ema_decay is not None and not 0.0 < args.ema_decay < 1.0:
        parser.error(f"--ema_decay must be in (0, 1), got {args.ema_decay}.")


def resolve_backbone_dropout(model_name: str, value: Optional[float]) -> Optional[float]:
    """Return ``value`` when this backbone supports it, otherwise None with a notice."""
    if value is None:
        return None
    if model_name not in BACKBONE_DROPOUT_MODELS:
        print(
            f"NOTICE: --backbone_dropout {value} is ignored for {model_name}; "
            "its backbone keeps the checkpoint dropout config."
        )
        return None
    return value


def apply_backbone_dropout(config: Any, value: float) -> None:
    """Write ``value`` into the ModernBERT dropout fields of ``config``.

    Aborts when the config has none of ``MODERNBERT_DROPOUT_FIELDS``, so a
    renamed field in a future transformers release cannot be ignored silently.
    """
    fields = [name for name in MODERNBERT_DROPOUT_FIELDS if hasattr(config, name)]
    if not fields:
        raise SystemExit(
            f"{type(config).__name__} has none of {MODERNBERT_DROPOUT_FIELDS}; "
            "cannot apply --backbone_dropout."
        )
    for name in fields:
        print(f"  backbone config {name}: {getattr(config, name)} -> {value}")
        setattr(config, name, value)


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    eps: float,
    head_lr_mult: float = 1.0,
    no_decay_bias_norm: bool = False,
) -> torch.optim.Optimizer:
    """AdamW, optionally with a head/backbone LR split and no decay on bias/norm.

    With both options at their defaults this is the old call,
    ``AdamW(model.parameters(), lr, weight_decay, eps)``: one group, decay on
    every parameter. Otherwise parameters are split into up to four groups:
    names under ``backbone.`` get ``lr``, everything else (pool and
    regression head) gets ``lr * head_lr_mult``; with ``no_decay_bias_norm``
    each of those is split again, and biases plus 1-D weights of modules whose
    class name contains "Norm" (LayerNorm, RMSNorm, ...) get weight decay 0.
    Schedulers scale every group from its own starting LR.
    """
    if head_lr_mult == 1.0 and not no_decay_bias_norm:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)

    no_decay_ids: set[int] = set()
    if no_decay_bias_norm:
        for module in model.modules():
            is_norm = "norm" in type(module).__name__.lower()
            for param_name, param in module.named_parameters(recurse=False):
                if param.ndim <= 1 and (param_name == "bias" or is_norm):
                    no_decay_ids.add(id(param))

    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = (not name.startswith("backbone."), id(param) in no_decay_ids)
        groups.setdefault(key, []).append(param)

    param_groups = []
    for (is_head, no_decay), params in sorted(groups.items()):
        group_lr = lr * head_lr_mult if is_head else lr
        group_wd = 0.0 if no_decay else weight_decay
        count = sum(param.numel() for param in params)
        print(
            f"  param group {'head' if is_head else 'backbone'}"
            f"{' no_decay' if no_decay else ''}: {len(params)} tensors, "
            f"{count:,} params, lr={group_lr:g}, weight_decay={group_wd:g}"
        )
        param_groups.append({"params": params, "lr": group_lr, "weight_decay": group_wd})
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, eps=eps)


def lr_factor(
    scheduler_name: str,
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    floor: float,
) -> float:
    """LR multiplier of one param group before optimizer step ``step`` (0-based).

    Pure Python, so it can be checked without torch. ``floor`` is
    ``COSINE_ETA_MIN / group_lr``, so the decaying schedules end at 5e-6
    whatever the group's peak LR is. With ``p`` the fraction of the
    post-warmup steps already taken:

        warmup (all)   (step + 1) / warmup_steps, for step < warmup_steps
        step_cosine    floor + (1 - floor) * 0.5 * (1 + cos(pi * p))
        step_linear    floor + (1 - floor) * (1 - p)
        constant       1
        wsd            1 until the last decay_steps steps, then a cosine from
                       1 to floor over those steps
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if scheduler_name == "constant":
        return 1.0
    if scheduler_name == "wsd":
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / max(1, decay_steps))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    if scheduler_name == "step_cosine":
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if scheduler_name == "step_linear":
        return floor + (1.0 - floor) * (1.0 - progress)
    raise ValueError(f"lr_factor does not handle scheduler {scheduler_name!r}.")


class PlateauWithWarmup:
    """``ReduceLROnPlateau`` after an optional per-step linear warmup.

    ``step`` is called after every optimizer step and only drives the
    warmup: LR of each group is ``group_lr * (step + 1) / warmup_steps``
    until ``warmup_steps`` steps are done, then exactly ``group_lr``, and the
    method stops touching the LR. ``epoch_step(metric)`` is called once per
    epoch with the best.pt selection metric; it is ignored while the warmup
    is unfinished and when the metric is NaN, otherwise it calls
    ``ReduceLROnPlateau(mode="max", factor=0.5, patience=1,
    min_lr=5e-6).step(metric)``. The two therefore never set the LR in the
    same phase.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.num_steps = 0
        self.plateau = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, min_lr=COSINE_ETA_MIN
        )
        self._set_warmup_lr()

    def _set_warmup_lr(self) -> None:
        if self.warmup_steps <= 0:
            return
        factor = min(1.0, (self.num_steps + 1) / self.warmup_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        if self.num_steps >= self.warmup_steps:
            return
        self.num_steps += 1
        self._set_warmup_lr()

    def epoch_step(self, metric: float) -> None:
        if self.num_steps < self.warmup_steps or math.isnan(metric):
            return
        self.plateau.step(metric)


def build_schedulers(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epoch_num: int,
    steps_per_epoch: int,
    warmup_ratio: float = 0.0,
    decay_ratio: Optional[float] = None,
) -> tuple[Optional[Any], Optional[Any]]:
    """Return ``(epoch_scheduler, step_scheduler)``.

    The caller passes ``epoch_scheduler`` to ``step_epoch_scheduler`` once
    per epoch after validation and calls ``step_scheduler.step()`` after
    every optimizer step. Total steps are ``epoch_num * steps_per_epoch``,
    warmup steps ``round(warmup_ratio * total)``, wsd decay steps
    ``round(decay_ratio * total)``. Every group is scaled from its own
    starting LR, so ``--head_lr_mult`` holds in every schedule.

    * ``epoch_cosine``: the old schedule, ``CosineAnnealingLR(T_max=epoch_num,
      eta_min=5e-6)`` stepped per epoch. No step scheduler.
    * ``step_cosine``, ``step_linear``, ``constant``, ``wsd``: ``LambdaLR``
      with ``lr_factor`` per group, stepped per optimizer step.
    * ``onecycle``: ``OneCycleLR(max_lr=[group lrs], total_steps,
      pct_start=0.3, three_phase=False, cycle_momentum=False)`` stepped per
      optimizer step, as the MPRA-MNIST CNN trainers configure it. Its own
      defaults apply: it starts at ``max_lr / 25`` and anneals (cosine) to
      ``max_lr / 25e4``; the 5e-6 floor does not apply.
    * ``plateau``: ``PlateauWithWarmup``; the same object is the epoch
      scheduler and, when there is a warmup, the step scheduler.
    """
    if scheduler_name == "epoch_cosine":
        print(f"  scheduler epoch_cosine: T_max={epoch_num} epochs, eta_min={COSINE_ETA_MIN:g}")
        return CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=COSINE_ETA_MIN), None
    if scheduler_name not in SCHEDULER_CHOICES:
        raise ValueError(f"scheduler must be one of {SCHEDULER_CHOICES}, got {scheduler_name!r}.")

    total_steps = max(1, epoch_num * steps_per_epoch)
    warmup_steps = int(round(warmup_ratio * total_steps))
    decay_steps = int(round((decay_ratio or 0.0) * total_steps)) if scheduler_name == "wsd" else 0
    print(
        f"  scheduler {scheduler_name}: {total_steps} optimizer steps "
        f"({steps_per_epoch} per epoch), {warmup_steps} warmup steps"
        + (f", {decay_steps} decay steps" if scheduler_name == "wsd" else "")
    )

    if scheduler_name == "onecycle":
        return None, OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=0.3,
            three_phase=False,
            cycle_momentum=False,
        )
    if scheduler_name == "plateau":
        plateau = PlateauWithWarmup(optimizer, warmup_steps)
        return plateau, (plateau if warmup_steps > 0 else None)

    def make_lambda(base_lr: float) -> Callable[[int], float]:
        floor = min(1.0, COSINE_ETA_MIN / base_lr) if base_lr > 0 else 0.0

        def factor(step: int) -> float:
            return lr_factor(scheduler_name, step, total_steps, warmup_steps, decay_steps, floor)

        return factor

    lambdas = [make_lambda(group["lr"]) for group in optimizer.param_groups]
    return None, LambdaLR(optimizer, lr_lambda=lambdas)


def step_epoch_scheduler(scheduler: Optional[Any], metric: float) -> None:
    """Per-epoch scheduler step after validation; ``metric`` is the best.pt score.

    Only ``plateau`` reads the metric (and skips NaN). ``epoch_cosine`` steps
    unconditionally, as before. A None scheduler (step-level schedules) is a no-op.
    """
    if scheduler is None:
        return
    if isinstance(scheduler, PlateauWithWarmup):
        scheduler.epoch_step(metric)
    else:
        scheduler.step()


class ModelEMA:
    """Exponential moving average of every trainable parameter.

    ``update`` is called after every optimizer step with decay
    ``d_t = min(ema_decay, (1 + t) / (10 + t))``, ``t`` = updates so far, so
    the average follows the live weights closely during the first steps.
    The shadow copy is fp32 on the parameters' device. Buffers (none of these
    backbones keeps running statistics) are not averaged.

    Memory on the training GPU: the shadow costs 4 bytes per trainable
    parameter for the whole run, and ``swapped_in`` holds a second, temporary
    copy of the live weights (4 bytes per parameter; the backbones are kept
    in fp32 and only autocast to bfloat16) while validation runs. For N
    parameters that is 4N bytes persistent plus 4N bytes during validation,
    e.g. NTv3 650M: about 2.6 GB plus 2.6 GB; a 100M model: 0.4 GB plus
    0.4 GB. Compare with AdamW, which already keeps 8N bytes of state.

    ``state_dict_for`` returns the model's state dict with the EMA tensors in
    place of the parameters, so ``best.pt`` loads with the usual
    ``model.load_state_dict`` and the test path needs no EMA logic.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            name: param.detach().float().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._name_by_id = {
            id(param): name for name, param in model.named_parameters() if name in self.shadow
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        self.num_updates += 1
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.detach().float(), alpha=1.0 - decay)

    def state_dict_for(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """``model.state_dict()`` with every averaged parameter replaced by its EMA."""
        state = model.state_dict()
        for key, param in model.named_parameters(remove_duplicate=False):
            name = self._name_by_id.get(id(param))
            if name is not None and key in state:
                state[key] = self.shadow[name].to(dtype=state[key].dtype)
        return state

    @contextlib.contextmanager
    def swapped_in(self, model: nn.Module):
        """Load the EMA weights into ``model`` for the block, then restore the live ones."""
        live = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    live[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in live:
                        param.copy_(live[name])


def ema_scope(ema: Optional[ModelEMA], model: nn.Module):
    """``ema.swapped_in(model)`` when EMA is on, otherwise a no-op context."""
    return ema.swapped_in(model) if ema is not None else contextlib.nullcontext()


def checkpoint_state(model: nn.Module, ema: Optional[ModelEMA]) -> dict[str, torch.Tensor]:
    """State dict written to ``best.pt``: the EMA weights when EMA is on."""
    return model.state_dict() if ema is None else ema.state_dict_for(model)


def make_step_hook(
    step_scheduler: Optional[Any],
    ema: Optional[ModelEMA],
    model: nn.Module,
) -> Optional[Callable[[], None]]:
    """Callback for ``train_one_epoch`` after each optimizer step; None when unused."""
    if step_scheduler is None and ema is None:
        return None

    def on_step() -> None:
        if step_scheduler is not None:
            step_scheduler.step()
        if ema is not None:
            ema.update(model)

    return on_step


def split_rc_batch(
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], Optional[dict[str, torch.Tensor]]]:
    """Split a collate batch into forward inputs and reverse-complement inputs.

    The collate adds ``rc_input_ids`` (and ``rc_attention_mask``) only for an
    evaluation loader built with ``add_rc=True``. The reverse-complement
    dict reuses every other key, including ``labels``, so both forwards
    score the same rows in the same order. Returns ``(batch, None)`` when
    there are no ``rc_`` keys.
    """
    forward = {key: value for key, value in batch.items() if not key.startswith("rc_")}
    if "rc_input_ids" not in batch:
        return forward, None
    reverse = dict(forward)
    reverse.update({key[len("rc_"):]: value for key, value in batch.items() if key.startswith("rc_")})
    return forward, reverse


# =============================================================================
# Train / validate / test
# =============================================================================
# train_one_epoch updates weights and reports the training-mode metrics.
# evaluate_loader does not update weights. Call evaluate_loader on the
# validation split at the end of each epoch, and on the test split only after
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
    on_step: Optional[Callable[[], None]] = None,
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
        if on_step is not None:
            on_step()
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
    Use this for the validation split at the end of every epoch and,
    after the best checkpoint has been loaded, for the test split. An empty
    loader returns NaN rather than raising, so a bad split is visible in
    the TSV.
    """
    model.eval()
    running = 0.0
    steps = 0
    preds, labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _move_batch(batch, device)
        forward_batch, rc_batch = split_rc_batch(batch)
        with _amp_ctx(spec, device):
            outputs = model(**forward_batch)
            loss = outputs["loss"].mean()
            logits = outputs["logits"].float()
            if rc_batch is not None:
                # --rc_tta: same rows, reverse-complement strand; average both.
                rc_outputs = model(**rc_batch)
                loss = (loss + rc_outputs["loss"].mean()) / 2.0
                logits = (logits + rc_outputs["logits"].float()) / 2.0
        running += loss.item()
        steps += 1
        preds.append(logits.cpu())
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
    """Pearson r of each promoter-activity column, then the mean of the two.

    ``preds`` and ``labels`` are ``[n_sequences, 2]`` in ``CELL_TYPES``
    order. Columns are not pooled into one correlation: Dev_log2 is never
    mixed with Hk_log2. The mean, stored as ``avg_{split}_pearson``, is
    NaN if either column is NaN.
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
# Train / val / test split, both promoter activities
# =============================================================================
# main() calls run_split once. Training is --epoch_num epochs (default 10).
# There is no k-fold.


def _dataset(
    split: str,
    transform: t.Compose,
    root: str,
    use_original_reverse_complement: bool,
) -> deAlmeidaDataset:
    """One deAlmeida split for Dev_log2 and Hk_log2.

    ``use_original_reverse_complement`` is passed through only for
    ``split='train'``. Val and test always receive false, so those tables
    are not doubled. ``root`` is the data directory; the dataset class
    finds its table there.
    """
    augment = use_original_reverse_complement and split == "train"
    return deAlmeidaDataset(
        split=split,
        cell_type=list(CELL_TYPES),
        use_original_reverse_complement=augment,
        transform=transform,
        root=root,
    )


def _score_fields(prefix: str, scores: Optional[PearsonScores]) -> dict[str, float]:
    """TSV cells for one split: one Pearson per promoter activity, then ``avg_`` mean."""
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


def set_run_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for one run.

    ``ReverseComplement`` draws ``torch.rand``, and the training
    DataLoader is given a generator created after this call, so both
    follow ``seed``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_seed_file(folder: str, seed: int) -> str:
    """Write ``seed`` as a single line to ``folder/seed.txt``."""
    path = os.path.join(folder, "seed.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{seed}\n")
    return path


def run_split(
    args: argparse.Namespace,
    spec: ModelSpec,
    transforms: dict[str, t.Compose],
    device: torch.device,
) -> PearsonScores:
    """Train one model on split=train and test on split=test.

    Validation is the first half of chromosome 2R. It is never used as the
    test set, and the test split is never used to choose the checkpoint.
    Both promoter activities are trained together.

    After each epoch the validation loader is scored and one row is
    appended to ``epochs.tsv``. The state dict is saved only when ``avg_val_pearson``
    strictly improves. After the last epoch that file is loaded and the
    test split is scored; the test metrics go to
    ``test.tsv``. The weights sitting in memory after the last
    epoch are not the test weights unless that epoch happened to be the best.

    Returns the test scores of the loaded checkpoint. The mean is NaN if
    no finite validation Pearson was observed and therefore no file was
    written.
    """
    cell_label = ",".join(CELL_TYPES)

    tokenizer, model = build_tokenizer_and_model(spec, args.backbone_dropout)
    model = model.to(device)
    collate_fn = make_collate_fn(tokenizer, spec)
    eval_collate_fn = make_collate_fn(tokenizer, spec, add_rc=args.rc_tta)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.run_seed)
    train_loader = DataLoader(
        _dataset("train", transforms["train"], args.root, args.use_original_reverse_complement),
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        _dataset("val", transforms["eval"], args.root, args.use_original_reverse_complement),
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=eval_collate_fn,
    )
    test_loader = DataLoader(
        _dataset("test", transforms["eval"], args.root, args.use_original_reverse_complement),
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=eval_collate_fn,
    )

    # Caduceus source runs use AdamW eps 1e-6; the other backbones use 1e-8.
    # The cosine schedule decays the learning rate across this run's epochs
    # and is stepped once per epoch, after validation, so epoch 1 still sees
    # the peak learning rate.
    # With the default fine-tuning flags build_optimizer and build_schedulers
    # return exactly this AdamW over model.parameters() and the per-epoch
    # CosineAnnealingLR; see the "Opt-in fine-tuning options" section.
    optimizer = build_optimizer(
        model,
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        eps=1e-6 if args.model == "caduceus" else 1e-8,
        head_lr_mult=args.head_lr_mult,
        no_decay_bias_norm=args.no_decay_bias_norm,
    )
    scheduler, step_scheduler = build_schedulers(
        optimizer,
        args.scheduler,
        args.epoch_num,
        len(train_loader),
        args.warmup_ratio,
        args.decay_ratio,
    )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay is not None else None
    on_step = make_step_hook(step_scheduler, ema, model)

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
        train_loss, train_scores = train_one_epoch(
            model, train_loader, optimizer, spec, device, on_step=on_step
        )
        # With --ema_decay, validation (and therefore best.pt selection)
        # runs on the EMA weights; the live weights are restored after.
        with ema_scope(ema, model):
            val_loss, val_scores = evaluate_loader(model, val_loader, spec, device)
        print(
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}\n"
            f"  train {train_scores.format()}\n"
            f"  val   {val_scores.format()}"
        )
        append_row(
            os.path.join(args.result_dir, "epochs.tsv"),
            {
                "model": args.model,
                "cell_types": cell_label,
                "epoch": epoch_num,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **_score_fields("train", train_scores),
                **_score_fields("val", val_scores),
            },
            EPOCH_COLUMNS,
        )

        # Selection metric is avg_val_pearson only, not validation loss
        # and not avg_train_pearson. A NaN on any cell makes the average
        # NaN and must not overwrite the file.
        if not np.isnan(val_scores.mean) and val_scores.mean > best_val:
            best_val = val_scores.mean
            best_epoch = epoch_num
            torch.save(checkpoint_state(model, ema), ckpt_path)
            print(f"new best avg_val_pearson={best_val:.4f} -> {ckpt_path}")

        # plateau reads the selection score; epoch_cosine just steps.
        step_epoch_scheduler(scheduler, val_scores.mean)

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
        append_row(
            os.path.join(args.result_dir, "test.tsv"),
            {
                "model": args.model,
                "cell_types": cell_label,
                "best_epoch": best_epoch,
                "test_loss": test_loss,
                **_score_fields("test", test_scores),
            },
            TEST_COLUMNS,
        )

    del model, optimizer, scheduler, step_scheduler, ema, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_scores


# =============================================================================
# Main
# =============================================================================


def ensure_results_file(path: str, columns: list[str]) -> None:
    """Create ``path`` with ``columns``, or accept an existing file with that header.

    A header mismatch means an older schema. Pass a new ``--result_dir``
    instead of appending incompatible rows.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=columns).to_csv(path, sep="\t", index=False)
        return
    existing = pd.read_csv(path, sep="\t", nrows=0)
    if list(existing.columns) != columns:
        raise SystemExit(
            f"{path} has columns {list(existing.columns)}, expected {columns}. "
            "Pass a new --result_dir."
        )


def append_row(path: str, row: dict[str, Any], columns: list[str]) -> None:
    """Append one metrics row.

    ``epochs.tsv`` is rewritten in epoch order after each append, so a
    killed run keeps the epochs already finished. ``test.tsv`` only
    grows by one row per finished test.
    """
    ensure_results_file(path, columns)
    pd.DataFrame([row], columns=columns).to_csv(
        path, sep="\t", index=False, mode="a", header=False
    )
    if "epoch" in columns:
        frame = pd.read_csv(path, sep="\t")
        if not frame.empty:
            frame = frame.assign(_epoch=pd.to_numeric(frame["epoch"], errors="coerce"))
            frame = frame.sort_values(by=["_epoch"], ascending=True, kind="mergesort")
            frame = frame.drop(columns=["_epoch"])
            frame.to_csv(path, sep="\t", index=False)


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
    """Train Dev_log2 and Hk_log2 on the deAlmeida train/val/test split.

    Order of work: resolve the model spec, then ``run_split`` once per
    entry of ``--runs``. Each run writes ``run{k}/`` under
    ``--result_dir`` (and ``run{k}/best.pt`` under ``--model_dir``).
    """
    args = parse_args()
    spec = resolve_spec(args)
    args.backbone_dropout = resolve_backbone_dropout(args.model, args.backbone_dropout)
    apply_runtime_quirks(spec)
    result_parent = args.result_dir
    model_parent = args.model_dir

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA is not available; running on CPU (this will be very slow).")
    else:
        device = torch.device(f"cuda:{args.device}")

    transforms = build_string_transforms(
        spec.rc_train_prob, args.use_original_reverse_complement
    )

    print("=" * 60)
    print("LAUNCH PARAMETERS")
    print("=" * 60)
    print(f"  model          : {args.model}")
    print(f"  hub_id         : {spec.hub_id}")
    print(f"  outputs        : {NUM_OUTPUTS} ({', '.join(CELL_TYPES)})")
    print("  split          : train / val (2R first half) / test (2R second half)")
    print("  labels         : Dev_log2, Hk_log2")
    print(f"  original RC    : {args.use_original_reverse_complement}")
    print(f"  batch_size     : {spec.batch_size}")
    print(f"  runs           : {args.runs}")
    print(f"  seed           : {args.seed} (run k uses {args.seed} + k - 1)")
    print(f"  epoch_num      : {args.epoch_num}")
    print(f"  lr             : {spec.lr}")
    print(f"  weight_decay   : {spec.weight_decay}")
    print(f"  max_length     : {spec.max_length}")
    print(f"  head_dropout   : {spec.head_dropout}")
    print(f"  scheduler      : {args.scheduler}")
    print(f"  warmup_ratio   : {args.warmup_ratio}")
    print(f"  decay_ratio    : {args.decay_ratio if args.scheduler == 'wsd' else '-'}")
    print("  (optimizer steps, warmup and decay steps are printed when the scheduler is built)")
    print(f"  head_lr_mult   : {args.head_lr_mult}")
    print(f"  no_decay_bn    : {args.no_decay_bias_norm}")
    print(
        "  backbone_drop  : "
        f"{'checkpoint config' if args.backbone_dropout is None else args.backbone_dropout}"
    )
    print(f"  ema_decay      : {'off' if args.ema_decay is None else args.ema_decay}")
    print(f"  rc_tta         : {args.rc_tta}{' (val and test)' if args.rc_tta else ''}")
    print(f"  device         : {device}")
    print(f"  root           : {args.root}")
    print(f"  result_dir     : {result_parent}")
    print(f"  model_dir      : {model_parent}")
    print("=" * 60)

    for run_idx in range(1, args.runs + 1):
        args.result_dir = os.path.join(result_parent, f"run{run_idx}")
        args.model_dir = os.path.join(model_parent, f"run{run_idx}")
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)
        run_seed = args.seed + run_idx - 1
        args.run_seed = run_seed
        seed_path = write_seed_file(args.result_dir, run_seed)
        set_run_seed(run_seed)
        print(
            f"\n### run {run_idx}/{args.runs} — train/val/test — model: {args.model} — "
            f"targets: {', '.join(CELL_TYPES)} ###"
        )
        print(f"  folder: {args.result_dir}")
        print(f"  seed: {run_seed} -> {seed_path}")
        test_scores = run_split(args, spec, transforms, device)

        print("\n" + "=" * 60)
        for name in ("epochs.tsv", "test.tsv"):
            path = os.path.join(args.result_dir, name)
            if os.path.exists(path):
                print("\n" + path)
                print(pd.read_csv(path, sep="\t").to_string(index=False))
        print("=" * 60)
        print(
            f"run {run_idx} avg_test_pearson: "
            f"{None if np.isnan(test_scores.mean) else round(test_scores.mean, 4)}"
        )
        for index, cell in enumerate(CELL_TYPES):
            value = test_scores.per_cell[index]
            printed = None if np.isnan(value) else round(float(value), 4)
            print(f"run {run_idx} test Pearson {cell}: {printed}")
        print(f"checkpoint: {os.path.join(args.model_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
