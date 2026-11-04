"""
Fine-tune one Hugging Face DNA model on Rafi2024.

Flags
-----
--model          backbone; default moderngena-large
--data_type      test subsets; default all nine
--device         CUDA index; default 0
--num_workers    DataLoader workers; default 0
--root           data directory; default <repo>/data
--result_dir     results folder; default ./results/{model}
--model_dir      checkpoint folder; default the same results folder
--lr             learning-rate override
--wd             weight-decay override
--batch_size     batch size; default 256
--runs           independent training runs; default 5
--seed           seed of run 1; run k uses seed+k-1; default 0
--epoch_num      epochs per run; default 10
--eval_only      no training: score run{k}/best.pt, rewrite run{k}/test.tsv
--checkpoint     score this one best.pt (implies --eval_only)
--max_length     tokenizer length override
--head_dropout   head dropout for Caduceus and DNABERT-2
--scheduler      epoch_cosine (default), step_cosine, step_linear,
                 onecycle, wsd, constant, plateau
--warmup_ratio   linear warmup fraction (not epoch_cosine/onecycle); default 0
--decay_ratio    wsd final decay fraction; default 0.2 with wsd
--head_lr_mult   pool/head LR = lr * this; default 1
--no_decay_bias_norm no weight decay on biases and norm weights
--backbone_dropout ModernGENA backbone dropout; default checkpoint
--ema_decay      EMA weights for val, best.pt, test; default off
--rc_tta         average val over forward and RC; default off

The description continues below.

This file must stay at ``MPRA-MNIST/mpramnist/Rafi2024/`` and must be
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
    (Python 3.10 or newer) and trains the 18-bin head defined below.
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

One launch, one yeast promoter task
-----------------------------------
Rafi2024 is the Random Promoter DREAM Challenge: 80 bp random promoters
measured in S. cerevisiae. ``--root`` is the data directory (default
``<repo>/data``). The dataset class locates ``Dream_train.tsv``,
``Dream_single.tsv``, and ``Dream_paired.tsv`` under that directory.

    train    split="train", about 6.7 million sequences. The label is the
             FACS expression bin, 0..17.
    val      split="val" (the public leaderboard set), data_type="all".
    test     split="test" (the private set), every subset in --data_type.

Training labels are FACS bins; validation and test labels are MAUDE
estimates on another scale. Pearson does not depend on the scale, so the
expected bin is correlated with those labels directly.

18-bin head (as in ``Rafi_model_launch.py``)
--------------------------------------------
The pool reduces the token states to one vector. The head is dropout
(Caduceus and DNABERT-2 only) followed by ``nn.Linear(hidden_size, 18)``,
one logit per FACS bin. Training targets come from
``target_transforms.SoftBinTarget()``: a normal with mean ``label + 0.5``
and scale 0.5, integrated over the bin edges. The loss is
``KLDivLoss(reduction="batchmean")`` between ``log_softmax(logits)`` and
those soft bins, the same loss as ``LitModel_Rafi``. The prediction is
the expected bin, ``sum_k softmax(logits)_k * k``. Train Pearson,
validation Pearson, and test Pearson all use that expected bin.

The authors' CNNs also receive an ``is_singleton`` channel and a reverse
strand channel. Those are extra one-hot channels; a tokenizer has no
place for them, so these token models see only the DNA string.

Sequences
---------
Each insert is placed in its plasmid context: the left flank is the
120 bp of ``RafiDataset.PLASMID`` before the 80-N insert site, the right
flank is ``RafiDataset.RIGHT_FLANK``, and ``LeftCrop(120, 120)`` keeps
the last 120 bp. Training then reverse-complements with probability 0.5
(0 for Caduceus, whose pool already averages the reverse complement).
Validation uses the forward strand only. Sequences stay strings.
``Seq2Tensor`` is not used.

Runs, 10 epochs
---------------
One launch repeats training ``--runs`` times (default 5), as in the
Fromel2025 launcher. Each run has its own folder and seed, and trains
for ``--epoch_num`` epochs (default 10). The datasets are loaded once
and shared by the runs:

    for epoch in 1..10:
        train, then validate on val "all"
        append one row to epochs.tsv
        if val_pearson is the best so far, write best.pt
    after the last epoch:
        load best.pt
        score every test subset with those weights
        write one row per subset to test.tsv

Test
----
Every subset is predicted twice, on the forward strand and on the
reverse complement, and the two predictions are averaged, as in
``Rafi_model_launch.py``. Single subsets (``all``, ``high``, ``low``,
``yeast``, ``random``, ``challenging``) correlate that average with the
measured expression. Paired subsets (``snv``, ``perturbation``,
``tiling``) predict the reference and the alternative sequence and
correlate ``alt - ref`` with ``delta_measured``. Each row of
``test.tsv`` stores Pearson r and r^2 of one subset; the authors
report r^2.

Test only (``--eval_only``)
----------------------------
``--eval_only`` skips training and validation and does not load the
6.7 million training sequences. For run k = 1..``--runs`` it builds the
``--model`` architecture, loads ``run{k}/best.pt`` from ``--model_dir``
(default ``--result_dir``), scores the subsets in ``--data_type`` exactly
as the end of training does (forward and reverse-complement averaged,
``alt - ref`` for paired subsets), and rewrites ``run{k}/test.tsv``.
``epochs.tsv``, ``seed.txt`` and ``best.pt`` are not touched; a run
without ``best.pt`` is skipped. ``--checkpoint PATH`` scores one file
instead (for example a ``best.pt`` from before the run folders existed)
and writes ``test.tsv`` to ``--result_dir``, by default the checkpoint's
own folder. ``--model`` must be the one the checkpoint was trained with;
training-only flags (lr, scheduler, EMA, ...) are ignored.

Example::

    python Rafi_hf_model_launch.py --model caduceus --eval_only
    python Rafi_hf_model_launch.py --model caduceus --checkpoint results/caduceus/best.pt

Outputs, in the working directory
----------------------------------
    ./results/{model}/run1/
    ./results/{model}/run2/
        One folder per run, ``run1`` through ``run{--runs}``, as in the
        Fromel2025 launcher. Inside each folder:
            epochs.tsv
                One row per epoch: training KL loss, train Pearson, and
                validation Pearson and r^2.
            test.tsv
                Pearson r and r^2 of best.pt, one row per test subset in
                the order all, high, low, yeast, random, challenging, snv,
                perturbation, tiling. The header is ``data_type``,
                ``pearson_r``, ``pearson_r2``::

                    data_type         pearson_r  pearson_r2
                    all               0.97       0.94
                    ...
                    tiling            0.92       0.85

                The file is rewritten after every subset, so a relaunch
                into the same folder replaces it.
            best.pt
                ``state_dict`` of the epoch with the highest
                ``val_pearson`` in that run. ``test.tsv`` is scored with
                this file.
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
    Off by default. Validation only: the val ``all`` prediction is the mean
    of the forward string and its reverse complement
    (``mpramnist.transforms.reverse_complement`` applied in the collate to
    the flanked and cropped string, same batch, same rows and order). The
    checkpoint is then chosen by that strand-averaged ``val_pearson``, so a
    run with ``--rc_tta`` may select a different epoch than one without. The
    test already averages the forward and reverse-complement passes and is
    unchanged. Training is unchanged.

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
``--data_type`` selects the test subsets (default all nine).

Leaving ``--batch_size``, ``--lr``, ``--wd``, ``--max_length``, and
``--head_dropout`` unset uses the values in ``MODEL_SPECS``. The shared
batch size is 256. Learning rate is 1e-4 except Caduceus (3e-4). Weight
decay is 0.02 except Caduceus (0.05). Tokenizer length is 128 except
Caduceus (120, the crop length). ``--head_dropout`` sets dropout
on the head for Caduceus (default 0.3, applied in the pool and again
before the linear map) and DNABERT-2 (default 0.1, applied only before
the linear map). It is ignored for ModernGENA and NTv3. NTv3 pads each
batch to a multiple of 128. ``--epoch_num`` defaults to 10. ``--device``
is the CUDA index (default 0). ``--root`` defaults to
``<repository>/data``.

Example::

    cd /path/to/MPRA-MNIST/mpramnist/Rafi2024
    python Rafi_hf_model_launch.py --model caduceus
    python Rafi_hf_model_launch.py --model ntv3 --data_type all snv

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

        <repo>/mpramnist/Rafi2024/Rafi_hf_model_launch.py

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
        "This script must stay at MPRA-MNIST/mpramnist/Rafi2024/"
        "Rafi_hf_model_launch.py.\n"
        f"This file is: {os.path.abspath(__file__)}\n"
        f"Expected package init: {package_init}"
    )


# Insert the repository root before importing mpramnist. The script's own
# directory is already on sys.path, but that directory is the Rafi2024
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
import mpramnist.target_transforms as t_t
from mpramnist.Rafi2024.dataset import RafiDataset


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

# The insert is cropped to SEQ_LEN after the plasmid flanks are added, as
# in Rafi_model_launch.py. The head has one logit per FACS bin.
SEQ_LEN = 120
N_BINS = 18
VAL_DATA_TYPE = "all"
SINGLE_DATA_TYPES = ["all", "high", "low", "yeast", "random", "challenging"]
PAIRED_DATA_TYPES = ["snv", "perturbation", "tiling"]
TEST_DATA_TYPES = SINGLE_DATA_TYPES + PAIRED_DATA_TYPES

# epochs.tsv and test.tsv (one row per test subset). Pearson is the
# correlation of the expected bin; *_r2 is its square, the number the
# Rafi2024 README reports. The checkpoint uses val_pearson.
EPOCH_COLUMNS = [
    "model",
    "val_data_type",
    "epoch",
    "train_loss",
    "train_pearson",
    "val_pearson",
    "val_pearson_r2",
]
TEST_COLUMNS = ["pearson_r", "pearson_r2"]


def kind_of(data_type: str) -> str:
    """``paired`` for snv, perturbation, and tiling; ``single`` otherwise."""
    return "paired" if data_type in PAIRED_DATA_TYPES else "single"


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
    # Tokenizer truncation length. Rafi sequences are cropped to 120 bp.
    # Caduceus uses 120, the crop length. The other tokenizers use 128
    # so a 120 bp string plus the usual special tokens is not truncated.
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
    # True only for DNABERT-2. TokenBinModel then applies
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
        max_length=128,
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
        max_length=128,
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
        max_length=128,
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
        max_length=120,
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
        max_length=128,
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
# Every launch trains one model on the full training set and scores the
# test subsets in --data_type. Result and checkpoint paths include the
# model, so two backbones do not share a file. A second launch that
# appends to an existing TSV continues that file.


def parse_args() -> argparse.Namespace:
    """Parse the launch command.

    ``--data_type`` is any subset of ``TEST_DATA_TYPES``. Hyperparameters
    left as None are filled from ``MODEL_SPECS`` in ``resolve_spec``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a Hugging Face DNA model on Rafi2024 (DREAM yeast "
            "promoters) with an 18-bin head, then score every test subset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("general", "General run settings")
    general.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help=(
            "Results folder. Each run writes run{k}/epochs.tsv, run{k}/test.tsv "
            "(one row per test subset), run{k}/seed.txt, and run{k}/best.pt. "
            "Default: ./results/{model}"
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
    general.add_argument(
        "--eval_only",
        action="store_true",
        help=(
            "Skip training: for each run k load run{k}/best.pt from --model_dir, "
            "score the test subsets, and rewrite run{k}/test.tsv. Train and val "
            "are not loaded. --model must match the checkpoint."
        ),
    )
    general.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Score this one best.pt instead of the run folders (implies "
            "--eval_only); test.tsv goes to --result_dir, default the "
            "checkpoint's own folder."
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
        "--data_type",
        nargs="+",
        default=list(TEST_DATA_TYPES),
        choices=TEST_DATA_TYPES,
        help=(
            "Test subsets. Single: all, high, low, yeast, random, challenging. "
            "Paired (alt - ref): snv, perturbation, tiling."
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
            "Dropout on the head for Caduceus (default 0.3) and "
            "DNABERT-2 (default 0.1). Ignored for ModernGENA and NTv3."
        ),
    )

    add_finetune_args(parser)

    args = parser.parse_args()
    check_finetune_args(parser, args)
    # Keep the canonical order and drop repeats, so the rows of test.tsv
    # always come out in the same order.
    args.data_type = [name for name in TEST_DATA_TYPES if name in set(args.data_type)]
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise SystemExit(f"--checkpoint {args.checkpoint} does not exist.")
        args.eval_only = True
        if args.result_dir is None:
            args.result_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    if args.result_dir is None:
        args.result_dir = os.path.join("results", args.model)
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
# to one vector. The head is a linear map from that vector to N_BINS logits,
# one per FACS bin.
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
# 18-bin wrappers
# =============================================================================
# Both wrappers end in Linear(hidden, N_BINS) and BinExpectation. The loss
# is KL divergence to the SoftBinTarget probabilities, the same as
# LitModel_Rafi's clf_loss. Only training batches carry those
# probabilities; validation and test batches return the expected bin and
# no loss.


class BinExpectation(nn.Module):
    """Bin logits -> log-probabilities and expected bin.

    The same computation as ``AutosomeFinalLayersBlock`` in
    ``Rafi_model_launch.py``: ``log_softmax`` for the KL loss and
    ``sum_k softmax_k * k`` as the scalar prediction. Logits are upcast to
    fp32 first so the softmax is not computed in bfloat16.
    """

    def __init__(self, n_bins: int = N_BINS):
        super().__init__()
        self.register_buffer("bins", torch.arange(n_bins, dtype=torch.float32), persistent=False)

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        score = (log_probs.exp() * self.bins).sum(dim=-1)
        return log_probs, score


def bin_kl_loss(log_probs: torch.Tensor, probs: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """``KLDivLoss(reduction="batchmean")`` against soft bins, or None without targets."""
    if probs is None:
        return None
    return F.kl_div(log_probs, probs.float(), reduction="batchmean")


class TokenBinModel(nn.Module):
    """Masked-pool 18-bin wrapper for ModernGENA, NTv3, and DNABERT-2.

    Use ``use_hidden_states=True`` only for NTv3, whose forward returns
    ``hidden_states`` rather than ``last_hidden_state``. After the masked
    attention pool, ``nn.Linear(hidden_size, 18)`` writes one logit per
    FACS bin. ``score`` is the expected bin, shaped ``[batch]``.

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
        self.bin_head = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(hidden_size, N_BINS),
        ).float()
        self.expectation = BinExpectation(N_BINS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
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
        log_probs, score = self.expectation(self.bin_head(pooled))
        return {"loss": bin_kl_loss(log_probs, probs), "log_probs": log_probs, "score": score}


class CaduceusBinModel(nn.Module):
    """Caduceus-PS wrapper: RC-invariant pool, dropout, 18-bin head.

    Do not pass ``attention_mask``. Extra keyword arguments are ignored so
    the shared loop can call every wrapper the same way. ``hidden_size``
    must be 512 (``pooling_hidden_size``), because Caduceus-PS
    concatenates both directions. Dropout is applied inside the pool and
    again on the pooled vector; both are inactive after ``model.eval()``.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int = 512, head_dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.pool = CaduceusAttentionPooling(hidden_size, dropout=head_dropout)
        # Dropout on the pooled vector is a no-op after model.eval(), so the
        # checkpoint selected by validation Pearson is the one used at test.
        self.bin_head = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(hidden_size, N_BINS),
        ).float()
        self.expectation = BinExpectation(N_BINS)

    def forward(
        self,
        input_ids: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        **_unused: Any,
    ) -> dict[str, Optional[torch.Tensor]]:
        outputs = self.backbone(input_ids=input_ids, return_dict=True)
        pooled = self.pool(outputs.last_hidden_state)
        log_probs, score = self.expectation(self.bin_head(pooled))
        return {"loss": bin_kl_loss(log_probs, probs), "log_probs": log_probs, "score": score}


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
       ``TokenBinModel.forward``.

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
    """Download one backbone and wrap it with the matching 18-bin head.

    The wrapper is ``CaduceusBinModel`` when ``pooling_hidden_size``
    is set, otherwise ``TokenBinModel``. Both end in
    ``Linear(hidden, 18)``.

    Called once per launch so training starts from the published
    checkpoint. The branch
    is selected by ``ModelSpec``, not by the CLI name:

    * ``load_as_masked_lm`` — NTv3. Drop the LM head and read the last
      hidden state.
    * ``pooling_hidden_size is not None`` — Caduceus-PS. Turn the KV cache
      off and use the 512-d RC-concat width.
    * otherwise — ``AutoModel`` plus ``TokenBinModel``. ModernGENA
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
        model = TokenBinModel(backbone, hidden_size=hidden, use_hidden_states=True)
        return tokenizer, model

    if spec.pooling_hidden_size is not None:
        # Caduceus-PS is not used for generation here. use_cache=False avoids
        # storing a KV cache on every forward. The pool width is 512.
        config = AutoConfig.from_pretrained(spec.hub_id, **load_kw)
        config.use_cache = False
        backbone = AutoModel.from_pretrained(spec.hub_id, config=config, **load_kw)
        model = CaduceusBinModel(
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
    model = TokenBinModel(
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

    ``RafiDataset`` yields ``(sequence, target)``. For paired subsets the
    sequence is a dict with ``seq`` (reference) and ``seq_alt``; the
    reference is tokenized into ``input_ids`` and the alternative into
    ``alt_input_ids``. On the training split the target is the
    ``SoftBinTarget`` dict: ``value`` goes to ``labels`` and ``probs``
    (``[batch, 18]``) to ``probs``. Otherwise the target is one number
    and only ``labels`` is set. Sequences are padded and truncated to
    ``spec.max_length``. The attention mask is included only when
    ``spec.uses_attention_mask`` is true, because Caduceus-PS errors if
    it receives one. Keep ``num_workers`` at 0 unless you have checked
    that this tokenizer is safe to call from worker processes.

    ``add_rc=True`` is for evaluation loaders under ``--rc_tta``: the batch
    also gets ``rc_input_ids`` (and ``rc_attention_mask``), the tokens of
    ``mpramnist.transforms.reverse_complement`` of each string after the
    evaluation transform. ``split_rc_batch`` separates the two strands.
    """

    def as_string(sequence: Any) -> str:
        return "".join(sequence) if isinstance(sequence, list) else str(sequence)

    def encode(sequences: list[str], prefix: str, out: dict[str, torch.Tensor]) -> None:
        encoded = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=spec.max_length,
            pad_to_multiple_of=spec.pad_to_multiple_of,
            return_tensors="pt",
        )
        out[f"{prefix}input_ids"] = encoded["input_ids"]
        if spec.uses_attention_mask:
            if "attention_mask" in encoded:
                out[f"{prefix}attention_mask"] = encoded["attention_mask"]
            else:
                out[f"{prefix}attention_mask"] = torch.ones_like(encoded["input_ids"])

    def scalar(value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32).reshape(())

    def collate_fn(batch: list[tuple[Any, Any]]) -> dict[str, torch.Tensor]:
        sequences = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        out: dict[str, torch.Tensor] = {}
        if isinstance(sequences[0], dict):
            if add_rc:
                raise ValueError("add_rc is only for the single-sequence validation set.")
            encode([as_string(item["seq"]) for item in sequences], "", out)
            encode([as_string(item["seq_alt"]) for item in sequences], "alt_", out)
        else:
            encode([as_string(item) for item in sequences], "", out)
            if add_rc:
                encode([t.reverse_complement(as_string(item)) for item in sequences], "rc_", out)
        if isinstance(targets[0], dict):
            out["labels"] = torch.stack([scalar(target["value"]) for target in targets])
            out["probs"] = torch.stack(
                [torch.as_tensor(target["probs"], dtype=torch.float32).reshape(N_BINS) for target in targets]
            )
        else:
            out["labels"] = torch.stack([scalar(target) for target in targets])
        return out

    return collate_fn


def build_string_transforms(rc_train_prob: float) -> dict[str, t.Compose]:
    """Return the transforms of ``Rafi_model_launch.py``, left as strings.

    These backbones tokenize strings. ``Seq2Tensor`` is not applied, and
    the ``is_singleton`` and reverse-strand channels are not added: those
    are one-hot channels for the CNNs.

    Every transform adds the 120 bp of plasmid before the 80-N insert site
    on the left and ``RafiDataset.RIGHT_FLANK`` on the right, then keeps
    the last 120 bp. ``train`` reverse-complements with ``rc_train_prob``
    (0 for Caduceus, 0.5 otherwise). ``val`` and ``forward`` keep the
    forward strand. ``reverse`` always reverse-complements; test scores
    average ``forward`` and ``reverse``.
    """
    plasmid = RafiDataset.PLASMID.upper()
    insert_start = plasmid.find("N" * 80)
    left_flank = plasmid[insert_start - SEQ_LEN : insert_start]
    right_flank = RafiDataset.RIGHT_FLANK

    def compose(rc_prob: float) -> t.Compose:
        return t.Compose(
            [
                t.AddFlanks(left_flank, right_flank),
                t.LeftCrop(SEQ_LEN, SEQ_LEN),
                t.ReverseComplement(rc_prob),
            ]
        )

    return {
        "train": compose(rc_train_prob),
        "val": compose(0),
        "forward": compose(0),
        "reverse": compose(1),
    }


# Help text of --rc_tta, the one per-launcher line of the fine-tuning options.
RC_TTA_HELP = (
    "Average validation predictions over the forward string and its reverse "
    "complement; checkpoint selection then uses the averaged val_pearson. "
    "The test already averages both strands and is unchanged."
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
# predict_scores does not update weights. Call evaluate_val at the end of
# each epoch, and evaluate_test only after the best state dict has been
# loaded. Pearson for model selection is always the validation one.


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


def _forward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    prefix: str = "",
    probs: Optional[torch.Tensor] = None,
) -> dict[str, Optional[torch.Tensor]]:
    """Run the wrapper on the reference (``prefix=""``) or alternative (``"alt_"``) tokens."""
    inputs = {"input_ids": batch[f"{prefix}input_ids"]}
    if f"{prefix}attention_mask" in batch:
        inputs["attention_mask"] = batch[f"{prefix}attention_mask"]
    return model(**inputs, probs=probs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    spec: ModelSpec,
    device: torch.device,
    on_step: Optional[Callable[[], None]] = None,
) -> tuple[float, float]:
    """Train for one epoch.

    The model is set to train mode, so Caduceus dropout is on. Returns the
    mean KL loss over optimizer steps and the Pearson correlation between
    the expected bin and the training label of those same forwards. This
    Pearson is not comparable to validation Pearson: augmentation and
    dropout are active. Selection uses ``val_pearson``.
    """
    model.train()
    running = 0.0
    preds, labels = [], []
    bar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(bar):
        optimizer.zero_grad(set_to_none=True)
        batch = _move_batch(batch, device)
        with _amp_ctx(spec, device):
            outputs = _forward(model, batch, probs=batch["probs"])
            loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=spec.max_grad_norm)
        optimizer.step()
        if on_step is not None:
            on_step()
        running += loss.item()
        preds.append(outputs["score"].detach().float().cpu())
        labels.append(batch["labels"].detach().float().cpu())
        bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running / (step + 1):.4f}")
    mean_loss = running / max(len(loader), 1)
    if not preds:
        return mean_loss, float("nan")
    return mean_loss, pearson_from_arrays(torch.cat(preds).numpy(), torch.cat(labels).numpy())


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    loader: DataLoader,
    spec: ModelSpec,
    device: torch.device,
    paired: bool,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Expected bin of every sequence in ``loader``, in loader order.

    Dropout is off. Returns ``(ref, alt, labels)``. ``alt`` is None for a
    single subset. For a paired subset ``labels`` is ``delta_measured``.
    """
    model.eval()
    refs, alts, labels = [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _move_batch(batch, device)
        with _amp_ctx(spec, device):
            ref = _forward(model, batch)["score"].float()
            if "rc_input_ids" in batch:
                # --rc_tta (validation only): mean of forward and reverse complement.
                ref = (ref + _forward(model, batch, prefix="rc_")["score"].float()) / 2.0
            refs.append(ref.cpu())
            if paired:
                alts.append(_forward(model, batch, prefix="alt_")["score"].float().cpu())
        labels.append(batch["labels"].float().cpu())
    if not refs:
        empty = np.empty(0, dtype=np.float32)
        return empty, (empty if paired else None), empty
    ref = torch.cat(refs).numpy()
    alt = torch.cat(alts).numpy() if paired else None
    return ref, alt, torch.cat(labels).numpy()


def evaluate_val(
    model: nn.Module,
    loader: DataLoader,
    spec: ModelSpec,
    device: torch.device,
) -> float:
    """Pearson of the forward-strand expected bin on val ``all``."""
    ref, _, labels = predict_scores(model, loader, spec, device, paired=False)
    return pearson_from_arrays(ref, labels)


def evaluate_test(
    model: nn.Module,
    forward_loader: DataLoader,
    reverse_loader: DataLoader,
    spec: ModelSpec,
    device: torch.device,
    paired: bool,
) -> tuple[float, int]:
    """Score one test subset as ``Rafi_model_launch.py`` does.

    The forward-strand and reverse-complement predictions are averaged.
    For a paired subset the averaged ``alt - ref`` is correlated with
    ``delta_measured``; for a single subset the averaged prediction is
    correlated with the measured expression. Both loaders must iterate
    the same dataset rows in the same order (``shuffle=False``); the
    labels of the two passes are compared to check that. Returns
    ``(pearson, n_sequences)``.
    """
    fw_ref, fw_alt, labels = predict_scores(model, forward_loader, spec, device, paired)
    rc_ref, rc_alt, rc_labels = predict_scores(model, reverse_loader, spec, device, paired)
    if not np.array_equal(labels, rc_labels, equal_nan=True):
        raise RuntimeError("Forward and reverse-complement test loaders are not in the same order.")
    ref = (fw_ref + rc_ref) / 2.0
    pred = (fw_alt + rc_alt) / 2.0 - ref if paired else ref
    return pearson_from_arrays(pred, labels), int(labels.shape[0])


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


def r_squared(value: float) -> float:
    """Square of a Pearson r; NaN stays NaN."""
    return float("nan") if np.isnan(value) else float(value) ** 2


# =============================================================================
# Datasets, training run
# =============================================================================
# main() builds the datasets and calls run_split once. Training is
# --epoch_num epochs (default 10) on the full training split.


def build_datasets(
    args: argparse.Namespace,
    transforms: dict[str, t.Compose],
) -> dict[str, Any]:
    """Load every split once.

    The training split gets ``SoftBinTarget()``, so each item carries the
    18-bin probabilities used by the KL loss. Validation is the public
    ``all`` set. Each test subset is loaded twice, with the forward and
    the reverse-complement transform. Transforms draw from ``torch.rand``,
    which ``set_run_seed`` seeds before training.

    With ``--eval_only`` the training and validation splits are not loaded
    (``train`` and ``val`` are None): only the test subsets are needed.
    """
    train = val = None
    if not args.eval_only:
        print("loading train ...")
        train = RafiDataset(
            split="train",
            transform=transforms["train"],
            target_transform=t_t.SoftBinTarget(),
            root=args.root,
        )
        print(f"  train: {len(train)} sequences")
        val = RafiDataset(split="val", data_type=VAL_DATA_TYPE, transform=transforms["val"], root=args.root)
        print(f"  val ({VAL_DATA_TYPE}): {len(val)} sequences")
    tests = {}
    for data_type in args.data_type:
        tests[data_type] = {
            strand: RafiDataset(split="test", data_type=data_type, transform=transforms[strand], root=args.root)
            for strand in ("forward", "reverse")
        }
        print(f"  test {data_type} ({kind_of(data_type)}): {len(tests[data_type]['forward'])} sequences")
    return {"train": train, "val": val, "test": tests}


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
    datasets: dict[str, Any],
    device: torch.device,
) -> list[tuple[str, float, int]]:
    """Train one model on the training split and score every test subset with best.pt.

    Validation is the public ``all`` set. It is never used as a test set,
    and the test subsets are never used to choose the checkpoint.

    After each epoch the validation loader is scored and one row is
    appended to ``epochs.tsv``. The state dict is saved only when
    ``val_pearson`` strictly improves. After the last epoch that file is
    loaded, each test subset is scored, and ``test.tsv`` is rewritten
    with one row per subset scored so far. The
    weights sitting in memory after the last epoch are not the test
    weights unless that epoch happened to be the best.

    Returns ``(data_type, pearson, n_sequences)`` per test subset; empty
    if no finite validation Pearson was observed and therefore no
    checkpoint was written.
    """
    tokenizer, model = build_tokenizer_and_model(spec, args.backbone_dropout)
    model = model.to(device)
    collate_fn = make_collate_fn(tokenizer, spec)
    # --rc_tta adds reverse-complement tokens to the validation batches only.
    val_collate_fn = make_collate_fn(tokenizer, spec, add_rc=args.rc_tta)

    def eval_loader(dataset: RafiDataset, collate: Callable = collate_fn) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.run_seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        generator=loader_generator,
    )
    val_loader = eval_loader(datasets["val"], val_collate_fn)

    # Caduceus source runs use AdamW eps 1e-6; the other backbones use 1e-8.
    # The cosine schedule decays the learning rate across the epochs
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
        print(f"\n=== {args.model} | Rafi2024 | epoch {epoch_num}/{args.epoch_num} ===")
        train_loss, train_pearson = train_one_epoch(
            model, train_loader, optimizer, spec, device, on_step=on_step
        )
        # With --ema_decay, validation (and therefore best.pt selection)
        # runs on the EMA weights; the live weights are restored after.
        with ema_scope(ema, model):
            val_pearson = evaluate_val(model, val_loader, spec, device)
        print(
            f"train_loss={train_loss:.4f} train_pearson={train_pearson:.4f}\n"
            f"  val {VAL_DATA_TYPE}: pearson={val_pearson:.4f} r2={r_squared(val_pearson):.4f}"
        )
        append_row(
            os.path.join(args.result_dir, "epochs.tsv"),
            {
                "model": args.model,
                "val_data_type": VAL_DATA_TYPE,
                "epoch": epoch_num,
                "train_loss": train_loss,
                "train_pearson": train_pearson,
                "val_pearson": val_pearson,
                "val_pearson_r2": r_squared(val_pearson),
            },
            EPOCH_COLUMNS,
        )

        # Selection metric is val_pearson only, not the training loss and
        # not train_pearson. A NaN must not overwrite the file.
        if not np.isnan(val_pearson) and val_pearson > best_val:
            best_val = val_pearson
            best_epoch = epoch_num
            torch.save(checkpoint_state(model, ema), ckpt_path)
            print(f"new best val_pearson={best_val:.4f} -> {ckpt_path}")

        # plateau reads the selection score; epoch_cosine just steps.
        step_epoch_scheduler(scheduler, val_pearson)

    # Test only the saved epoch. Reloading the file also drops dropout state
    # and any in-memory weights from the final epoch when it was not best.
    # Every subset uses the same weights.
    scored: list[tuple[str, float, int]] = []
    if best_epoch is None or not os.path.exists(ckpt_path):
        print("=== TEST: no checkpoint, test skipped ===")
        return scored

    model.load_state_dict(_load_state_dict(ckpt_path, device))
    scored = score_test_subsets(
        args, spec, model, collate_fn, datasets, device, f"{ckpt_path} (epoch {best_epoch})"
    )

    del model, optimizer, scheduler, step_scheduler, ema, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scored


def score_test_subsets(
    args: argparse.Namespace,
    spec: ModelSpec,
    model: nn.Module,
    collate_fn: Callable,
    datasets: dict[str, Any],
    device: torch.device,
    source: str,
) -> list[tuple[str, float, int]]:
    """Score every subset in ``--data_type`` with the weights already in ``model``.

    Shared by training and ``--eval_only``, so both report the same
    numbers for the same checkpoint: forward and reverse-complement
    predictions are averaged, paired subsets use ``alt - ref``.
    ``test.tsv`` in ``args.result_dir`` is rewritten after every subset.
    ``source`` only labels the printout.
    """
    def eval_loader(dataset: RafiDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )

    scored: list[tuple[str, float, int]] = []
    for data_type in args.data_type:
        kind = kind_of(data_type)
        pearson, n_sequences = evaluate_test(
            model,
            eval_loader(datasets["test"][data_type]["forward"]),
            eval_loader(datasets["test"][data_type]["reverse"]),
            spec,
            device,
            paired=kind == "paired",
        )
        print(
            f"=== TEST {data_type} ({kind}, n={n_sequences}): {source} "
            f"pearson={pearson:.4f} r2={r_squared(pearson):.4f} ==="
        )
        scored.append((data_type, pearson, n_sequences))
        write_test_table(os.path.join(args.result_dir, "test.tsv"), scored)
    return scored


# Lazily created rotary-embedding caches (NTv3) that a fresh model lacks.
ROTARY_CACHE_SUFFIXES = (".cos_cached", ".sin_cached")


def evaluate_checkpoint(
    args: argparse.Namespace,
    spec: ModelSpec,
    datasets: dict[str, Any],
    device: torch.device,
    ckpt_path: str,
) -> list[tuple[str, float, int]]:
    """``--eval_only``: build the ``--model`` architecture, load ``ckpt_path``, score the test.

    No training, no validation, no optimizer. The checkpoint must come
    from the same ``--model`` (the 18-bin head and pool are part of the
    state dict); a mismatch aborts with the missing and unexpected keys.
    With ``--ema_decay`` during training, best.pt already holds the EMA
    weights, so nothing extra is needed here.
    """
    tokenizer, model = build_tokenizer_and_model(spec, args.backbone_dropout)
    state = _load_state_dict(ckpt_path, torch.device("cpu"))
    # NTv3's remote code registers its rotary cos/sin tables as buffers on
    # the first forward pass, so a trained best.pt contains them while a
    # freshly built model does not. They are recomputed, not learned, so
    # they are the only keys allowed to be left over; any missing key or any
    # other unexpected key still aborts.
    result = model.load_state_dict(state, strict=False)
    rebuilt = [key for key in result.unexpected_keys if key.endswith(ROTARY_CACHE_SUFFIXES)]
    other_unexpected = [key for key in result.unexpected_keys if not key.endswith(ROTARY_CACHE_SUFFIXES)]
    if result.missing_keys or other_unexpected:
        raise SystemExit(
            f"{ckpt_path} does not match --model {args.model}. "
            "Pass the --model the checkpoint was trained with.\n"
            f"missing keys: {result.missing_keys}\n"
            f"unexpected keys: {other_unexpected}"
        )
    if rebuilt:
        print(f"  skipped {len(rebuilt)} rotary cache buffers (recomputed on the first forward)")
    model = model.to(device)
    scored = score_test_subsets(
        args, spec, model, make_collate_fn(tokenizer, spec), datasets, device, ckpt_path
    )
    del model, tokenizer, state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scored


# =============================================================================
# Main
# =============================================================================


def ensure_results_file(path: str, columns: list[str]) -> None:
    """Create ``path`` with ``columns``, or accept an existing file with that header.

    A header mismatch means an older schema.
    Pass a new ``--result_dir`` instead of appending incompatible rows.
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
    killed run keeps the epochs already finished.
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


def write_test_table(path: str, scored: list[tuple[str, float, int]]) -> None:
    """Rewrite ``test.tsv``: header ``data_type, pearson_r, pearson_r2``, one row per subset."""
    table = pd.DataFrame(
        [[pearson, r_squared(pearson)] for _, pearson, _ in scored],
        index=pd.Index([data_type for data_type, _, _ in scored], name="data_type"),
        columns=TEST_COLUMNS,
    )
    table.to_csv(path, sep="\t")


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


def print_scored(prefix: str, scored: list[tuple[str, float, int]]) -> None:
    """One line per test subset: Pearson r and r^2."""
    for data_type, pearson, n_sequences in scored:
        printed_r = None if np.isnan(pearson) else round(pearson, 4)
        printed_r2 = None if np.isnan(pearson) else round(r_squared(pearson), 4)
        print(
            f"{prefix}test {data_type:<12} ({kind_of(data_type)}, n={n_sequences}): "
            f"pearson={printed_r} r2={printed_r2}"
        )


def run_eval_only(
    args: argparse.Namespace,
    spec: ModelSpec,
    datasets: dict[str, Any],
    device: torch.device,
    result_parent: str,
    model_parent: str,
) -> None:
    """``--eval_only``: score existing checkpoints, no training.

    With ``--checkpoint`` one file is scored and ``test.tsv`` is written to
    ``--result_dir``. Otherwise run k loads ``run{k}/best.pt`` under
    ``--model_dir`` and rewrites ``run{k}/test.tsv`` under ``--result_dir``
    for k = 1..``--runs``; a run without best.pt is skipped with a notice.
    ``epochs.tsv`` and ``seed.txt`` are left untouched.
    """
    if args.checkpoint is not None:
        os.makedirs(result_parent, exist_ok=True)
        args.result_dir = result_parent
        print(f"\n### eval_only — model: {args.model} — {args.checkpoint} ###")
        scored = evaluate_checkpoint(args, spec, datasets, device, args.checkpoint)
        print("\n" + os.path.join(result_parent, "test.tsv"))
        print_scored("", scored)
        return

    for run_idx in range(1, args.runs + 1):
        ckpt_path = os.path.join(model_parent, f"run{run_idx}", "best.pt")
        args.result_dir = os.path.join(result_parent, f"run{run_idx}")
        if not os.path.isfile(ckpt_path):
            print(f"\n### eval_only run {run_idx}/{args.runs}: {ckpt_path} not found, skipped ###")
            continue
        os.makedirs(args.result_dir, exist_ok=True)
        print(f"\n### eval_only run {run_idx}/{args.runs} — model: {args.model} — {ckpt_path} ###")
        scored = evaluate_checkpoint(args, spec, datasets, device, ckpt_path)
        print("\n" + os.path.join(args.result_dir, "test.tsv"))
        print_scored(f"run {run_idx} ", scored)


def main() -> None:
    """Train on Rafi2024 ``--runs`` times and score every test subset.

    Order of work: resolve the model spec, load the datasets once, then
    ``run_split`` once per run. Each run writes ``epochs.tsv``,
    ``test.tsv``, and ``seed.txt`` to ``run{k}/`` under ``--result_dir``
    and ``best.pt`` to ``run{k}/`` under ``--model_dir``.
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

    transforms = build_string_transforms(spec.rc_train_prob)

    print("=" * 60)
    print("LAUNCH PARAMETERS")
    print("=" * 60)
    print(f"  model          : {args.model}")
    print(f"  hub_id         : {spec.hub_id}")
    print(f"  outputs        : {N_BINS} FACS bins, KLDiv to SoftBinTarget, expected bin")
    print(f"  split          : train full, val {VAL_DATA_TYPE} (public)")
    print(f"  test subsets   : {', '.join(args.data_type)} (private)")
    print(f"  seq_len        : {SEQ_LEN}")
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
    print(f"  rc_tta         : {args.rc_tta}{' (val only)' if args.rc_tta else ''}")
    print(f"  device         : {device}")
    print(f"  root           : {args.root}")
    print(f"  result_dir     : {result_parent}")
    print(f"  model_dir      : {model_parent}")
    if args.eval_only:
        print(f"  eval_only      : {args.checkpoint or 'run{k}/best.pt for k = 1..' + str(args.runs)}")
        print("  (training flags above are ignored; --model must match the checkpoint)")
    print("=" * 60)

    # Datasets are seed-independent (transforms draw torch.rand per item), so
    # they are loaded once and shared; each run reseeds before training.
    datasets = build_datasets(args, transforms)

    if args.eval_only:
        run_eval_only(args, spec, datasets, device, result_parent, model_parent)
        return

    for run_idx in range(1, args.runs + 1):
        args.result_dir = os.path.join(result_parent, f"run{run_idx}")
        args.model_dir = os.path.join(model_parent, f"run{run_idx}")
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)
        run_seed = args.seed + run_idx - 1
        args.run_seed = run_seed
        seed_path = write_seed_file(args.result_dir, run_seed)
        set_run_seed(run_seed)
        print(f"\n### run {run_idx}/{args.runs} — model: {args.model} — Rafi2024 ###")
        print(f"  folder: {args.result_dir}")
        print(f"  seed: {run_seed} -> {seed_path}")
        scored = run_split(args, spec, datasets, device)

        print("\n" + "=" * 60)
        for name in ("epochs.tsv", "test.tsv"):
            path = os.path.join(args.result_dir, name)
            if os.path.exists(path):
                print("\n" + path)
                print(pd.read_csv(path, sep="\t").to_string(index=False))
        print("=" * 60)
        print_scored(f"run {run_idx} ", scored)
        print(f"checkpoint: {os.path.join(args.model_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
