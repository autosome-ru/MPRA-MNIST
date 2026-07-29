import pandas as pd
import numpy as np
import os

from mpramnist.dataclass import seqobj, ScalarFeature, Categorial
from typing import Callable, ClassVar

from mpramnist.mpradataset import MpraDataset

class FromelDataset(MpraDataset):
    """
    Dataset class for the "Fromel2025" MPRA (Massively Parallel Reporter Assay) data.
 
    The raw data is stored in a single long-format TSV file where **each row
    corresponds to a (sequence, cell state) pair**, not to a sequence alone.
    That is, the same regulatory sequence (identified by ``Seq`` / ``CRS`` /
    ``Library``) appears in *several* rows -- one row per measured cell
    state / cluster (column ``clusterID``, e.g. ``State_1M``, ``State_2D``, ...).
 
    During ``__init__`` the data is pivoted from this long format into a
    wide format, where each unique sequence becomes exactly one row and each
    cell state becomes its own column/target. This is what turns "one raw
    TSV row" into a *vector* of several target values per sequence -- see
    the detailed comment above the ``data.pivot(...)`` call below.
 
    Two cell types are supported:
      - ``HSPC``: 7 regression targets (one per hematopoietic stem/progenitor
        cell state, see ``HSPC_TARGETS``).
      - ``K562``: 1 regression target (``State_9K``).
    """

    # ------------------------------------------------------------------
    # Constant flanking sequences that are attached/expected around every
    # oligo in the library (used for validation / sequence reconstruction
    # elsewhere in the pipeline, not directly used inside this class body).
    # ------------------------------------------------------------------
    CONSTANT_LEFT_FLANK: ClassVar[str] = "AGGACCGGATCAACT"  # required for each sequence
    CONSTANT_RIGHT_FLANK: ClassVar[str] = "CATTGCGTGAACCGA"  # required for each sequence
    LEFT_FLANK: ClassVar[str] = "GGCCCGCTCTAGACCTGCAGG" 
    RIGHT_FLANK: ClassVar[str] = (
        "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"
    )

    SUBDATASETS: ClassVar[dict[str, list[str]]] = {
        "HSPC_TRAINVALID" : [
            'HSPC.libB.DATA',
            'HSPC.libB.CONTROLS.GENERAL',
            'HSPC.libB.CONTROLS.TP53',
            'HSPC.libA.DATA',
            'HSPC.libC.DATA',
            'HSPC.libC.CONTROLS.GENERAL',
            'HSPC.libC.CONTROLS.TP53',
            'HSPC.libF.DATA',
            'HSPC.libF.CONTROLS.GENERAL'
            'HSPC.libF.CONTROLS.TP53'],
        "K562_TRAINVALID": [
            'K562.libC.minP.tra.DATA',
            'K562.libC.minP.tra.CONTROLS.GENERAL',
            'K562.libC.minP.tra.CONTROLS.TP53',
            'K562.libA.minP.tra.DATA',
            'K562.libB.minP.tra.DATA',
            'K562.libB.minP.tra.CONTROLS.GENERAL',
            'K562.libB.minP.tra.CONTROLS.TP53'],
        "TEST_GENOMIC": ['HSPC.libG.DATA'],
        "TEST_SYNTHETIC": ['HSPC.libH.DATA'],
        "TEST_GENERATED": ['HSPC.libD.DATA'],
        "K562_INT": [
            'K562.libB.minP.int.DATA',
            'K562.libB.minP.int.CONTROLS.GENERAL',
            'K562.libB.minP.int.CONTROLS.TP53'
        ],
        "K562_minCMV": [
            'K562.libB.minCMV.tra.DATA',
            'K562.libB.minCMV.tra.CONTROLS.GENERAL',
            'K562.libB.minCMV.tra.CONTROLS.TP53'
        ]
    }

    # The 7 HSPC cell states. Each of these is a distinct value that can
    # appear in the `clusterID` column of the TSV, and -- after pivoting --
    # each becomes one output column / regression target.
    HSPC_TARGETS: list[str] = [
        'State_1M',
        'State_2D',
        'State_3E',
        'State_4M',
        'State_5M',
        'State_6N',
        'State_7M',
    ]

    # Maps a library name to a numeric "batch" id. Used as an optional extra
    # categorical input feature so that a model can
    # learn/condition on batch effects between libraries.
    HSPC_BATCHES = {
        'libA': 0,
        'libB': 1,
        'libC': 2,
        'libH': 2,
        'libD': 3,
        'libF': 4,
        'libG': 4
    }


    CELL_TYPES = ["HSPC", "K562"]
    FLAG: ClassVar[str] = 'Fromel2025' # used as a filename prefix for the downloaded data file

    def __init__(self,
                 split: list[int] | str | int, # folds 
                 cell_type: str = 'HSPC',
                 upper_seq: bool = True, # return sequence in upper-case or return in mixed-case format showing motif placement for most sequences in the dataset
                 targets: list[str] | str | None = None,
                 add_batch_info: bool = True,
                 state_level_value: str = 'mean.norm.adj',
                 transform: Callable | None = None,
                 target_transform: Callable | None = None,
                 root: str | None = None
                ):
        """
        Load, filter and reshape the Fromel2025 MPRA data.
 
        Parameters
        ----------
        split : list[int] | str | int
            Which fold(s)/subset of the data to use. Can be:
              - ``"train"`` / ``"val"`` / ``"test"``: predefined fold groups
                (folds 0-8 / fold 9 / fold 10 respectively), drawn from the
                train/valid subdataset for the given ``cell_type``.
              - ``"genome"`` / ``"synthetic"`` / ``"generated"``: special
                held-out test subdatasets (HSPC only) that use *all* their
                rows (``folds = "test"``) regardless of fold number.
              - a single ``int`` or a ``list[int]``: explicit fold id(s) to
                use from the train/valid subdataset.
        cell_type : str, default "HSPC"
            Either ``"HSPC"`` or ``"K562"``. Determines which subdataset(s)
            and which targets are used (see ``process_targets``).
        upper_seq : bool, default True
            If True, upper-case all sequences. If False, keep the original
            mixed-case sequences, where lower-case stretches typically mark
            background/spacer sequence and upper-case stretches mark inserted
            motifs (this depends on how the raw ``Seq`` column was produced
            upstream).
        targets : list[str] | str | None
            Which target column(s) to expose. Only meaningful for
            ``cell_type == "HSPC"`` (K562 always has exactly one target,
            ``State_9K``); see ``process_targets`` for validation logic.
        add_batch_info : bool, default True
            If True (and ``cell_type == "HSPC"``), attach a categorical
            "batch" feature (see ``HSPC_BATCHES``) to each returned sample.
            Ignored for K562.
        state_level_value : str, default "mean.norm.adj"
            Name of the TSV column holding the per-(sequence, state) scalar
            value to use as the regression target (e.g. a normalized/adjusted
            activity score). This is the column that gets spread across
            columns during the pivot step below.
        transform : Callable | None
            Optional transform applied to the ``seqobj`` before returning it.
        target_transform : Callable | None
            Optional transform applied to the target vector before returning it.
        root : str | None
            Root directory where the raw data file is/will be downloaded.
        """

        super().__init__(split, root)
        self.cell_type = cell_type 
        self.split = split
        self.state_level_value = state_level_value
        self.upper_seq = upper_seq
        # Validate/normalize the requested target column names for this cell type.
        self.targets = self.process_targets(targets)
        # Figure out which `source` values (subdataset) and which `fold`
        # values correspond to the requested `split`.
        subdataset_name, folds = self.split_parse(split)

        subdataset_cols = self.SUBDATASETS[subdataset_name]

        self.prefix = self.FLAG + "_"

        # ------------------------------------------------------------
        # Download (if needed) and load the single raw TSV file that
        # contains ALL sequences, all cell types, all libraries and all
        # cell states in long format.
        # ------------------------------------------------------------
        try:
            file_name = self.prefix + "Fromel" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            data = pd.read_csv(file_path, sep="\t", dtype={'fold': 'string'})
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.upper_seq:
            data['Seq'] = data['Seq'].str.upper()

        # ------------------------------------------------------------
        # HSPC-specific filtering: keep only rows whose `source` starts
        # with "HSPC", and derive a numeric `batch` id from the library
        # part of `source` (e.g. "HSPC.libA.DATA" -> "libA" -> batch 0).
        # ------------------------------------------------------------
        if self.cell_type == 'HSPC':
            data = data[data['source'].str.startswith('HSPC')]
            data['batch'] = data['source'].apply(lambda x: self.HSPC_BATCHES[x.split('.', 2)[1]])
            self.add_batch_info = add_batch_info
        else:
            # K562 has no per-library batch grouping defined here, so
            # batch info is never attached for it, regardless of the
            # `add_batch_info` argument.
            self.add_batch_info = False

        # Keep only rows belonging to the requested subdataset (list of
        # `source` values) and to the requested fold(s).
        data = data[data['source'].isin(subdataset_cols)]
        data = data[data['fold'].isin(folds)]

        # `index` defines what uniquely identifies ONE sequence/sample
        # after pivoting (see below). `batch` is included here so that
        # rows for the same sequence keep their batch id attached to the
        # resulting single row.
        index = ["Seq", "CRS", "Library"]
        if self.add_batch_info:
            index.append('batch')

        # ------------------------------------------------------------
        # *** This is where "one row per (sequence, state)" becomes
        # "one row per sequence, with one column per state" ***
        #
        # In the raw TSV, a single sequence (same Seq/CRS/Library, and
        # same batch if HSPC) appears in MULTIPLE rows -- one row per
        # distinct `clusterID` value (e.g. "State_1M", "State_2D", ...
        # up to "State_7M" for HSPC, or "State_9K" for K562). Each of
        # those rows carries the scalar value for that one state in the
        # `state_level_value` column (default: "mean.norm.adj").
        #
        # `pivot(index=..., columns="clusterID", values=state_level_value)`
        # groups all rows that share the same `index` (i.e. all rows
        # belonging to the same physical sequence) and spreads their
        # `clusterID` values out into separate columns, placing the
        # corresponding `state_level_value` into each column. So instead
        # of N rows x 1 value-column, you get 1 row x N state-columns.
        #
        # This is exactly why `__getitem__` can return a vector of 7
        # target values (for HSPC) from what looks like "one row" in a
        # small excerpt of the TSV -- the other 6 rows for that same
        # sequence (with clusterID = State_2D, State_3E, ... State_7M)
        # are elsewhere in the file and get folded into the same row here.
        # ------------------------------------------------------------
        data = data.pivot(index=index,
           columns="clusterID",
           values=self.state_level_value).reset_index()
        
        if self.split == 'generated' or self.split == 'genome':
            data['State_5M'] = np.nan
            # Library F and two additional libraries (D and G, see below)
            # were measured in six cell states, since differences between 
            # early and late monocyte precursors in Library A and B were minimal

        self.data = data
        self.seqs = data['Seq'].values
        if self.add_batch_info:
            self.batch = data['batch'].values

        # Select just the requested target columns (in the requested order)
        # as the final target matrix, shape (n_samples, n_targets).
        self.target = data[self.targets].values

        self.name_for_split_info = ''
        self.info = {'task': 'regression', 'description': 'TODO'}
        self.transform = transform
        self.target_transform = target_transform

    def process_targets(self, targets: list[str] | str | None):
        """
        Validate and normalize the `targets` argument for the given cell type.
 
        - For K562: always returns ``['State_9K']`` (the only available
          target), ignoring whatever was passed in.
        - For HSPC: if `targets` is None, `'all'`, or already equal to the
          full `HSPC_TARGETS` list, returns all 7 targets. Otherwise,
          normalizes a single string into a one-element list and validates
          that every requested target name is one of `HSPC_TARGETS`,
          raising an Exception otherwise.
 
        Parameters
        ----------
        targets : list[str] | str | None
            Requested target column name(s).
 
        Returns
        -------
        list[str]
            The validated list of target column names to use.
        """
        if self.cell_type == "K562":
            targets = ['State_9K']
            
        elif self.cell_type == "HSPC":
            if targets is None or targets == 'all' or targets == self.HSPC_TARGETS:
                targets = list(self.HSPC_TARGETS)
            else:
                if isinstance(targets, str):
                    targets = [targets]
                for ta in targets:
                    if ta not in self.HSPC_TARGETS:
                        raise Exception(f'Wrong target {ta} for cell type {self.cell_type}')
        else:
            raise Exception(f'Wrong cell type: {self.cell_type}')
        return targets

    def split_parse(self, split: str | list[int] | int) -> tuple[str, list[str]]:
        """
        Translate the user-facing `split` argument into:
          1. the name of the `SUBDATASETS` entry to filter `source` by, and
          2. the list of `fold` values (as strings, matching the TSV's
             string-typed `fold` column) to keep.
 
        Supported values of `split`:
          - "train": folds 0-8 (as strings), subdataset = train/valid set
            for the current `cell_type`.
          - "val": fold 9, same subdataset selection as "train".
          - "test": fold 10, same subdataset selection as "train".
          - "genome" / "synthetic" / "generated": special held-out test
            subdatasets available only for HSPC; uses **all** rows of that
            subdataset (folds = ["test"], a sentinel that does not filter
            by an actual fold id since these subdatasets don't use the
            regular fold scheme the same way).
          - int: a single explicit fold id from the train/valid subdataset.
          - list[int]: multiple explicit fold ids from the train/valid
            subdataset.
 
        Returns
        -------
        tuple[str, list[str]]
            (subdataset_name, folds) where `subdataset_name` indexes into
            `SUBDATASETS` and `folds` is the list of fold-id strings (or
            the special ["test"] sentinel) to filter the `fold` column by.
        """
        # Process string input
        if split == 'train':
            folds = [0,1,2,3,4,5,6,7,8]
        elif split == 'val':
            folds = [9]
        elif split == 'test':
            folds = [10]
        elif split == 'genome':
            folds = 'test'
            if self.cell_type == 'HSPC':
                subdataset = 'TEST_GENOMIC'
            elif self.cell_type == 'K562':
                raise Exception(f'Genomic sequences were not measured for {self.cell_type}')
            else:
                raise Exception(f'Wrong {self.cell_type}')
        elif split == 'synthetic':
            folds = 'test'
            if self.cell_type == 'HSPC':
                subdataset = 'TEST_SYNTHETIC'
            elif self.cell_type == 'K562':
                raise Exception(f'Synthetic sequences were not measured for {self.cell_cell_typeline}')
            else:
                raise Exception(f'Wrong {self.cell_type}')
        elif split == 'generated':
            folds = 'test'
            if self.cell_type == 'HSPC':
                subdataset = 'TEST_GENERATED'
            elif self.cell_type == 'K562':
                raise Exception(f'Generated sequences were not measured for {self.cell_type}')
            else:
                raise Exception(f'Wrong {self.cell_type}')
        elif isinstance(split, int):
            folds = [split]
        elif isinstance(split, list):
            for i in split:
                if not isinstance(i, int):
                    raise Exception(f'Wrong fold value: {i}')
            folds = split
        else:
            raise Exception(f'Wrong split: {split}')
        
        if isinstance(folds, list): # list of ints 
            if self.cell_type == 'HSPC':
                subdataset = 'HSPC_TRAINVALID'
            elif self.cell_type == 'K562':
                subdataset = 'K562_TRAINVALID'
            else:
                raise Exception(f'Wrong {self.cell_type}')
            folds = [str(i) for i in folds]
        else: # str
            folds = [folds]
            
        return subdataset, folds

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):  
        """
        Build and return one training sample.
 
        Parameters
        ----------
        idx : int
            Row index into the pivoted (one-row-per-sequence) data.
 
        Returns
        -------
        tuple
            ``(seq, target)`` where:
              - ``seq`` is the (optionally transformed) encoded sequence
                produced by ``seqobj`` (e.g. a one-hot tensor, optionally
                with an extra batch-id feature channel).
              - ``target`` is a ``np.float32`` array with one value per
                requested target column (e.g. shape ``(7,)`` for HSPC when
                all states are requested, ``(1,)`` for K562), optionally
                passed through ``target_transform``.
        """
        sequence = self.seqs[idx]

        # Wrap the raw nucleotide string into a `seqobj`, optionally
        # reserving an extra feature channel for the batch id.
        seq = seqobj(seq=sequence,
                     scalars={},
                     vectors={}, 
                     add_feature_channel=self.add_batch_info)
        if self.add_batch_info:
            # Attach the batch id as a categorical scalar feature, using
            # the library names as the category levels.
            seq.scalars['batch'] = ScalarFeature(self.batch[idx], tp=Categorial(levels=list(self.HSPC_BATCHES.keys())))

        if self.transform is not None:
            seq = self.transform(seq)

        # `self.target[idx]` is the row of the pivoted target matrix for
        # this sequence: one value per requested target/state column.
        target = self.target[idx].astype(np.float32)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return seq.seq, target
