import pandas as pd
from typing import List
import os
import warnings

from mpramnist.mpradataset import MpraDataset


class RafiDataset(MpraDataset):
    """
    Dataset class for DREAM challenge MPRA data with yeast-specific processing.

    This class handles loading, filtering, and preprocessing of MPRA (Massively Parallel
    Reporter Assay) data from the DREAM challenge, specifically designed for yeast
    (S. cerevisiae) regulatory element analysis. It supports various dataset types
    and experimental conditions relevant to yeast biology.

    The dataset contains sequences with measured transcriptional activity in yeast
    strains (primarily S288C background with ura3 auxotrophy), enabling the study
    of regulatory logic in yeast promoters and other DNA elements.

    Attributes
    ----------
    FLAG : str
        Identifier flag for Rafi datasets ("Dream")
    PLASMID : str
        Constant plasmid backbone sequence used in the MPRA constructs
    LEFT_FLANK : str
        Left flanking sequence used for sequence extraction and alignment
    RIGHT_FLANK : str
        Right flanking sequence used for sequence extraction and alignment
    TYPES : List[str]
        Supported dataset types:
        - "all": All sequences in the test data.
        - "high": Sequences designed to have high expression.
        - "low": Sequences designed to have low expression.
        - "yeast": Sequences that are present in the yeast genome.
        - "random": Random DNA sequences.
        - "challenging": Sequences designed to maximize the differences 
            between a convolutional model and a biochemical model trained on the same data.
        - "snv": Two sequences that differ by only a single base.
        - "perturbation": Two sequences that differ due to perturbations to specific known TF binding site.
        - "tiling": Two sequences that differ due to tiling known TF binding sites across random sequences.

    Parameters
    ----------
    split : str
        Data split specification. Valid values:
        - "train": Training data
        - "val" or "public": Validation/public test data  
        - "test" or "private": Private test data
    data_type : str | List[str], optional
        Specific dataset type(s) to load. For training split, this parameter is ignored.
        Single types: "high", "low", "yeast", "random", "challenging", "all"
        Paired types: "snv", "perturbation", "tiling"
    transform : callable, optional
        Transformation function applied to each sequence. Useful for data augmentation
        or sequence encoding. Should accept a sequence string and return transformed data.
    target_transform : callable, optional  
        Transformation function applied to target values. Useful for normalization
        or target processing.
    root : str, optional
        Root directory for data storage. If None, uses default data directory.

    Raises
    ------
    FileNotFoundError
        If the required data file cannot be found or downloaded
    ValueError
        If provided split or data_type parameters are invalid

    Examples
    --------
    >>> # Load training data for yeast regulatory elements
    >>> train_dataset = RafiDataset(split="train")
    >>> 
    >>> # Load validation data for high-activity sequences
    >>> val_dataset = RafiDataset(split="val", data_type="high")
    >>>
    >>> # Load test data for SNV analysis
    >>> test_dataset = RafiDataset(split="test", data_type="snv")
    >>>
    >>> # Load multiple dataset types
    >>> multi_dataset = RafiDataset(split="val", data_type=["high", "yeast"])

    Notes
    -----
    - Yeast strain information: Uses S288C background with ura3 auxotrophy
    - Sequence context: All sequences must be embedded in the specified plasmid backbone
    - Target values: Represent measured transcriptional activity in yeast
    - For paired types (snv/perturbation/tiling), both reference and alternative
      sequences are provided for comparative analysis

    """

    FLAG = "Dream"
    PLASMID = "aactctcaaggatcttaccgctgttgagatccagttcgatgtaacccactcgtgcacccaactgatcttcagcatcttttactttcaccagcgtttctgggtgagcaaaaacaggaaggcaaaatgccgcaaaaaagggaataagggcgacacggaaatgttgaatactcatactcttcctttttcaatattattgaagcatttatcagggttattgtctcatgagcggatacatatttgaatgtatttagaaaaataaacaaataggggttccgcgcacatttccccgaaaagtgccacctgacgtcatctatattaccctgttatccctagcggatctgccggtagaggtgtggtcaataagagcgacctcatactatacctgagaaagcaacctgacctacaggaaagagttactcaagaataagaattttcgttttaaaacctaagagtcactttaaaatttgtatacacttattttttttataacttatttaataataaaaatcataaatcataagaaattcgcttatttagaagtGGCGCGCCGGTCCGttacttgtacagctcgtccatgccgccggtggagtggcggccctcggcgcgttcgtactgttccacgatggtgtagtcctcgttgtgggaggtgatgtccaacttgatgttgacgttgtaggcgccgggcagctgcacgggcttcttggccttgtaggtggtcttgacctcagcgtcgtagtggccgccgtccttcagcttcagcctctgcttgatctcgcccttcagggcgccgtcctcggggtacatccgctcggaggaggcctcccagcccatggtcttcttctgcattacggggccgtcggaggggaagttggtgccgcgcagcttcaccttgtagatgaactcgccgtcctgcagggaggagtcctgggtcacggtcaccacgccgccgtcctcgaagttcatcacgcgctcccacttgaagccctcggggaaggacagcttcaagtagtcggggatgtcggcggggtgcttcacgtaggccttggagccgtacatgaactgaggggacaggatgtcccaggcgaagggcagggggccacccttggtcaccttcagcttggcggtctgggtgccctcgtaggggcggccctcgccctcgccctcgatctcgaactcgtggccgttcacggagccctccatgtgcaccttgaagcgcatgaactccttgatgatggccatgttatcctcctcgcccttgctcacCATGGTACTAGTGTTTAGTTAATTATAGTTCGTTGACCGTATATTCTAAAAACAAGTACTCCTTAAAAAAAAACCTTGAAGGGAATAAACAAGTAGAATAGATAGAGAGAAAAATAGAAAATGCAAGAGAATTTATATATTAGAAAGAGAGAAAGAAAAATGGAAAAAAAAAAATAGGAAAAGCCAGAAATAGCACTAGAAGGAGCGACACCAGAAAAGAAGGTGATGGAACCAATTTAGCTATATATAGTTAACTACCGGCTCGATCATCTCTGCCTCCAGCATAGTCGAAGAAGAATTTTTTTTTTCTTGAGGCTTCTGTCAGCAACTCGTATTTTTTCTTTCTTTTTTGGTGAGCCTAAAAAGTTCCCACGTTCTCTTGTACGACGCCGTCACAAACAACCTTATGGGTAATTTGTCGCGGTCTGGGTGTATAAATGTGTGGGTGCAACATGAATGTACGGAGGTAGTTTGCTGATTGGCGGTCTATAGATACCTTGGTTATGGCGCCCTCACAGCCGGCAGGGGAAGCGCCTACGCTTGACATCTACTATATGTAAGTATACGGCCCCATATATAggccctttcgtctcgcgcgtttcggtgatgacggtgaaaacctctgacacatgcagctcccggagacggtcacagcttgtctgtaagcggatgccgggagcagacaagcccgtcagggcgcgtcagcgggtgttggcgggtgtcggggctggcttaactatgcggcatcagagcagattgtactgagagtgcaccatatggacatattgtcgttagaacgcggctacaattaatacataaccttatgtatcatacacatacgatttaggtgacactatagaacgcggccgccagctgaagctttaactatgcggcatcagagcagattgtactgagagtgcaccataccaccttttcaattcatcattttttttttattcttttttttgatttcggtttccttgaaatttttttgattcggtaatctccgaacagaaggaagaacgaaggaaggagcacagacttagattggtatatatacgcatatgtagtgttgaagaaacatgaaattgcccagtattcttaacccaactgcacagaacaaaaacctgcaggaaacgaagataaatcatgtcgaaagctacatataaggaacgtgctgctactcatcctagtcctgttgctgccaagctatttaatatcatgcacgaaaagcaaacaaacttgtgtgcttcattggatgttcgtaccaccaaggaattactggagttagttgaagcattaggtcccaaaatttgtttactaaaaacacatgtggatatcttgactgatttttccatggagggcacagttaagccgctaaaggcattatccgccaagtacaattttttactcttcgaagacagaaaatttgctgacattggtaatacagtcaaattgcagtactctgcgggtgtatacagaatagcagaatgggcagacattacgaatgcacacggtgtggtgggcccaggtattgttagcggtttgaagcaggcggcagaagaagtaacaaaggaacctagaggccttttgatgttagcagaattgtcatgcaagggctccctatctactggagaatatactaagggtactgttgacattgcgaagagcgacaaagattttgttatcggctttattgctcaaagagacatgggtggaagagatgaaggttacgattggttgattatgacacccggtgtgggtttagatgacaagggagacgcattgggtcaacagtatagaaccgtggatgatgtggtctctacaggatctgacattattattgttggaagaggactatttgcaaagggaagggatgctaaggtagagggtgaacgttacagaaaagcaggctgggaagcatatttgagaagatgcggccagcaaaactaaaaaactgtattataagtaaatgcatgtatactaaactcacaaattagagcttcaatttaattatatcagttattaccctatgcggtgtgaaataccgcacagatgcgtaaggagaaaataccgcatcaggaaattgtaagcgttaatattttgttaaaattcgcgttaaatttttgttaaatcagctcattttttaaccaataggccgaaatcggcaaaatcccttataaatcaaaagaatagaccgagatagggttgagtgttgttccagtttggaacaagagtccactattaaagaacgtggactccaacgtcaaagggcgaaaaaccgtctatcagggcgatggcccactacgtgaaccatcaccctaatcaagtGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGAAGGCAAAGatgtctaaaggtgaagaattattcactggtgttgtcccaattttggttgaattagatggtgatgttaatggtcacaaattttctgtctccggtgaaggtgaaggtgatgctacttacggtaaattgaccttaaaattgatttgtactactggtaaattgccagttccatggccaaccttagtcactactttaggttatggtttgcaatgttttgctagatacccagatcatatgaaacaacatgactttttcaagtctgccatgccagaaggttatgttcaagaaagaactatttttttcaaagatgacggtaactacaagaccagagctgaagtcaagtttgaaggtgataccttagttaatagaatcgaattaaaaggtattgattttaaagaagatggtaacattttaggtcacaaattggaatacaactataactctcacaatgtttacatcactgctgacaaacaaaagaatggtatcaaagctaacttcaaaattagacacaacattgaagatggtggtgttcaattagctgaccattatcaacaaaatactccaattggtgatggtccagtcttgttaccagacaaccattacttatcctatcaatctgccttatccaaagatccaaacgaaaagagagaccacatggtcttgttagaatttgttactgctgctggtattacccatggtatggatgaattgtacaaataaggcgcgccacttctaaataagcgaatttcttatgatttatgatttttattattaaataagttataaaaaaaataagtgtatacaaattttaaagtgactcttaggttttaaaacgaaaattcttattcttgagtaactctttcctgtaggtcaggttgctttctcaggtatagtatgaggtcgctcttattgaccacacctctaccggcagatccgctagggataacagggtaatataGATCTGTTTAGCTTGCCTCGTCCCCGCCGGGTCACCCGGCCAGCGACATGGAGGCCCAGAATACCCTCCTTGACAGTCTTGACGTGCGCAGCTCAGGGGCATGATGTGACTGTCGCCCGTACATTTAGCCCATACATCCCCATGTATAATCATTTGCATCCATACATTTTGATGGCCGCACGGCGCGAAGCAAAAATTACGGCTCCTCGCTGCAGACCTGCGAGCAGGGAAACGCTCCCCTCACAGACGCGTTGAATTGTCCCCACGCCGCGCCCCTGTAGAGAAATATAAAAGGTTAGGATTTGCCACTGAGGTTCTTCTTTCATATACTTCCTTTTAAAATCTTGCTAGGATACAGTTCTCACATCACATCCGAACATAAACAACCATGGGTACCACTCTTGACGACACGGCTTACCGGTACCGCACCAGTGTCCCGGGGGACGCCGAGGCCATCGAGGCACTGGATGGGTCCTTCACCACCGACACCGTCTTCCGCGTCACCGCCACCGGGGACGGCTTCACCCTGCGGGAGGTGCCGGTGGACCCGCCCCTGACCAAGGTGTTCCCCGACGACGAATCGGACGACGAATCGGACGACGGGGAGGACGGCGACCCGGACTCCCGGACGTTCGTCGCGTACGGGGACGACGGCGACCTGGCGGGCTTCGTGGTCGTCTCGTACTCCGGCTGGAACCGCCGGCTGACCGTCGAGGACATCGAGGTCGCCCCGGAGCACCGGGGGCACGGGGTCGGGCGCGCGTTGATGGGGCTCGCGACGGAGTTCGCCCGCGAGCGGGGCGCCGGGCACCTCTGGCTGGAGGTCACCAACGTCAACGCACCGGCGATCCACGCGTACCGGCGGATGGGGTTCACCCTCTGCGGCCTGGACACCGCCCTGTACGACGGCACCGCCTCGGACGGCGAGCAGGCGCTCTACATGAGCATGCCCTGCCCCTAATCAGTACTGACAATAAAAAGATTCTTGTTTTCAAGAACTTGTCATTTGTATAGTTTTTTTATATTGTAGTTGTTCTATTTTAATCAAATGTTAGCGTGATTTATATTTTTTTTCGCCTCGACATCATCTGCCCAGATGCGAAGTTAAGTGCGCAGAAAGTAATATCATGCGTCAATCGTATGTGAATGCTGGTCGCTATACTGCTGTCGATTCGATACTAACGCCGCCATCCAGTGTCGAAAACGAGCTCGaattcctgggtccttttcatcacgtgctataaaaataattataatttaaattttttaatataaatatataaattaaaaatagaaagtaaaaaaagaaattaaagaaaaaatagtttttgttttccgaagatgtaaaagactctagggggatcgccaacaaatactaccttttatcttgctcttcctgctctcaggtattaatgccgaattgtttcatcttgtctgtgtagaagaccacacacgaaaatcctgtgattttacattttacttatcgttaatcgaatgtatatctatttaatctgcttttcttgtctaataaatatatatgtaaagtacgctttttgttgaaattttttaaacctttgtttatttttttttcttcattccgtaactcttctaccttctttatttactttctaaaatccaaatacaaaacataaaaataaataaacacagagtaaattcccaaattattccatcattaaaagatacgaggcgcgtgtaagttacaggcaagcgatccgtccGATATCatcagatccactagtggcctatgcggccgcggatctgccggtctccctatagtgagtcgtattaatttcgataagccaggttaacctgcattaatgaatcggccaacgcgcggggagaggcggtttgcgtattgggcgctcttccgcttcctcgctcactgactcgctgcgctcggtcgttcggctgcggcgagcggtatcagctcactcaaaggcggtaatacggttatccacagaatcaggggataacgcaggaaagaacatgtgagcaaaaggccagcaaaaggccaggaaccgtaaaaaggccgcgttgctggcgtttttccataggctccgcccccctgacgagcatcacaaaaatcgacgctcaagtcagaggtggcgaaacccgacaggactataaagataccaggcgtttccccctggaagctccctcgtgcgctctcctgttccgaccctgccgcttaccggatacctgtccgcctttctcccttcgggaagcgtggcgctttctcaTAgctcacgctgtaggtatctcagttcggtgtaggtcgttcgctccaagctgggctgtgtgcacgaaccccccgttcagcccgaccgctgcgccttatccggtaactatcgtcttgagtccaacccggtaagacacgacttatcgccactggcagcagccactggtaacaggattagcagagcgaggtatgtaggcggtgctacagagttcttgaagtggtggcctaactacggctacactagaagAacagtatttggtatctgcgctctgctgaagccagttaccttcggaaaaagagttggtagctcttgatccggcaaacaaaccaccgctggtagcggtggtttttttgtttgcaagcagcagattacgcgcagaaaaaaaggatctcaagaagatcctttgatcttttctacggggtctgacgctcagtggaacgaaaactcacgttaagggattttggtcatgagattatcaaaaaggatcttcacctagatccttttaaattaaaaatgaagttttaaatcaatctaaagtatatatgagtaaacttggtctgacagttaccaatgcttaatcagtgaggcacctatctcagcgatctgtctatttcgttcatccatagttgcctgactccccgtcgtgtagataactacgatacgggagggcttaccatctggccccagtgctgcaatgataccgcgagacccacgTtcaccggctccagatttatcagcaataaaccagccagccggaagggccgagcgcagaagtggtcctgcaactttatccgcctccatccagtctattaattgttgccgggaagctagagtaagtagttcgccagttaatagtttgcgcaacgttgttgccattgctacaggcatcgtggtgtcacgctcgtcgtttggtatggcttcattcagctccggttcccaacgatcaaggcgagttacatgatcccccatgttgtgcaaaaaagcggttagctccttcggtcctccgatcgttgtcagaagtaagttggccgcagtgttatcactcatggttatggcagcactgcataattctcttactgtcatgccatccgtaagatgcttttctgtgactggtgagtactcaaccaagtcattctgagaatagtgtatgcggcgaccgagttgctcttgcccggcgtcaatacgggataataccgcgccacatagcagaactttaaaagtgctcatcattggaaaacgttcttcggggcgaa"

    LEFT_FLANK = "TGCATTTTTTTCACATC"
    RIGHT_FLANK = "GGTTACGGCTGTT"
    # Supported dataset types
    TYPES = [
        "all",  # all, high, low, yeast, random, challenging are single types
        "high",
        "low",
        "yeast",  # yeast is the 'native' type
        "random",
        "challenging",
        "snv",  # snv's, perturbation, tiling are paired types
        "perturbation",
        "tiling",
    ]

    def __init__(
        self,
        split: str,
        data_type: str | List[str] = None,
        transform=None,
        target_transform=None,
        joint_transform=None,
        root=None,
    ):
        """
        Initialize RafiDataset for yeast MPRA data analysis.

        The dataset is specifically designed for studying regulatory elements
        in yeast (S. cerevisiae) using MPRA technology. It provides access to
        various experimental conditions and sequence types relevant to yeast
        transcriptional regulation research.

        Parameters
        ----------
        split : str
        Data split specification. Valid values:
            - "train": Training data
            - "val" or "public": Validation/public test data  
            - "test" or "private": Private test data
        data_type : str | List[str], optional
            Specific dataset type(s) to load. For training split, this parameter is ignored.
            Single types: "high", "low", "yeast", "random", "challenging", "all"
            Paired types: "snv", "perturbation", "tiling"
        transform : callable, optional
            Transformation function applied to each sequence. Useful for data augmentation
            or sequence encoding. Should accept a sequence string and return transformed data.
        target_transform : callable, optional  
            Transformation function applied to target values. Useful for normalization
            or target processing.
        root : str, optional
            Root directory for data storage. If None, uses default data directory.
        """
        super().__init__(split, root)

        # Initialize transformations
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.prefix = self.FLAG + "_"
        self.cell_type = "strains S288C::ura3, etc"

        # Parse and validate inputs
        self.split = self.split_parse(split)

        # Load and prepare dataset
        self.dataset, self.data_type = self._define_dataset(self.split, data_type)
        self.target_column = (
            "label" if self.dataset in ["train", "single"] else "delta_measured"
        )
        self.df = self._load_and_prepare_data(self.dataset)

        # Prepare data structure based on split type
        self._prepare_data_structure()

        self.name_for_split_info = self.prefix

    def _load_and_prepare_data(self, dataset: str) -> pd.DataFrame:
        """
        Load and prepare the dataset from TSV files.

        Downloads the data file if not already present, then loads it into a pandas DataFrame.
        Handles both local and remote data sources through the parent class download mechanism.

        Parameters
        ----------
        dataset : str
            Name of the dataset to load ('train', 'single', or 'paired')

        Returns
        -------
        pd.DataFrame
            Loaded and filtered dataset

        Raises
        ------
        FileNotFoundError
            If the data file cannot be found after attempted download
        """
        try:
            file_name = self.prefix + dataset + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            print(file_path)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        return self._filter_dataset(df, self.split, self.data_type)

    def _prepare_data_structure(self):
        """
        Prepare the internal data structure based on dataset type.

        For single-type datasets (train, single), creates a dictionary with:
        - 'targets': numpy array of target values (labels or delta_measured)
        - 'seq': numpy array of sequence strings

        For paired-type datasets (paired), additionally includes:
        - 'seq_alt': numpy array of alternative sequences for comparative analysis

        """
        if self.dataset in ["train", "single"]:
            self.ds = {
                "targets": self.df[self.target_column].to_numpy(),
                "seq": self.df.seq.to_numpy(),
            }
        elif self.dataset == "paired":
            self.ds = {
                "targets": self.df[self.target_column].to_numpy(),
                "seq": self.df.seq.to_numpy(),
                "seq_alt": self.df.seq_alt.to_numpy(),
            }

    def _filter_dataset(
        self, df: pd.DataFrame, split: str, dataset_type: str | List[str]
    ) -> pd.DataFrame:
        """
        Filter dataset based on split and data type specifications.

        For training splits, returns the entire dataset without filtering.
        For validation/test splits, applies filtering based on public/private
        designation and specific data types.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the full dataset
        split : str
            Data split to filter ('public' or 'private')
        dataset_type : str | List[str]
            Specific data type(s) to include

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only the specified split and types
        """
        if split == "train":
            return df
        else:
            public_or_private = df[df[split] == 1]
            if isinstance(dataset_type, str):
                dataset_type = [dataset_type]
            selected_rows = public_or_private[
                (public_or_private[dataset_type] == 1).any(axis=1)
            ]
            return selected_rows

    def _define_dataset(
        self, split: str, dataset_type: str | List[str]
    ) -> tuple[str, str | List[str]]:
        """
        Determine which dataset to load based on type and split parameters.

        Maps user-friendly type names to internal dataset categories and validates
        type combinations. Handles both single and multiple type specifications.

        Parameters
        ----------
        split : str
            Data split specification
        dataset_type : str | List[str]
            Requested dataset type(s)

        Returns
        -------
        tuple[str, str | List[str]]
            Tuple containing:
            - dataset_category: 'train', 'single', or 'paired'
            - processed_type: validated and normalized type specification

        Raises
        ------
        ValueError
            If invalid type combinations are provided
        """
        single_types = ["high", "low", "yeast", "challenging", "random", "all"]
        paired_types = ["snv", "perturbation", "tiling"]

        if split == "train":
            if dataset_type is not None:
                warnings.warn(
                    "WARNING! The training set was selected, "
                    "\nso the 'type' parameter is ignored.",
                    stacklevel=1,
                )
            return "train", None

        if isinstance(dataset_type, str):
            return self._handle_single_type(dataset_type, single_types, paired_types)
        elif isinstance(dataset_type, List):
            return self._handle_multiple_types(dataset_type, single_types, paired_types)
        else:
            raise ValueError(f"Invalid type: {dataset_type}")

    def _handle_single_type(
        self, dataset_type: str, single_types: list, paired_types: list
    ) -> tuple[str, str]:
        """
        Process single dataset type specification.

        Parameters
        ----------
        dataset_type : str
            Single dataset type string
        single_types : list
            Valid single dataset types
        paired_types : list
            Valid paired dataset types

        Returns
        -------
        tuple[str, str]
            Dataset category and normalized type

        Raises
        ------
        ValueError
            If the specified type is not in supported types
        """
        lower_type = dataset_type.lower()
        if lower_type in single_types:
            return "single", lower_type
        elif lower_type in paired_types:
            return "paired", lower_type
        raise ValueError(f"Invalid type: {dataset_type}. Expected one of: {self.types}")

    def _handle_multiple_types(
        self, dataset_types: List[str], single_types: list, paired_types: list
    ) -> tuple[str, List[str]]:
        """
        Process multiple dataset type specifications.

        Validates that all specified types belong to the same category
        (either all single types or all paired types). Mixed categories
        are not allowed.

        Parameters
        ----------
        dataset_types : List[str]
            List of dataset type strings
        single_types : list
            Valid single dataset types
        paired_types : list
            Valid paired dataset types

        Returns
        -------
        tuple[str, List[str]]
            Dataset category and normalized types

        Raises
        ------
        ValueError
            If types are mixed between single and paired categories
        """
        lower_types = [t.lower() for t in dataset_types]

        if all(t in single_types for t in lower_types):
            return "single", lower_types
        elif all(t in paired_types for t in lower_types):
            return "paired", lower_types
        raise ValueError(
            f"Invalid types: {dataset_types}. "
            f"All types must be from {single_types} or {paired_types}"
        )

    def split_parse(self, split: str) -> str:
        """
        Parse and validate the split parameter.

        Converts user-friendly split names to internal representations:
        - 'train' -> 'train'
        - 'val' or 'public' -> 'public' 
        - 'test' or 'private' -> 'private'

        Parameters
        ----------
        split : str
            Input split specification

        Returns
        -------
        str
            Validated internal split representation

        Raises
        ------
        ValueError
            If split parameter is not recognized

        Examples
        --------
        >>> dataset.split_parse('val')
        'public'
        >>> dataset.split_parse('test')
        'private'
        >>> dataset.split_parse('train')
        'train'
        """
        valid_splits = {
            "train": "train",
            "val": "public",
            "public": "public",
            "test": "private",
            "private": "private",
        }

        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: {split}. "
                "Expected 'train', 'val'/'public', or 'test'/'private'."
            )

        return valid_splits[split]
