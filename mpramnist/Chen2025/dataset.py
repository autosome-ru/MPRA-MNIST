import pandas as pd
import os
from typing import List, Union, Dict, Optional
from collections import defaultdict
import pyfaidx
import subprocess
import bioframe as bf
import torch
from ..dataclass import seqobj
import numpy as np

from mpramnist.mpradataset import MpraDataset


class ChenMultiDataset(MpraDataset):
    """
    Dataset class for Chen MPRA (Massively Parallel Reporter Assay) data.
    
    This class handles loading, filtering, and processing of genomic sequence data
    from the Chen et al. study, which contains information about 599 non-coding 
    Late-onset Alzheimer's disease associated SNPs/variants, tested in 855 
    constructs across multiple cell types.

    The dataset uses human genome assembly hg38 with 0-based coordinate indexing.                         
    All genomic positions (start, end) follow 0-based indexing convention.

    Inherits from:
        MpraDataset: Base class for MPRA datasets

    Constants:
        FLAG (str): Dataset identifier flag: 'Chen'
        CELL_TYPE (dict): Mapping of cell types to their corresponding states

    Examples:
        >>> # Load data for specific cell types
        >>> dataset = ChenMultiDataset(cell_types=['THP1', 'HMC3'])
        >>> 
        >>> # Load data for specific cell types and states
        >>> dataset = ChenMultiDataset(cell_types=['THP1', 'HMC3'], 
        >>>                                 states = ['IFNB', 'IFNG'])
        >>> 
        >>> # Load data with custom sequence length
        >>> dataset = ChenMultiDataset(length=200, cell_types='Brain', states='Cortex')
        >>> 
        >>> # Load data filtered by genomic regions
        >>> dataset = ChenMultiDataset(
        ...     genomic_regions='path/to/regions.bed',
        ...     cell_types='Brain',
        ...     targets='Cortex'
        ... )
    """
    
    FLAG = "Chen"
    
    # Mapping of elements to their corresponding cell types
    CELL_TYPE = {'THP1': ['aggregated', 'Naive', 'IFNB', 'IFNG', 'LPSIFNG'],
                 'HMC3': ['aggregated','Naive', 'IFNB', 'IFNG', 'LPSIFNG'],
                 'Brain': ['aggregated','Cortex', 'Hippocampus', 'Striatum']}

    BASE_COLUMNS = ['RSID', 'interval_type', 'chromosome', 'pos_hg38', 'ref', 'alt', 'hg', 'snp_position', 'interval_center', 'reverse_prediction']

    def __init__(
        self,
        split: str = "test",
        length: int = 227,  # length of cutted sequence
        cell_types: list[str] | str = None,
        states: list[list[str]] = None,
        interval_type: str = None,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize the Chen MPRA dataset.
        
        Attributes
        ----------
        split : str, optional
            Specifies how to split the data. Currently only "test" is supported.
            Default is "test".
        length : int, optional  
            Length of the sequence for the differential expression experiment. 
            Must be positive integer. Default is 227.
        cell_types : Union[list[str], str], optional
            List of cell types to filter by.
            Can be a single string or list of strings.
        states : Union[list[list[str]], list[str]], optional
            List of states to be used for each cell type. 
        interval_type : str, optional
            Type of intervals to be used from the assay.
            Can be 'SNPCENTER' with SNP in the center
            or 'PEAKCENTER' with open chromatin peak summit in the center.
        genomic_regions : str | List[Dict], optional
            Genomic regions to include/exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool
            If True, exclude the specified regions instead of including them.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data (expression values).
        root : str, optional
            Root directory where data is stored. If None, uses default data directory.
        """
        # Initialize parent class
        super().__init__(split, root)

        self.transform = transform
        self.target_transform = target_transform
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions
        self.prefix = self.FLAG + "_"  # Prefix for file names

        # validate cell types
        if cell_types and (
            (isinstance(cell_types, str) and (cell_types not in self.CELL_TYPE))
            or
            (isinstance(cell_types, list) and not all(ct in self.CELL_TYPE for ct in cell_types))
        ):
            raise ValueError("Invalid cell type")
        

        # Validate sequence length parameter
        if not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"Parameter 'length' must be natural integer, not {length}."
            )
        self.length = length

        if cell_types:
            if not states: 
                states = [['all'] * len(cell_types)]
            states = [[state] if not isinstance(state, list) else state for state in states]
            if len(states) != len(cell_types):
                raise ValueError("Provide states for every cell line")
            for state, cell_type in zip(states, cell_types):
                if not all(st in self.CELL_TYPE[cell_type] for st in state):
                    raise ValueError(f"Invalid states {state} for cell type {cell_type}")

        try:
            # Load the data file
            file_name = self.prefix + "SNP" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process data - ensure proper chromosome formatting
        df.chromosome = df.chromosome.astype(str)
        df.snp_position = df.snp_position.astype(int)
        df.interval_center = df.interval_center.astype(int)
        self.ds = df

        if self.genomic_regions is None:
    
            # Filter by cell types if specified
            if cell_types is not None:
                # Convert single cell type to list for consistency
                if isinstance(cell_types, str):
                    cell_types = [cell_types]

                all_cells = []
                
                # Validate states provided for each cell type
                if states:
                    if len(states) != len(cell_types):
                        raise ValueError("Provide states for every cell line")
                    states = [[state] if not isinstance(state, list) else state for state in states]
                    
                    for state, cell_type in zip(states, cell_types):
                        if not all(st in self.CELL_TYPE[cell_type] for st in state):
                            raise ValueError(f"Invalid states {state} for cell type {cell_type}")
                        
                        if 'all' in state:
                            all_cells = all_cells + [f'{cell_type}_{st}' for st in self.CELL_TYPE[cell_type]]
                        else:
                            all_cells = all_cells + [f'{cell_type}_{st}' for st in state]

                # Filter rows where any of the processed cell types matches the requested cell types
                self.ds = self.ds[self.BASE_COLUMNS + [f'{prefix}_{ct}' for ct in all_cells for prefix in ['logFC', 'fdr', 'pval', 'statistic']]]

        else:
            # If self.genomic_regions is not None filter by genomic regions 
            self.ds = self.filter_by_genomic_regions(self.ds)

        if interval_type:
            if not (isinstance(interval_type, str) and interval_type in ['SNPCENTER', 'PEAKCENTER']):
                raise ValueError(
                    f"Invalid interval type"
                )
            self.ds = self.ds[self.ds['interval_type'] == interval_type].reset_index(drop = True)
        
        # Set up FASTA reference file for sequence extraction
        hg19_path, hg38_path = self._setup_fasta_file()

        # Load FASTA reference genome
        try:
            self.hg19 = pyfaidx.Fasta(hg19_path)
        except Exception as e:
            raise IOError(f"Error loading FASTA file {hg19_path}: {str(e)}") from e

        try:
            self.hg38 = pyfaidx.Fasta(hg38_path)
        except Exception as e:
            raise IOError(f"Error loading FASTA file {hg38_path}: {str(e)}") from e
        

        # Extract alternative sequences (with SNP/varaint)
        self.ds["seq_alt"] = self.ds.apply(
            lambda row: self.get_sequence(
                hg=row.hg,
                chromosome=row.chromosome,
                length=self.length,
                snp_pos=row.snp_position,
                int_center=row.interval_center,
                ref=row.ref,
                alt=row.alt,
            ),
            axis=1,
        )

        # Extract reference sequences (without variant)
        self.ds["seq_ref"] = self.ds.apply(
            lambda row: self.get_sequence(
                hg=row.hg,
                chromosome=row.chromosome,
                length=self.length,
                snp_pos=row.snp_position,
                int_center=row.interval_center,
                ref=row.ref,
                alt=row.ref,  # Use reference allele instead of alternative
            ),
            axis=1,
        )

        # Identifier for split information
        target_column = [f'logFC_{ct}' for ct in all_cells] if cell_types else [col for col in self.ds.columns if 'logFC' in col]
        fdr_column = [f'fdr_{ct}' for ct in all_cells] if cell_types else [col for col in self.ds.columns if 'fdr' in col]
        
        targets = self.ds[target_column].to_numpy()
        fdrs = self.ds[fdr_column].to_numpy()
        seq_alt = self.ds.seq_alt.to_numpy()
        seq_ref = self.ds.seq_ref.to_numpy()
        rev_pred = self.ds.reverse_prediction.to_numpy()

        self.ds = {"targets": targets, "seq": seq_ref, "seq_alt": seq_alt, "fdrs": fdrs, "reverse_prediction": rev_pred}

        self.name_for_split_info = self.prefix


    def __getitem__(self, idx):
        # Find all names start with 'seq' (e.g, 'seq', 'seq1', 'seq2', etc)
        seq_keys = [key for key in self.ds.keys() if key.startswith("seq")]

        seqs_datasets = {}
        for seq_key in seq_keys:
            sequence = self.ds[seq_key][idx]

            scals = (
                {name: sc[idx] for name, sc in self.scalars.items()}
                if hasattr(self, "scalars")
                else {}
            )
            vecs = (
                {name: vec[idx] for name, vec in self.vectors.items()}
                if hasattr(self, "vectors")
                else {}
            )

            Seq = seqobj(seq=sequence, scalars=scals, vectors=vecs, split=self.split)

            if self.transform is not None:
                Seq = self.transform(Seq)

            # Using original key name (seq, seq1, etc)
            seqs_datasets[seq_key] = Seq.seq

        target = torch.tensor(self.ds["targets"][idx].astype(np.float32))
        fdr = torch.tensor(self.ds["fdrs"][idx].astype(np.float32))
        rev_pred = torch.tensor(self.ds["reverse_prediction"][idx].astype(np.int8))

        if self.target_transform is not None:
            target = self.target_transform(target)

        if len(seqs_datasets) > 1:
            return seqs_datasets, target, fdr, rev_pred  # {seq : seq, seq1 : seq1, ..., targets, fdr}
        else:
            return seqs_datasets["seq"], target  # sequences, targets


    def filter_by_genomic_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe based on genomic regions using bioframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic data with columns:
            - 'chromosome': chromosome name (hg19)
            - 'position': variant position (0-based, hg19)

        Returns
        -------
        pd.DataFrame
            Filtered dataframe containing only sequences that overlap (or don't overlap)
            with the specified genomic regions

        Notes
        -----
        - Uses bioframe library for genomic interval operations
        - All genomic coordinates use hg38 assembly with 0-based indexing
        - Sequences are defined as regions centered on variant positions
        - Input regions should be provided in hg38 coordinates with 0-based indexing
        """
        if self.genomic_regions is None:
            return df

        # Prepare the genomic regions for bioframe
        if isinstance(self.genomic_regions, str):
            # Load from BED file
            regions_df = bf.read_table(self.genomic_regions, schema="bed")
            regions_df["chrom"] = regions_df["chrom"].astype(str)
        else:
            # Convert list of dicts to DataFrame
            regions_df = pd.DataFrame(self.genomic_regions)

        # Prepare our data for bioframe intersection
        # Create start and end positions based on the mutation position and desired length
        data_df = df.copy()
        half_length = self.length // 2
        
        # Calculate start and end positions for each sequence
        data_df["start"] = data_df["position"] - half_length
        data_df["end"] = data_df["position"] + half_length
        data_df["chrom"] = data_df["chromosome"]
        
        # Convert to integer if possible
        for col in ["start", "end"]:
            data_df[col] = pd.to_numeric(data_df[col], errors="coerce").astype("Int64")

        # Find intersections
        intersections = bf.overlap(data_df, regions_df, how="inner", return_index=True)
        
        if self.exclude_regions:
            # Exclude sequences that overlap with specified regions
            filtered_df = df[~df.index.isin(intersections["index"])]
        else:
            # Include only sequences that overlap with specified regions
            filtered_df = df[df.index.isin(intersections["index"])]

        return filtered_df
        
    def _setup_fasta_file(self) -> str:
        """
        Ensure FASTA file exists and is ready for use.

        Returns
        -------
        str
            Path to the FASTA file

        Raises
        ------
        IOError
            If the FASTA file cannot be downloaded or decompressed

        Notes
        -----
        - Downloads hg38 reference genome from UCSC if not present
        - Uses 0-based coordinate system for sequence extraction
        """

        hg19_path = os.path.join(self._data_path, "hg19.fa")
        hg38_path = os.path.join(self._data_path, "hg38.fa")

        if not os.path.exists(hg19_path):
            url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
            try:
                # Download and decompress
                subprocess.run(["wget", url, "-O", f"{hg19_path}.gz"], check=True)
                subprocess.run(["gunzip", hg19_path + ".gz"], check=True)
            except subprocess.CalledProcessError as e:
                raise IOError(
                    f"Failed to download/decompress FASTA file: {str(e)}"
                ) from e
        else:
            # print("FASTA file already exists. Skipping download.")
            pass

        if not os.path.exists(hg38_path):
            url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
            try:
                # Download and decompress
                subprocess.run(["wget", url, "-O", f"{hg38_path}.gz"], check=True)
                subprocess.run(["gunzip", hg38_path + ".gz"], check=True)
            except subprocess.CalledProcessError as e:
                raise IOError(
                    f"Failed to download/decompress FASTA file: {str(e)}"
                ) from e
        else:
            # print("FASTA file already exists. Skipping download.")
            pass

        return hg19_path, hg38_path

    def get_sequence(
        self, hg: str, chromosome: str, length: int, snp_pos: int, int_center: int, ref: str, alt: str
    ) -> str:
        """
        Extract sequence from a FASTA file with padding to a fixed length.

        Parameters
        ----------
        hg : str
            genome version to get the sequence (hg38 or hg19)
        chromosome : str
            Chromosome name (without 'chr' prefix, will be added automatically)
        length : int
            Total length of sequence to extract
        snp_pos : int
            Variant position (0-based)
        int_cnter: int
            Interval center position (0-based)
        ref : str
            Reference allele (single character)
        alt : str
            Alternative allele (single character or '-' for deletion)

        Returns
        -------
        str
            Extracted sequence with variant incorporated

        Raises
        ------
        ValueError
            - If reference nucleotide doesn't match expected
            - If sequence extraction fails

        Notes
        -----
        - Uses hg19 or hg38 reference genome with 0-based coordinates
        - Sequences are centered on the interval center
        - Handles both substitutions and deletions
        - For deletions, the sequence length is maintained by removing the deleted base
        """
        # Input validation
        if not isinstance(ref, str) or len(ref) != 1:
            raise ValueError(
                f"Reference nucleotide should be single character, got {ref}"
            )

        if hg == 'hg38':
            ref_genome = self.hg38
        elif hg == 'hg19':
            ref_genome = self.hg19
        else:
            raise ValueError(
                f'Invalid genome version (must be hg38 or hg19)'
            )

        half_len = length // 2
        
        if length % 2 == 1:  
            start = int_center - half_len - 1
            end = int_center + half_len 
            ref_pos_in_seq = snp_pos - start 
        else:  
            start = int_center - half_len
            end = int_center + half_len
            ref_pos_in_seq = snp_pos - start

        if snp_pos < start or snp_pos > end:  # the SNP lies OUTSIDE of the interval 
            return None

        # Verify reference nucleotide matches expected
        observed_ref = str(ref_genome[chromosome][snp_pos-1 : snp_pos]).upper()
        if observed_ref != ref.upper():
            raise ValueError(
                        f"no matching: expected {observed_ref}, found {ref}"
                            )

        try:
            if alt == "-":
                # Handle deletion
                seq = str(ref_genome[chromosome][start : end + 1])
                modified_seq = seq[:ref_pos_in_seq] + seq[ref_pos_in_seq + 1 :]
            else:
                # Handle substitution or insertion
                seq = str(ref_genome[chromosome][start : end])
                modified_seq = seq[:ref_pos_in_seq] + alt + seq[ref_pos_in_seq + 1:]

            return modified_seq.upper()
        except Exception as e:
            raise ValueError(
                f"Error processing {chromosome}:{snp_pos}-{ref}>{alt}: {str(e)}"
            ) from e







class ChenSingleDataset(MpraDataset):
    """
    Dataset class for Chen MPRA (Massively Parallel Reporter Assay) data.
    
    This class handles loading, filtering, and processing of genomic sequence data
    from the Chen et al. study, which contains information about 599 non-coding 
    Late-onset Alzheimer's disease associated SNPs/variants, tested in 855 
    constructs across multiple cell types.

    The dataset uses human genome assembly hg38 and hg19 with 0-based coordinate indexing.                         
    All genomic positions (start, end) follow 0-based indexing convention.

    Inherits from:
        MpraDataset: Base class for MPRA datasets

    Constants:
        FLAG (str): Dataset identifier flag: 'Chen'
        CELL_TYPE (dict): Mapping of cell types to their corresponding states

    Examples:
        >>> # Load data for specific cell type
        >>> dataset = ChenSingleDataset(cell_type='THP1')
        >>> 
        >>> # Load data for specific cell type and state
        >>> dataset = ChenSingleDataset(cell_type='INFG')
        >>> 
        >>> # Load data with custom sequence length
        >>> dataset = ChenSingleDataset(length=145, cell_type='THP1')
        >>> 
        >>> # Load data filtered by genomic regions
        >>> dataset = ChenSingleDataset(
        ...     genomic_regions='path/to/regions.bed',
        ...     cell_type='THP1']
        ... )
    """
    
    FLAG = "Chen"
    
    # Mapping of elements to their corresponding cell types
    CELL_TYPES = {'THP1': ['aggregated', 'Naive', 'IFNB', 'IFNG', 'LPSIFNG'],
                 'HMC3': ['aggregated','Naive', 'IFNB', 'IFNG', 'LPSIFNG'],
                 'Brain': ['aggregated','Cortex', 'Hippocampus', 'Striatum']}

    BASE_COLUMNS = ['RSID', 'interval_type', 'chromosome', 'pos_hg38', 'ref', 'alt', 'hg', 'snp_position', 'interval_center', 'reverse_prediction']

    def __init__(
        self,
        cell_type: str,
        split: str = "test",
        length: int = 227,  # length of cutted sequence
        state: str = 'aggregated',
        interval_type: str = None,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize the Chen MPRA dataset.
        
        Attributes
        ----------
        split : str, optional
            Specifies how to split the data. Currently only "test" is supported.
            Default is "test".
        length : int, optional  
            Length of the sequence for the differential expression experiment. 
            Must be positive integer. Default is 227.
        cell_type : str
            List of cell types to filter by.
            Can be a single string or list of strings.
        state : str, optional
            List of states to be used for each cell type. 
        interval_type : str, optional
            Type of intervals to be used from the assay.
            Can be 'SNPCENTER' with SNP in the center
            or 'PEAKCENTER' with open chromatin peak summit in the center.
        genomic_regions : str | List[Dict], optional
            Genomic regions to include/exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool
            If True, exclude the specified regions instead of including them.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data (expression values).
        root : str, optional
            Root directory where data is stored. If None, uses default data directory.
        """
        # Initialize parent class
        super().__init__(split, root)

        self.transform = transform
        self.target_transform = target_transform
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions
        self.prefix = self.FLAG  # Prefix for file names
        
        # Validate promoter-enhancer input
        if not (isinstance(cell_type, str) and cell_type in self.CELL_TYPES):
            raise ValueError("Invalid cell type")

        if not (isinstance(state, str) and state in self.CELL_TYPES[cell_type]):
            raise ValueError(f"Invalid state for {cell_type}")

        # Validate sequence length parameter
        if not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"Parameter 'length' must be natural integer, not {length}."
            )
        
        self.length = length

        try:
            # Load the data file
            file_name = self.prefix + "_" + cell_type + '_' + state + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process data - ensure proper chromosome formatting
        df.chromosome = df.chromosome.astype(str)
        df.snp_position = df.snp_position.astype(int)
        df.interval_center = df.interval_center.astype(int)
        self.ds = df

        if self.genomic_regions is not None:
            self.ds = self.filter_by_genomic_regions(self.ds)
        
        if interval_type:
            if not (isinstance(interval_type, str) and interval_type in ['SNPCENTER', 'PEAKCENTER']):
                raise ValueError(
                    f"Invalid interval type"
                )
            self.ds = self.ds[self.ds['interval_type'] == interval_type].reset_index(drop = True)
        
        # Set up FASTA reference file for sequence extraction
        hg19_path, hg38_path = self._setup_fasta_file()

        # Load FASTA reference genome
        try:
            self.hg19 = pyfaidx.Fasta(hg19_path)
        except Exception as e:
            raise IOError(f"Error loading FASTA file {hg19_path}: {str(e)}") from e

        try:
            self.hg38 = pyfaidx.Fasta(hg38_path)
        except Exception as e:
            raise IOError(f"Error loading FASTA file {hg38_path}: {str(e)}") from e

        # Extract alternative sequences (with SNP/varaint)
        self.ds["seq_alt"] = self.ds.apply(
            lambda row: self.get_sequence(
                hg=row.hg,
                chromosome=row.chromosome,
                length=self.length,
                snp_pos=row.snp_position,
                int_center=row.interval_center,
                ref=row.ref,
                alt=row.alt,
            ),
            axis=1,
        )

        # Extract reference sequences (without variant)
        self.ds["seq_ref"] = self.ds.apply(
            lambda row: self.get_sequence(
                hg=row.hg,
                chromosome=row.chromosome,
                length=self.length,
                snp_pos=row.snp_position,
                int_center=row.interval_center,
                ref=row.ref,
                alt=row.ref,  # Use reference allele instead of alternative
            ),
            axis=1,
        )

        # Identifier for split information
        target_column = 'logFC'
        fdr_column = 'fdr'
        targets = self.ds[target_column].to_numpy()
        fdrs = self.ds[fdr_column].to_numpy()
        seq_alt = self.ds.seq_alt.to_numpy()
        seq_ref = self.ds.seq_ref.to_numpy()
        rev_pred = self.ds.reverse_prediction.to_numpy()

        self.ds = {"targets": targets, "seq": seq_ref, "seq_alt": seq_alt, "fdrs": fdrs, "reverse_prediction": rev_pred}

        self.name_for_split_info = self.prefix
    def __getitem__(self, idx):
        # Find all names start with 'seq' (e.g, 'seq', 'seq1', 'seq2', etc)
        seq_keys = [key for key in self.ds.keys() if key.startswith("seq")]

        seqs_datasets = {}
        for seq_key in seq_keys:
            sequence = self.ds[seq_key][idx]

            scals = (
                {name: sc[idx] for name, sc in self.scalars.items()}
                if hasattr(self, "scalars")
                else {}
            )
            vecs = (
                {name: vec[idx] for name, vec in self.vectors.items()}
                if hasattr(self, "vectors")
                else {}
            )

            Seq = seqobj(seq=sequence, scalars=scals, vectors=vecs, split=self.split)

            if self.transform is not None:
                Seq = self.transform(Seq)

            # Using original key name (seq, seq1, etc)
            seqs_datasets[seq_key] = Seq.seq

        target = torch.tensor(self.ds["targets"][idx].astype(np.float32))
        fdr = torch.tensor(self.ds["fdrs"][idx].astype(np.float32))
        rev_pred = torch.tensor(self.ds["reverse_prediction"][idx].astype(np.int8))

        if self.target_transform is not None:
            target = self.target_transform(target)

        if len(seqs_datasets) > 1:
            return seqs_datasets, target, fdr, rev_pred  # {seq : seq, seq1 : seq1, ..., targets, fdr}
        else:
            return seqs_datasets["seq"], target  # sequences, targets


    def filter_by_genomic_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe based on genomic regions using bioframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic data with columns:
            - 'chromosome': chromosome name (hg19)
            - 'position': variant position (0-based, hg19)

        Returns
        -------
        pd.DataFrame
            Filtered dataframe containing only sequences that overlap (or don't overlap)
            with the specified genomic regions

        Notes
        -----
        - Uses bioframe library for genomic interval operations
        - All genomic coordinates use hg38 assembly with 0-based indexing
        - Sequences are defined as regions centered on variant positions
        - Input regions should be provided in hg38 coordinates with 0-based indexing
        """
        if self.genomic_regions is None:
            return df

        # Prepare the genomic regions for bioframe
        if isinstance(self.genomic_regions, str):
            # Load from BED file
            regions_df = bf.read_table(self.genomic_regions, schema="bed")
            regions_df["chrom"] = regions_df["chrom"].astype(str)
        else:
            # Convert list of dicts to DataFrame
            regions_df = pd.DataFrame(self.genomic_regions)

        # Prepare our data for bioframe intersection
        # Create start and end positions based on the mutation position and desired length
        data_df = df.copy()
        half_length = self.length // 2
        
        # Calculate start and end positions for each sequence
        data_df["start"] = data_df["position"] - half_length
        data_df["end"] = data_df["position"] + half_length
        data_df["chrom"] = data_df["chromosome"]
        
        # Convert to integer if possible
        for col in ["start", "end"]:
            data_df[col] = pd.to_numeric(data_df[col], errors="coerce").astype("Int64")

        # Find intersections
        intersections = bf.overlap(data_df, regions_df, how="inner", return_index=True)
        
        if self.exclude_regions:
            # Exclude sequences that overlap with specified regions
            filtered_df = df[~df.index.isin(intersections["index"])]
        else:
            # Include only sequences that overlap with specified regions
            filtered_df = df[df.index.isin(intersections["index"])]

        return filtered_df
        
    def _setup_fasta_file(self) -> str:
        """
        Ensure FASTA file exists and is ready for use.

        Returns
        -------
        str
            Path to the FASTA file

        Raises
        ------
        IOError
            If the FASTA file cannot be downloaded or decompressed

        Notes
        -----
        - Downloads hg38 reference genome from UCSC if not present
        - Uses 0-based coordinate system for sequence extraction
        """

        hg19_path = os.path.join(self._data_path, "hg19.fa")
        hg38_path = os.path.join(self._data_path, "hg38.fa")

        if not os.path.exists(hg19_path):
            url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
            try:
                # Download and decompress
                subprocess.run(["wget", url, "-O", f"{hg19_path}.gz"], check=True)
                subprocess.run(["gunzip", hg19_path + ".gz"], check=True)
            except subprocess.CalledProcessError as e:
                raise IOError(
                    f"Failed to download/decompress FASTA file: {str(e)}"
                ) from e
        else:
            # print("FASTA file already exists. Skipping download.")
            pass

        if not os.path.exists(hg38_path):
            url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
            try:
                # Download and decompress
                subprocess.run(["wget", url, "-O", f"{hg38_path}.gz"], check=True)
                subprocess.run(["gunzip", hg38_path + ".gz"], check=True)
            except subprocess.CalledProcessError as e:
                raise IOError(
                    f"Failed to download/decompress FASTA file: {str(e)}"
                ) from e
        else:
            # print("FASTA file already exists. Skipping download.")
            pass

        return hg19_path, hg38_path


    def get_sequence(
        self, hg: str, chromosome: str, length: int, snp_pos: int, int_center: int, ref: str, alt: str
    ) -> str:
        """
        Extract sequence from a FASTA file with padding to a fixed length.

        Parameters
        ----------
        hg : str
            genome version to get the sequence (hg38 or hg19)
        chromosome : str
            Chromosome name (without 'chr' prefix, will be added automatically)
        length : int
            Total length of sequence to extract
        snp_pos : int
            Variant position (0-based)
        int_cnter: int
            Interval center position (0-based)
        ref : str
            Reference allele (single character)
        alt : str
            Alternative allele (single character or '-' for deletion)

        Returns
        -------
        str
            Extracted sequence with variant incorporated

        Raises
        ------
        ValueError
            - If reference nucleotide doesn't match expected
            - If sequence extraction fails

        Notes
        -----
        - Uses hg19 or hg38 reference genome with 0-based coordinates
        - Sequences are centered on the interval center
        - Handles both substitutions and deletions
        - For deletions, the sequence length is maintained by removing the deleted base
        """
        # Input validation
        if not isinstance(ref, str) or len(ref) != 1:
            raise ValueError(
                f"Reference nucleotide should be single character, got {ref}"
            )

        if hg == 'hg38':
            ref_genome = self.hg38
        elif hg == 'hg19':
            ref_genome = self.hg19
        else:
            raise ValueError(
                f'Invalid genome version (must be hg38 or hg19)'
            )

        half_len = length // 2
        
        if length % 2 == 1:  
            start = int_center - half_len - 1
            end = int_center + half_len 
            ref_pos_in_seq = snp_pos - start 
        else:  
            start = int_center - half_len
            end = int_center + half_len
            ref_pos_in_seq = snp_pos - start

        if snp_pos < start or snp_pos > end:  # the SNP lies OUTSIDE of the interval 
            return None

        # Verify reference nucleotide matches expected
        observed_ref = str(ref_genome[chromosome][snp_pos-1 : snp_pos]).upper()
        if observed_ref != ref.upper():
            raise ValueError(
                        f"no matching: expected {observed_ref}, found {ref}"
                            )

        try:
            if alt == "-":
                # Handle deletion
                seq = str(ref_genome[chromosome][start : end + 1])
                modified_seq = seq[:ref_pos_in_seq] + seq[ref_pos_in_seq + 1 :]
            else:
                # Handle substitution or insertion
                seq = str(ref_genome[chromosome][start : end])
                modified_seq = seq[:ref_pos_in_seq] + alt + seq[ref_pos_in_seq + 1:]

            return modified_seq.upper()
        except Exception as e:
            raise ValueError(
                f"Error processing {chromosome}:{snp_pos}-{ref}>{alt}: {str(e)}"
            ) from e
