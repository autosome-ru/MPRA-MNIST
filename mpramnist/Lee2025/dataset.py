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

class LeeDataset(MpraDataset):
    """
    Dataset class for Lee MPRA (Massively Parallel Reporter Assay) data.
    
    This class handles loading, filtering, and processing of genomic sequence data
    from the Guo et al. study, which contains information about 13261 non-coding SNPs/variants, connected
    to eight psychiatric disorders. The experiment was conducted in human neural progenitor cells (HNPs)

    The dataset uses human genome assembly hg38 with 0-based coordinate indexing.                         
    All genomic positions (start, end) follow 0-based indexing convention.

    Inherits from:
        MpraDataset: Base class for MPRA datasets

    Constants:
        FLAG (str): Dataset identifier flag: 'Lee'

    Examples:
        >>> # Load default data 
        >>> dataset = LeeDataset()
        >>> 
        >>> # Load data with custom sequence length
        >>> dataset = GuoSingleDataset(length=200)
        >>> 
        >>> # Load data filtered by genomic regions
        >>> dataset = KircheGuoSingleDatasetrDataset(
        ...     genomic_regions='path/to/regions.bed',
        ... )
    """
    
    FLAG = "Lee"

    VARIANT_TYPE_MAPPING = {'emVar': 1, 'MPRA-Allelic': 2, 'MPRA-nonallelic': 3, 'Uncertain': 4}

    def __init__(
        self,
        split: str = "test",
        length: int = 150,  # length of cutted sequence
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize the Lee MPRA dataset.
        
        Attributes
        ----------
        split : str, optional
            Specifies how to split the data. Currently only "test" is supported.
            Default is "test".
        length : int, optional  
            Length of the sequence for the differential expression experiment. 
            Must be positive integer. Default is 150.
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
        self.prefix = self.FLAG + '_'  # Prefix for file names

        # Validate sequence length parameter
        if not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"Parameter 'length' must be natural integer, not {length}."
            )
        self.length = length

        try:
            # Load the data file
            file_name = self.prefix + 'SNP' + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process data - ensure proper chromosome formatting
        df.chromosome = df.chromosome.astype(str)
        df.position = df.position_hg38.astype(int)
        self.ds = df

        if self.genomic_regions is not None:
            self.ds = self.filter_by_genomic_regions(self.ds)
        
        # Set up FASTA reference file for sequence extraction
        fasta_file = self._setup_fasta_file()

        # Load FASTA reference genome
        try:
            ref = pyfaidx.Fasta(fasta_file)
        except Exception as e:
            raise IOError(f"Error loading FASTA file {fasta_file}: {str(e)}") from e                  
        
        # Extract alternative sequences (with SNP/varaint)
        self.ds["seq_alt"] = self.ds.apply(
            lambda row: self.get_sequence(
                ref_genome=ref,
                chromosome=row.chromosome,
                length=self.length,
                pos=row.position_hg38,
                ref=row.ref,
                alt=row.alt,
            ),
            axis=1,
        )

        # Extract reference sequences (without variant)
        self.ds["seq_ref"] = self.ds.apply(
            lambda row: self.get_sequence(
                ref_genome=ref,
                chromosome=row.chromosome,
                length=self.length,
                pos=row.position_hg38,
                ref=row.ref,
                alt=row.ref,  # Use reference allele instead of alternative
            ),
            axis=1,
        )

        self.ds.Variant_Class = self.ds.Variant_Class.map(self.VARIANT_TYPE_MAPPING).fillna(4).astype(int)

        # Identifier for split information
        target_column = 'MPRA_logFC'	
        var_type_column = 'Variant_Class'
        targets = self.ds[target_column].to_numpy()
        var_types = self.ds[var_type_column].to_numpy()
        seq_alt = self.ds.seq_alt.to_numpy()
        seq_ref = self.ds.seq_ref.to_numpy()
        rev_pred = self.ds.reverse_prediction.to_numpy()

        self.ds = {"targets": targets, "seq": seq_ref, "seq_alt": seq_alt, "var_types": var_types, "reverse_prediction": rev_pred}

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
        var_type = torch.tensor(self.ds["var_types"][idx].astype(np.int8))
        rev_pred = torch.tensor(self.ds["reverse_prediction"][idx].astype(np.int8))

        if self.target_transform is not None:
            target = self.target_transform(target)

        if len(seqs_datasets) > 1:
            return seqs_datasets, target, var_type, rev_pred  # {seq : seq, seq1 : seq1, ..., targets, fdr}
        else:
            return seqs_datasets["seq"], target  # sequences, targets


    def filter_by_genomic_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe based on genomic regions using bioframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic data with columns:
            - 'chromosome': chromosome name (hg38)
            - 'position': variant position (0-based, hg38)

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

        fasta_file = os.path.join(self._data_path, "hg38.fa")

        if not os.path.exists(fasta_file):
            url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
            try:
                # Download and decompress
                subprocess.run(["wget", url, "-O", f"{fasta_file}.gz"], check=True)
                subprocess.run(["gunzip", fasta_file + ".gz"], check=True)
            except subprocess.CalledProcessError as e:
                raise IOError(
                    f"Failed to download/decompress FASTA file: {str(e)}"
                ) from e
        else:
            # print("FASTA file already exists. Skipping download.")
            pass

        return fasta_file

    def get_sequence(
        self, ref_genome, chromosome: str, length: int, pos: int, ref: str, alt: str, 
    ) -> str:
        """
        Extract sequence from a FASTA file with padding to a fixed length.

        Parameters
        ----------
        ref_genome : pyfaidx.Fasta
            FASTA file object for sequence extraction
        chromosome : str
            Chromosome name (without 'chr' prefix, will be added automatically)
        length : int
            Total length of sequence to extract
        pos : int
            Variant position (0-based, hg38)
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
        - Uses hg19 reference genome with 0-based coordinates
        - Sequences are centered on the variant position
        - Handles both substitutions and deletions
        - For deletions, the sequence length is maintained by removing the deleted base
        """

        if not isinstance(ref, str) or len(ref) != 1:
            raise ValueError(
                f"Reference nucleotide should be single character, got {ref}"
            )

        # Verify reference nucleotide matches expected
        observed_ref = str(ref_genome[chromosome][pos-1 : pos]).upper()
        if observed_ref != ref.upper():
            return None

        half_len = length // 2
        
        if length % 2 == 1:  
            start = pos - half_len
            end = pos + half_len + 1
            ref_pos_in_seq = half_len  
        else:  
            start = pos - half_len
            end = pos + half_len
            ref_pos_in_seq = half_len

        try:
            ref_pos_in_seq = half_len

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
                f"Error processing {chromosome}:{pos}-{ref}>{alt}: {str(e)}"
            ) from e

