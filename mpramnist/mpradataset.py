import numpy as np
from typing import List, T, Union, Optional, Callable, Dict
import torch
import os
from .utils import md5_file
from torch.utils.data import Dataset
from .dataclass import seqobj, VectorDsFeature, ScalarDsFeature
from .info import INFO, HOMEPAGE, DEFAULT_ROOT


class MpraDataset(Dataset):
    """SEQUENCE DATASET."""

    """
        Dataset for working with sequences and their associated features.
        This dataset provides functionality to split the data, apply transformations, and 
    access cell-type-specific information.
        
        Attributes
        ----------
        split : str | List[int] | int
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        root : (string, optional)
            Root directory of dataset.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
    PARENT_FLAG = "MpraDaraset"

    def __init__(
        self,
        split: str | List[int] | int | List[Union[int, str]],
        root,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None
    ):
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self._scalars = {}
        self._vectors = {}

        if root is not None:
            self.root = os.path.join(root, self.FLAG)
            self._data_path = os.path.abspath(self.root)
            if not os.path.exists(self._data_path):
                os.makedirs(self._data_path)
        else:
            self.root = DEFAULT_ROOT
            self._data_path = os.path.join(DEFAULT_ROOT, self.FLAG)

        self.info = INFO[self.FLAG]

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
            seqs_datasets[seq_key] = Seq

        target = torch.tensor(self.ds["targets"][idx].astype(np.float32))
        
        if self.joint_transform is not None:
            for seq_key, sequence in seqs_datasets.items():
                Seq = self.joint_transform(sequence, target)
                seqs_datasets[seq_key] = Seq

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            for seq_key, sequence in seqs_datasets.items():
                Seq = self.transform(sequence)
                seqs_datasets[seq_key] = Seq

        for seq_key, sequence in seqs_datasets.items():
            seqs_datasets[seq_key] = sequence.seq

        if len(seqs_datasets) > 1:
            return seqs_datasets, target  # {seq : seq, seq1 : seq1, ..., targets}
        else:
            return seqs_datasets["seq"], target  # sequences, targets

    @property
    def scalars(self):
        return self._scalars

    @property
    def vectors(self):
        return self._vectors

    def add_numeric_scalar(self, name: str, val: List[T]):
        self._scalars[name] = ScalarDsFeature.numeric(val=val)

    def add_categorial_scalar(
        self, name: str, val: List[T], levels: Dict[T, int] | None = None
    ):
        self._scalars[name] = ScalarDsFeature.categorial(val=val, levels=levels)

    def add_numeric_vector(self, name: str, val: List[List[T]], pad_value: T):
        self._vectors[name] = VectorDsFeature.numeric(val=val, pad_value=pad_value)

    def add_categorial_vector(
        self,
        name: str,
        val: List[List[T]],
        pad_value: T,
        levels: Dict[T, int] | None = None,
    ):
        self._vectors[name] = VectorDsFeature.categorial(
            val=val, pad_value=pad_value, levels=levels
        )

    def __len__(self):
        return len(self.ds["seq"])

    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        name_of_split = self.name_for_split_info 
        head = f"Dataset {self.__class__.__name__} ({self.PARENT_FLAG})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Using split: {self.split}")
        if name_of_split + "split" in self.info:
            body.append(f"Split: {self.info[name_of_split + 'split']}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Description: {self.info['description']}")
        if name_of_split + "description" in self.info:
            body.append(f"{self.info[name_of_split + 'description']}")
        if 'licence'  in self.info:
            body.append(f"Licence: {self.info['licence']}")
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def download(self, file_path, file_name):
        candidate_file_path = os.path.join(file_path, file_name)
        if not os.path.exists(candidate_file_path):
            try:
                from torchvision.datasets.utils import download_url

                download_url(
                    url=self.info[f"url_{file_name}"],
                    root=file_path,
                    filename=file_name,
                    md5=self.info[f"MD5_{file_name}"],
                )
            except:
                raise RuntimeError(
                    f"""
                    Automatic download failed! Please download {file_name} manually.
                    1. [Optional] Check your network connection: 
                        Go to {HOMEPAGE} and find the Zenodo repository
                    2. Download the file from the Zenodo repository or its Zenodo data link: 
                        {self.info[f"url_{file_name}"]}
                    3. [Optional] Verify the MD5: 
                        {self.info[f"MD5_{file_name}"]}
                    4. Put the npz file under your MPRA-MNIST root folder: 
                        {file_path}
                    """
                )
        elif md5_file(candidate_file_path) != self.info[f"MD5_{file_name}"]:
            raise RuntimeError(
                f"""MD5 mismatch! Please remove the file and download it again:{file_name}. Also, you can download it manually.
                1. [Optional] Check your network connection: 
                    Go to {HOMEPAGE} and find the Zenodo repository
                2. Download the file from the Zenodo repository or its Zenodo data link: 
                    {self.info[f"url_{file_name}"]}
                3. [Optional] Verify the MD5: 
                    {self.info[f"MD5_{file_name}"]}
                4. Put the npz file under your MPRA-MNIST root folder: 
                    {file_path}""")