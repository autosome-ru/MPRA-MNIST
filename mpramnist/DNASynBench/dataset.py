from random import shuffle, seed, choices, randint
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from mpramnist.mpradataset import MpraDataset
from DNA_model import MarkovDNA


class SimpleMotifDataset(MpraDataset):
    """
    Binary classification task to find the motif in the sequence.
    """
    FLAG = "DNASynBench"

    def __init__(
        self,
        split=None,
        transform=None,
        target_transform=None
        ):
        """
        Initialize Dataset instance.
        
        Attributes
        ----------
        split : str
            Defines which data split to use. Must be one of: 'train', 'val', 'test'.
            The dataset filters sequences based on the 'split' column in the data file.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        
        super().__init__(split)

        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        self.ds = {"targets": [], "seq": []}
        self.name_for_split_info = self.prefix
        self.n_classes = 2
        
    def generate(self,
        motif: str,
        length=200,
        n_seqs=10000,
        ratio=0.2,
        train_size=0.7,
        random_state=42):
        
        seed(random_state)
        n_pos = int(n_seqs * ratio)
        n_neg = n_seqs - n_pos
        test_size = (1 - train_size) / 2
        seqs = []
        pos_model = MarkovDNA.load("promoter_model.pkl")
        neg_model = MarkovDNA.load("cds_model.pkl")
        with tqdm(total=n_seqs) as pbar:
            for i in range(n_pos):
                while True:
                    seq = pos_model.generate(length)
                    insert = randint(0, length - len(motif))
                    seq = seq[:insert] + motif + seq[insert+len(motif):]
                    if seq.count(motif) == 1:
                        seqs.append((seq, 1))
                        pbar.update(1)
                        break
            for i in range(n_neg):
                while True:
                    seq = neg_model.generate(length)
                    if motif not in seq:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
        shuffle(seqs)
        df = pd.DataFrame(seqs, columns=['sequence', 'target'])
        group = ['train'] * int(n_seqs*train_size) + ['val'] * int(n_seqs*test_size) + ['test'] * int(n_seqs*test_size)
        df['split'] = group
        self.df = df

    def get_split(self, split_name: str, transform=None, target_transform=None):
        """
        Extract data from dataset only for specified split.
        
        Args:
            split_name (str): 'train', 'val' or 'test'
        
        Returns:
            Dataset: new dataset with splitted df
        """
        
        splitted_df = self.df[self.df['split'] == split_name].copy()
        new_dataset = SimpleMotifDataset(split=split_name,
                                         transform=transform,
                                         target_transform=target_transform)
        new_dataset.df = splitted_df
        targets = new_dataset.df.target.to_numpy()
        seq = new_dataset.df.sequence.to_numpy()
        new_dataset.ds = {"targets": targets, "seq": seq}
        new_dataset.name_for_split_info = new_dataset.prefix
        return new_dataset


class LinCoopDataset(MpraDataset):
    """
    Regression task that assume that the activity of a sequence depends linearly on the number of motifs.
    """
    FLAG = "DNASynBench"

    def __init__(
        self,
        split=None,
        transform=None,
        target_transform=None
        ):
        
        super().__init__(split)

        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        self.ds = {"targets": [], "seq": []}
        self.name_for_split_info = self.prefix
        
    def generate(self,
        motif: str,
        length=200,
        n_seqs=10000,
        min_num=0,
        max_num=5,
        train_size=0.7,
        random_state=42):
        
        seed(random_state)
        test_size = (1 - train_size) / 2
        seqs = []
        pos_model = MarkovDNA.load("promoter_model.pkl")
        with tqdm(total=n_seqs) as pbar:
            for i in range(n_seqs):
                while True:
                    n_motifs = randint(min_num, max_num)
                    seq = pos_model.generate(length)
                    insert = -len(motif)
                    if n_motifs != 0:
                        section = (length - len(motif)) // n_motifs
                        for k in range(n_motifs):
                            insert = randint(insert+len(motif), (k+1)*section)
                            seq = seq[:insert] + motif + seq[insert+len(motif):]
                    if seq.count(motif) == n_motifs:
                        seqs.append((seq, n_motifs/max_num))
                        pbar.update(1)
                        break
        shuffle(seqs)
        df = pd.DataFrame(seqs, columns=['sequence', 'target'])
        group = ['train'] * int(n_seqs*train_size) + ['val'] * int(n_seqs*test_size) + ['test'] * int(n_seqs*test_size)
        df['split'] = group
        self.df = df

    def get_split(self, split_name: str, transform=None, target_transform=None):        
        splitted_df = self.df[self.df['split'] == split_name].copy()
        new_dataset = LinCoopDataset(split=split_name,
                                     transform=transform,
                                     target_transform=target_transform)
        new_dataset.df = splitted_df
        targets = new_dataset.df.target.to_numpy()
        seq = new_dataset.df.sequence.to_numpy()
        new_dataset.ds = {"targets": targets, "seq": seq}
        new_dataset.name_for_split_info = new_dataset.prefix
        return new_dataset


class NonlinCoopDataset(MpraDataset):
    """
    Regression task that assume that the activity of a sequence depends non-linearly on the number of motifs.
    """
    FLAG = "DNASynBench"

    def __init__(
        self,
        split=None,
        transform=None,
        target_transform=None
        ):
        
        super().__init__(split)

        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        self.ds = {"targets": [], "seq": []}
        self.name_for_split_info = self.prefix
        
    def generate(self,
        motif: str,
        length=200,
        n_seqs=10000,
        min_num=0,
        max_num=5,
        train_size=0.7,
        random_state=42):
        
        seed(random_state)
        test_size = (1 - train_size) / 2
        seqs = []
        pos_model = MarkovDNA.load("promoter_model.pkl")
        with tqdm(total=n_seqs) as pbar:
            for i in range(n_seqs):
                while True:
                    n_motifs = randint(min_num, max_num)
                    seq = pos_model.generate(length)
                    insert = -len(motif)
                    if n_motifs != 0:
                        section = (length - len(motif)) // n_motifs
                        for k in range(n_motifs):
                            insert = randint(insert+len(motif), (k+1)*section)
                            seq = seq[:insert] + motif + seq[insert+len(motif):]
                    if seq.count(motif) == n_motifs:
                        seqs.append((seq, activity(n_motifs, max_num)))
                        pbar.update(1)
                        break
        shuffle(seqs)
        df = pd.DataFrame(seqs, columns=['sequence', 'target'])
        group = ['train'] * int(n_seqs*train_size) + ['val'] * int(n_seqs*test_size) + ['test'] * int(n_seqs*test_size)
        df['split'] = group
        self.df = df

    def get_split(self, split_name: str, transform=None, target_transform=None):        
        splitted_df = self.df[self.df['split'] == split_name].copy()
        new_dataset = NonlinCoopDataset(split=split_name,
                                        transform=transform,
                                        target_transform=target_transform)
        new_dataset.df = splitted_df
        targets = new_dataset.df.target.to_numpy()
        seq = new_dataset.df.sequence.to_numpy()
        new_dataset.ds = {"targets": targets, "seq": seq}
        new_dataset.name_for_split_info = new_dataset.prefix
        return new_dataset


class AlienDataset(MpraDataset):
    """
    Binary classification task to find the target motif in the presence of another alien motif that does not affect activity.
    """
    FLAG = "DNASynBench"

    def __init__(
        self,
        split=None,
        transform=None,
        target_transform=None
        ):
        
        super().__init__(split)

        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        self.ds = {"targets": [], "seq": []}
        self.name_for_split_info = self.prefix
        self.n_classes = 2
        
    def generate(self,
        motif: str,
        alien: str,
        length=200,
        n_seqs=10000,
        ratio=0.2,
        rat_al=0.2,
        train_size=0.7,
        random_state=42):
        
        seed(random_state)
        n_mix = int(n_seqs * ratio * rat_al)
        n_pos = int(n_seqs * ratio) - n_mix
        n_al = int(n_seqs * rat_al) - n_mix
        n_neg = n_seqs - n_pos - n_mix - n_al
        max_len = max(len(motif), len(alien))
        test_size = (1 - train_size) / 2
        seqs = []
        pos_model = MarkovDNA.load("promoter_model.pkl")
        neg_model = MarkovDNA.load("cds_model.pkl")
        with tqdm(total=n_seqs) as pbar:
            for i in range(n_pos):
                while True:
                    seq = pos_model.generate(length)
                    insert = randint(0, length - len(motif))
                    seq = seq[:insert] + motif + seq[insert+len(motif):]
                    if seq.count(motif) == 1:
                        seqs.append((seq, 1))
                        pbar.update(1)
                        break
            for i in range(n_al):
                while True:
                    seq = pos_model.generate(length)
                    insert = randint(0, length - len(alien))
                    seq = seq[:insert] + alien + seq[insert+len(alien):]
                    if seq.count(motif) == 0:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
            for i in range(n_mix):
                while True:
                    seq = pos_model.generate(length)
                    insert_1 = randint(0, length//2 - max_len)
                    insert_2 = randint(length//2, length - max_len)
                    inserts = [insert_1, insert_2]
                    shuffle(inserts)
                    seq = seq[:inserts[0]] + motif + seq[inserts[0]+len(motif):]
                    seq = seq[:inserts[1]] + alien + seq[inserts[1]+len(alien):]
                    if seq.count(motif) == 1:
                        seqs.append((seq, 1))
                        pbar.update(1)
                        break
            for i in range(n_neg):
                while True:
                    seq = neg_model.generate(length)
                    if motif not in seq and alien not in seq:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
        shuffle(seqs)
        df = pd.DataFrame(seqs, columns=['sequence', 'target'])
        group = ['train'] * int(n_seqs*train_size) + ['val'] * int(n_seqs*test_size) + ['test'] * int(n_seqs*test_size)
        df['split'] = group
        self.df = df

    def get_split(self, split_name: str, transform=None, target_transform=None):        
        splitted_df = self.df[self.df['split'] == split_name].copy()
        new_dataset = AlienDataset(split=split_name,
                                   transform=transform,
                                   target_transform=target_transform)
        new_dataset.df = splitted_df
        targets = new_dataset.df.target.to_numpy()
        seq = new_dataset.df.sequence.to_numpy()
        new_dataset.ds = {"targets": targets, "seq": seq}
        new_dataset.name_for_split_info = new_dataset.prefix
        return new_dataset

class CombinationDataset(MpraDataset):
    """
    Binary classification task that implies that the activity requires the presence of both target and alien motifs.
    """
    FLAG = "DNASynBench"

    def __init__(
        self,
        split=None,
        transform=None,
        target_transform=None
        ):
        
        super().__init__(split)

        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        self.ds = {"targets": [], "seq": []}
        self.name_for_split_info = self.prefix
        self.n_classes = 2
        
    def generate(self,
        motif: str,
        alien: str,
        length=200,
        n_seqs=10000,
        ratio=0.2,
        rat_al=0.2,
        train_size=0.7,
        random_state=42):
        
        seed(random_state)
        n_mix = int(n_seqs * ratio * rat_al)
        n_pos = int(n_seqs * ratio) - n_mix
        n_al = int(n_seqs * rat_al) - n_mix
        n_neg = n_seqs - n_pos - n_mix - n_al
        max_len = max(len(motif), len(alien))
        test_size = (1 - train_size) / 2
        seqs = []
        pos_model = MarkovDNA.load("promoter_model.pkl")
        neg_model = MarkovDNA.load("cds_model.pkl")
        with tqdm(total=n_seqs) as pbar:
            for i in range(n_pos):
                while True:
                    seq = pos_model.generate(length)
                    insert = randint(0, length - len(motif))
                    seq = seq[:insert] + motif + seq[insert+len(motif):]
                    if seq.count(alien) == 0:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
            for i in range(n_al):
                while True:
                    seq = pos_model.generate(length)
                    insert = randint(0, length - len(alien))
                    seq = seq[:insert] + alien + seq[insert+len(alien):]
                    if seq.count(motif) == 0:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
            for i in range(n_mix):
                while True:
                    seq = pos_model.generate(length)
                    insert_1 = randint(0, length//2 - max_len)
                    insert_2 = randint(length//2, length - max_len)
                    inserts = [insert_1, insert_2]
                    shuffle(inserts)
                    seq = seq[:inserts[0]] + motif + seq[inserts[0]+len(motif):]
                    seq = seq[:inserts[1]] + alien + seq[inserts[1]+len(alien):]
                    if seq.count(motif) == 1 and seq.count(alien) == 1:
                        seqs.append((seq, 1))
                        pbar.update(1)
                        break
            for i in range(n_neg):
                while True:
                    seq = neg_model.generate(length)
                    if motif not in seq and alien not in seq:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
        shuffle(seqs)
        df = pd.DataFrame(seqs, columns=['sequence', 'target'])
        group = ['train'] * int(n_seqs*train_size) + ['val'] * int(n_seqs*test_size) + ['test'] * int(n_seqs*test_size)
        df['split'] = group
        self.df = df

    def get_split(self, split_name: str, transform=None, target_transform=None):        
        splitted_df = self.df[self.df['split'] == split_name].copy()
        new_dataset = CombinationDataset(split=split_name,
                                         transform=transform,
                                         target_transform=target_transform)
        new_dataset.df = splitted_df
        targets = new_dataset.df.target.to_numpy()
        seq = new_dataset.df.sequence.to_numpy()
        new_dataset.ds = {"targets": targets, "seq": seq}
        new_dataset.name_for_split_info = new_dataset.prefix
        return new_dataset


class DistanceDataset(MpraDataset):
    """
    Binary classification task that suppose the activity only if the motifs are located at a close distance from each other.
    """
    FLAG = "DNASynBench"

    def __init__(
        self,
        split=None,
        transform=None,
        target_transform=None
        ):
        
        super().__init__(split)

        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        self.ds = {"targets": [], "seq": []}
        self.name_for_split_info = self.prefix
        self.n_classes = 2
        
    def generate(self,
        motif: str,
        alien: str,
        length=200,
        act_dist=10,
        n_seqs=10000,
        ratio=0.2,
        train_size=0.7,
        random_state=42):
        
        seed(random_state)
        assert length >= len(motif) + len(alien) + act_dist*3, 'length must be greater'
        n_near = n_far = int(n_seqs * ratio // 2)
        n_neg = n_seqs - n_near - n_far
        test_size = (1 - train_size) / 2
        seqs = []
        pos_model = MarkovDNA.load("promoter_model.pkl")
        neg_model = MarkovDNA.load("cds_model.pkl")
        with tqdm(total=n_seqs) as pbar:
            for i in range(n_near):
                while True:
                    seq = pos_model.generate(length)
                    dist = pos_model.generate(randint(0, act_dist))
                    insert = randint(0, length - len(motif) - len(alien) - len(dist))
                    seq = seq[:insert] + motif + dist + alien + seq[insert+len(motif)+len(alien)+len(dist):]
                    if len(dist) <= act_dist:
                        seqs.append((seq, 1))
                        pbar.update(1)
                        break
            for i in range(n_far):
                while True:
                    seq = pos_model.generate(length)
                    dist = pos_model.generate(randint(act_dist+1, act_dist*3))
                    insert = randint(0, length - len(motif) - len(alien) - len(dist))
                    seq = seq[:insert] + motif + dist + alien + seq[insert+len(motif)+len(alien)+len(dist):]
                    if len(dist) >= act_dist:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
            for i in range(n_neg):
                while True:
                    seq = neg_model.generate(length)
                    if motif not in seq and alien not in seq:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
        shuffle(seqs)
        df = pd.DataFrame(seqs, columns=['sequence', 'target'])
        group = ['train'] * int(n_seqs*train_size) + ['val'] * int(n_seqs*test_size) + ['test'] * int(n_seqs*test_size)
        df['split'] = group
        self.df = df

    def get_split(self, split_name: str, transform=None, target_transform=None):        
        splitted_df = self.df[self.df['split'] == split_name].copy()
        new_dataset = DistanceDataset(split=split_name,
                                      transform=transform,
                                      target_transform=target_transform)
        new_dataset.df = splitted_df
        targets = new_dataset.df.target.to_numpy()
        seq = new_dataset.df.sequence.to_numpy()
        new_dataset.ds = {"targets": targets, "seq": seq}
        new_dataset.name_for_split_info = new_dataset.prefix
        return new_dataset

class OrderedDataset(MpraDataset):
    """
    Binary classification task that implies that the activity depends on order of target and alien motifs.
    """
    FLAG = "DNASynBench"

    def __init__(
        self,
        split=None,
        transform=None,
        target_transform=None
        ):
        
        super().__init__(split)

        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        self.ds = {"targets": [], "seq": []}
        self.name_for_split_info = self.prefix
        self.n_classes = 2
        
    def generate(self,
        motif: str,
        alien: str,
        length=200,
        n_seqs=10000,
        ratio=0.8,
        rat_al=0.8,
        train_size=0.7,
        random_state=42):
        
        seed(random_state)
        n_mix = int(n_seqs * ratio * rat_al)
        n_pos = int(n_seqs * ratio) - n_mix
        n_al = int(n_seqs * rat_al) - n_mix
        n_neg = n_seqs - n_pos - n_mix - n_al
        n_right = n_mix // 2
        n_back = n_mix - n_right
        test_size = (1 - train_size) / 2
        seqs = []
        pos_model = MarkovDNA.load("promoter_model.pkl")
        neg_model = MarkovDNA.load("cds_model.pkl")
        with tqdm(total=n_seqs) as pbar:
            for i in range(n_pos):
                while True:
                    seq = pos_model.generate(length)
                    insert = randint(0, length - len(motif))
                    seq = seq[:insert] + motif + seq[insert+len(motif):]
                    if seq.count(alien) == 0:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
            for i in range(n_al):
                while True:
                    seq = pos_model.generate(length)
                    insert = randint(0, length - len(alien))
                    seq = seq[:insert] + alien + seq[insert+len(alien):]
                    if seq.count(motif) == 0:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
            for i in range(n_right):
                while True:
                    seq = pos_model.generate(length)
                    insert_1 = randint(0, length//2 - len(motif))
                    insert_2 = randint(length//2, length - len(alien))
                    seq = seq[:insert_1] + motif + seq[insert_1+len(motif):]
                    seq = seq[:insert_2] + alien + seq[insert_2+len(alien):]
                    if seq.count(motif) == 1 and seq.count(alien) == 1:
                        seqs.append((seq, 1))
                        pbar.update(1)
                        break
            for i in range(n_back):
                while True:
                    seq = pos_model.generate(length)
                    insert_1 = randint(0, length//2 - len(alien))
                    insert_2 = randint(length//2, length - len(motif))
                    seq = seq[:insert_1] + alien + seq[insert_1+len(alien):]
                    seq = seq[:insert_2] + motif + seq[insert_2+len(motif):]
                    if seq.count(motif) == 1 and seq.count(alien) == 1:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
            for i in range(n_neg):
                while True:
                    seq = neg_model.generate(length)
                    if motif not in seq and alien not in seq:
                        seqs.append((seq, 0))
                        pbar.update(1)
                        break
        shuffle(seqs)
        df = pd.DataFrame(seqs, columns=['sequence', 'target'])
        group = ['train'] * int(n_seqs*train_size) + ['val'] * int(n_seqs*test_size) + ['test'] * int(n_seqs*test_size)
        df['split'] = group
        self.df = df

    def get_split(self, split_name: str, transform=None, target_transform=None):        
        splitted_df = self.df[self.df['split'] == split_name].copy()
        new_dataset = CombinationDataset(split=split_name,
                                         transform=transform,
                                         target_transform=target_transform)
        new_dataset.df = splitted_df
        targets = new_dataset.df.target.to_numpy()
        seq = new_dataset.df.sequence.to_numpy()
        new_dataset.ds = {"targets": targets, "seq": seq}
        new_dataset.name_for_split_info = new_dataset.prefix
        return new_dataset

'''def generation(length, gc_content, random_state):
    gc = int(length * gc_content)
    at = length - gc
    seqs = choices(['G', 'C'], k=gc) + choices(['A', 'T'], k=at)
    shuffle(seqs)
    return ''.join(seqs)'''

def activity(x, coef):
    return 1/(1 + np.exp(5-10*x/coef))
    #return -(2*x / coef - 1)**2 + 1
    #return -(0.4*x / coef - 0.2)**2 + 0.04
