# DNASynBench dataset

## Main Information

DNASynBench is a tool for creating benchmark datasets based on several regulatory genomics tasks and is designed to test models for their ability to capture biological patterns in data.

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/DNASynBench/mpramnist/DNASynBench/DNASynDataset_example.ipynb) for detailed usage example and training

## Tasks

| Number  | Name                        | Motifs per task | Type of task              |
|---------|-----------------------------|-----------------|---------------------------|
| task_1  | Motif presence              | 1               | Classification            |
| task_2  | Linear cooperativity        | variable        | Regression                |
| task_3  | Nonlinear cooperativity     | variable        | Regression                |
| task_4  | Alien motif                 | 2               | Classification            |
| task_5  | Motif combination           | 2               | Classification            |
| task_6  | Distance dependent activity | 2               | Classification            |
| task_7  | Order dependent activity    | 2               | Classification            |

### 1) Motif presence
Binary classification task to find the motif in the sequence.

**Settings:**
|  motif   | length | n_seqs | ratio |   train_size   | random_state |
|----------|--------|--------|-------|----------------|--------------|
| required |  200   | 10000  |  0.2  |       0.7      |      42      |

<img width="456" height="66" alt="1_eng" src="https://github.com/user-attachments/assets/51533320-a829-4f8f-ae2e-2a90b88d937b"/>

### 2, 3) Linear and Nonlinear cooperativity
Regression tasks that assume that the activity of a sequence depends linearly or non-linearly on the number of motifs.

**Settings:**
|  motif   | length | n_seqs | min_num | max_num |   train_size   | random_state |
|----------|--------|--------|---------|---------|----------------|--------------|
| required |  200   | 10000  |    0    |    5    |       0.7      |      42      |

<img width="461" height="114" alt="2_eng" src="https://github.com/user-attachments/assets/d4e7f039-3467-4ae1-92e1-536bb50c3a10" />

### 4) Alien motif
Binary classification task to find the target motif in the presence of another alien motif that does not affect activity, but can be recognized by models as a regularly occurring element.

**Settings:**
|  motif   |  alien   | length | n_seqs | ratio | rat_al |   train_size   | random_state |
|----------|----------|--------|--------|-------|--------|----------------|--------------|
| required | required |  200   | 10000  |  0.2  |  0.2   |       0.7      |      42      |

<img width="324" height="150" alt="4_eng" src="https://github.com/user-attachments/assets/eaeb83a7-c98f-479b-bd1d-cfaa256c002d" />

### 5) Motif combination
Binary classification task that implies that the activity requires the presence of both target and alien motifs.

**Settings:**
|  motif   |  alien   | length | n_seqs | ratio | rat_al |   train_size   | random_state |
|----------|----------|--------|--------|-------|--------|----------------|--------------|
| required | required |  200   | 10000  |  0.2  |  0.2   |       0.7      |      42      |

<img width="324" height="150" alt="5_eng" src="https://github.com/user-attachments/assets/a18a7275-2140-4b7d-88b4-dc90f8dbe7ac" />

### 6) Distance dependent activity
Binary classification task that suppose the activity only if the motifs are located at a close distance from each other. Maximum allowable distance is set as a parameter.

**Settings:**
|  motif   |  alien   | length | act_dist | n_seqs | ratio |   train_size   | random_state |
|----------|----------|--------|----------|--------|-------|----------------|--------------|
| required | required |  200   |    50    | 10000  |  0.2  |       0.7      |      42      |

<img width="457" height="104" alt="6_eng" src="https://github.com/user-attachments/assets/d755ea5d-f771-48d4-9e73-1944c5de20c0" />

### 7) Order dependent activity
Binary classification task that implies that the activity depends on order of first and second motifs.

**Settings:**
|  motif   |  alien   | length | n_seqs | ratio | rat_al |   train_size   | random_state |
|----------|----------|--------|--------|-------|--------|----------------|--------------|
| required | required |  200   | 10000  |  0.8  |  0.8   |       0.7      |      42      |

<img width="346" height="116" alt="7_eng" src="https://github.com/user-attachments/assets/780ab4b6-f4f6-4e51-a12b-4866870894ce" />

## Dataset parameters
#### **`split: str`**, optional
Defines which data split to use, one of: `'train'`, `'val'`, `'test'`.
It is used in `.generate()` method to extract necessary set.
#### **`transform: callable`**, optional
Transformation applied to each sequence.
#### **`target_transform: callable`**, optional
Transformation applied to the target data.

## Task parameters
#### **`motif; alien: str`**
A short DNA sequence that determines the observed value of activity in the intended target gene.  It can be a binding site of a transcription factor or a polymerase, a response element, etc. It must consist of nucleotides A, C, G, T.
#### **`length: int`**, optional, default=200
The expected length of all generated sequences in the dataset.
#### **`n_seqs: int`**, optional, default=10000
The expected number of all sequences in the dataset (including train, val and test).
#### **`act_dist: int`**, optional, default=10
The limiting distance between the motifs in task 6. When the motifs are located at a greater distance, their TF stop interacting and the sequence loses activity.
#### **`ratio, rat_al: float`**, optional, default=0.2
The frequency of occurrence of target and alien motifs, respectively. In tasks 1 and 4, value of `ratio` also means the percentage of the positive class.
#### **`min_num; max_num: int`**, optional, default=0; 5
The minimum and maximum number of motifs in regression tasks, respectively.
With min_num=0, activity tends to 0; max_num gives activity tends to 1.
#### **`train_size: float`**, optional, default=0.7
Fraction of training subset. Sizes of validation and testing sets are equal.
#### **`random_state: int`**, optional, default=42
A fixed random_state allows you to get reproducible datasets.

## Data Representation
```
                   sequence	                       target  split
TACAGATATGACACACAGATATTAGAGACACACAGGCACACCAAGAATAT	  0    train
CAGCCCCAAAGTAACGTCACTTTACCCAGGGGATATAGACAGAGACAGAT	 0.5    val
GAATCCTGTCACACAGAGGGATAAATACGAGGATTTTTGTCACAGATTCG	  1    test
...
```
**Interpretation**: 
  - Target = 1: If this sequence is located in the corresponding regulatory element (promoter, enhancer, etc.), then this construction demonstrate maximum activity and provide maximum expression of the downstream gene.
  - Target = 0: This sequence is background and has no regulatory activity.
  - Values between 0 and 1 (in regression tasks) indicate the intermediate level of expression of a potential target gene. This means that the regulatory structure contains binding motifs, but they are not sufficient for the full activity of the factors.

## Examples
### 1) Import important packages
```python
    from mpramnist.DNASynBench import SimpleMotifDataset
    from mpramnist.DNASynBench import LitModel_DNASyn
    from mpramnist import transforms as t
    import torch.utils.data as data
```

### 2) Transforms initialization
```python
    # Define set of transforms
    train_transform = t.Compose(
        [t.ReverseComplement(0.5),  # Apply reverse complement with 50% probability
            t.Seq2Tensor(),]
    )
    val_test_transform = t.Compose(
        [t.Seq2Tensor(),  # No reverse complement for validation/test
        ]
    )
```

### 3) Dataset creation
```python
    # Generating task
    dataset = SimpleMotifDataset()
    dataset.generate(motif='AGTCGACT',
        length=200,
        n_seqs=10000,
        train_size=0.7,
        random_state=42)

    # Extracting train, validation and testing subsets
    train_dataset = dataset.get_split("train", transform=train_transform)
    val_dataset = dataset.get_split("val", transform=val_test_transform)
    test_dataset = dataset.get_split("test", transform=val_test_transform)
```

### 4) Dataloader creation
```python 
    train_loader = data.DataLoader(
        dataset=train_dataset, 
        batch_size=1024, 
        shuffle=True,   # Shuffle for the training set
        num_workers=8
    )

    val_loader = data.DataLoader(
        dataset=val_dataset, 
        batch_size=1024, 
        shuffle=False,   # No need to shuffle for validation/testing
        num_workers=8
    )
```
