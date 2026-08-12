# Dream dataset

## Main Information

The Dream dataset is based on the **episomal** Random Promoter DREAM Challenge, where participants designed sequence-to-expression models trained on expression measurements of `promoters with random DNA sequences` ([Rafi et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10888977/)). The training data comprises `~6.7 million unique 80-bp random promoter sequences` tested in Saccharomyces cerevisiae. The input is a DNA sequence, and the output is a scalar `gene expression value` predicted from the sequence.

- The `training data` consists of random DNA sequences (as described in original studies).

- The `test data` includes specially designed subsets: High/Low expression, Native (yeast), Random, Challenging sequences, and paired datasets (SNVs, Motif Perturbation, Motif Tiling).

- Expression levels were quantified using MPRA (Massively Parallel Reporter Assay) and estimated using MAUDE software, using read abundance in sorting bins as input.

For neural network processing, sequences should be standardized to a fixed length. We recommend using constant flanking sequences from the original experimental system. Validation uses the public competition set, while testing uses the private held-out set.

See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/tree/main/mpramnist/Rafi2024) for detailed usage example and training

## Tasks

### Regression

The regression task involves predicting gene expression values from DNA sequences. In the DREAM Challenge, models received regulatory DNA sequences as input and predicted corresponding expression levels. The performance was evaluated across various test sequence types to understand model capabilities and limitations.

### Data Representation

```
===== Dream_train.tsv
    seq	            label
TCTTTGCCACGG...	    11.0
TATGTTGCGTTA...	    6.0
TGTGAAGAATAT...	    8.0
```

```
===== Dream_single.tsv
        seq	            label	            public	private	    high	low	    yeast	random	challenging	    all
TACCTTAGCGCTAAATG...	0.156331079044438	0	    1	        0	    0	    0	    0	        0	        1
ACGCACTCATCCGCTAT...	0.711709511132007	0	    1	        0	    0	    0	    0	        0	        1
ACGCACTCATCCGCTAT...	0.398692283648616	1	    0	        0	    0	    0	    0	        0	        1
```

```
===== Dream_paired.tsv
        seq	                seq_alt	            label_ref	        label_alt	        delta_measured	    public	 private    snv 	perturbation	tiling
TCATAGCGGTTACGGCTGTT...	TCATAGCGGTTACGGCTGTT...	-1.04089780020606	-0.911119035049065	0.12977876515699505	    0	    1	    1	        0	            0
AGCATATGGTTACGGCTGTT...	AGCATATGGTTACGGCTGTT...	-0.76009884467544	-0.888059185205019	-0.12796034052957905	0	    1	    1	        0	            0
GGTTGTGGGTTACGGCTGTT...	GGTTGTGGGTTACGGCTGTT...	-0.285989062488439	-0.185540625199234	0.10044843728920502	    0	    1	    1	        0	            0
```

## Parameters

### **`split : str`**

Data split specification. Valid values:
- `"train"`: Training data
- `"val"` or `"public"`: Validation/public test data  
- `"test"` or `"private"`: Private test data

### **`data_type : str | List[str]`**, optional

Specific dataset type(s) to load. For training split, this parameter is ignored.
- Single sequence types: `"high"`, `"low"`, `"yeast"`, `"random"`, `"challenging"`, `"all"`
- Paired sequence types: `"snv"`, `"perturbation"`, `"tiling"`

#### `Supported dataset types:`

- **`"all"`** : All sequences in the test data.
- **`"high"`** : Sequences designed to have high expression (Vaishnav et al. method).
- **`"low"`** : Sequences designed to have low expression.
- **`"yeast"`** : Sequences that are present in the yeast genome.
- **`"random"`** : Additional random sequences sampled from previous experiments.
- **`"challenging"`** : Sequences optimized via genetic algorithm to maximize differences between CNN and biochemical model predictions.
- **`snv`** : Single nucleotide variants, including mutation trajectories and random mutations.
- **`"perturbation"`** : Reb1 and Hsf1 transcription factor binding site perturbations.
- **`"tiling"`** : Systematic tiling of specific motifs (poly-A, Skn7, Mga1, Ume6, Mot3, Azf1).

*Note: The training set contains only random sequences from the N80 library.*

### **`transform : callable`**, optional

Transformation function applied to each sequence. Useful for data augmentation
or sequence encoding. Should accept a sequence string and return transformed data.

### **`target_transform : callable`**, optional 

Transformation function applied to target values. Useful for normalization or target processing.

### **`root : str`**, optional

Root directory for data storage. If `None`, uses default data directory.

## Data Handling Considerations

1) The `split` parameter allows you to select the data partition. Using `"val"` or `"public"` will load sequences from the public competition set. Using `"test"` or `"private"` will load data that was held out (private) during the competition.

2) The `data_type` parameter allows you to select the type of data to process. If you select one or more `"single"` dataset types (e.g., `"high"`, `"yeast"`), the task is to predict the regulatory activity of individual sequences. If you select one or more `"paired"` dataset types (e.g., `"snv"`, `"perturbation"`), the task is to predict the difference in activity (`label_alt - label_ref`) between the alternative and reference sequences.

3) For the training split, the `data_type` parameter is ignored, as the training set contains a general collection of sequences.

4) When loading paired data, the dataset returns tuples `(seq, seq_alt, delta_measured)`.

5) Using the `transform` argument is recommended to apply necessary preprocessing, such as adding constant flanking sequences and converting sequences to tensors.

6) **Example Usage**: See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/tree/main/mpramnist/Rafi2024) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    from mpramnist.Rafi2024 import RafiDataset
    from mpramnist.Rafi2024 import LitModel_Rafi

    import mpramnist.transforms as t

    from torch.utils.data import DataLoader

    length = 120
    plasmid = DreamDataset.PLASMID.upper()
    insert_start = plasmid.find("N" * 80)
    right_flank = DreamDataset.RIGHT_FLANK
    left_flank = plasmid[insert_start - length : insert_start]
```

### 2) Initialize transforms

```python

    train_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.LeftCrop(length, length),
            t.ReverseComplement(0.5), # Apply reverse complement with 50% probability for augmentation
            t.Seq2Tensor(),           # Convert sequence to tensor (e.g., one-hot)
        ]
    )
    val_test_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.LeftCrop(length, length),
            t.ReverseComplement(0),   # No augmentation for validation/test
            t.Seq2Tensor(),
        ]
    )
    
   
```

### 3) Dataset Creation

```python
    # Load training data (all sequences)
    train_dataset = RafiDataset(split="train", transform=train_transform)

    # Load validation data for high-activity sequences
    val_dataset = RafiDataset(split="val", data_type="high", transform=val_test_transform)

    # Load test data for SNV analysis
    test_dataset = RafiDataset(split="test", data_type="snv", transform=val_test_transform)

    # Load multiple dataset types for validation
    multi_dataset = RafiDataset(split="val", data_type=["high", "yeast"], transform=val_test_transform)
```

### 4) Dataloader Creation

```python 
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=1024, 
        shuffle=True, # True for training, False for val and test
        num_workers=8
    )

    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=1024, 
        shuffle=False, # Do not shuffle for validation and testing
        num_workers=8
    )
```

## Launch Parameters

```bash
#MPRALegNet
python3 Rafi_model_launch.py --model MPRALegNet --epoch_num 5 --runs 5 --result_dir ./rafi_legnet.tsv lr 1e-0.5 --wd 0.001
#Malinois
python3 Rafi_model_launch.py --model Malinois --epoch_num 30 --runs 5 --result_dir ./rafi_malinois.tsv
#MPRAnn
python3 Rafi_model_launch.py --model MPRAnn --epoch_num 30 --runs 5 --result_dir ./rafi_mprann.tsv
#PARM
python3 Rafi_model_launch.py --model PARM --epoch_num 30 --runs 5 --result_dir ./rafi_parm.tsv
#DREAM-RNN
python3 Rafi_model_launch.py --model DREAM-RNN --lr 0.005 --wd 0.01 --epoch_num 80 --runs 1 --result_dir ./rafi_dream_rnn_try_orig_params.tsv
```

## Achieved Performance Using Basic Models

Pearson correlation, r^2

Experiment | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM_RNN |
|:---------------:|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:| :--------------------:|
| All Sequences | 0,95 | 0,928 | 0,8521 | 0,7649 | 0,7741 | 0,95 |
| High | 0,51 | 0,285 | 0,0768 | 0,0378 | 0,0477 | 0,51 |
| Low | 0,51 | 0,152 | 0,2553 | 0,1109 | 0,1279 | 0,51 |
| Native | 0,78 | 0,745 | 0,6274 | 0,4144 | 0,407 | 0,78 |
| Random | 0,96 | 0,951 | 0,9238 | 0,8391 | 0,859 | 0,96 |
| Challenging | 0,94 | 0,909 | 0,7252 | 0,6494 | 0,7033 | 0,94 |
| SNVs | 0,73 | 0,688 | 0,5297 | 0,3366 | 0,3487 | 0,73 |
| Motif Perturbation | 0,96 | 0,945 | 0,8683 | 0,7209 | 0,7838 | 0,96 |
| Motif Tiling | 0,91 | 0,855 | 0,6644| 0,5916 | 0,5131 | 0,91 |

 

## Citation

When using this dataset, please cite the original publication:

[Rafi et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10888977/)

Rafi AM, Nogina D, Penzar D, Lee D, Lee D, Kim N, Kim S, Kim D, Shin Y, Kwak IY, Meshcheryakov G, Lando A, Zinkevich A, Kim BC, Lee J, Kang T, Vaishnav ED, Yadollahpour P; Random Promoter DREAM Challenge Consortium; Kim S, Albrecht J, Regev A, Gong W, Kulakovskiy IV, Meyer P, de Boer C. Evaluation and optimization of sequence-based gene regulatory deep learning models. bioRxiv [Preprint]. 2024 Feb 17:2023.04.26.538471. doi: 10.1101/2023.04.26.538471. Update in: Nat Biotechnol. 2025 Aug;43(8):1373-1383. doi: 10.1038/s41587-024-02414-w. PMID: 38405704; PMCID: PMC10888977.

```bibtex
    @article{rafi2023evaluation,
        title = {Evaluation and optimization of sequence-based gene regulatory deep learning models},
        url = {http://dx.doi.org/10.1101/2023.04.26.538471},
        DOI = {10.1101/2023.04.26.538471},
        publisher = {Cold Spring Harbor Laboratory},
        author = {Rafi, Abdul Muntakim and Nogina, Daria and Penzar, Dmitry and Lee, Dohoon and Lee, Danyeong and Kim, Nayeon and Kim, Sangyeup and Kim, Dohyeon and Shin, Yeojin and Kwak, Il-Youp and Meshcheryakov, Georgy and Lando, Andrey and Zinkevich, Arsenii and Kim, Byeong-Chan and Lee, Juhyun and Kang, Taein and Vaishnav, Eeshit Dhaval and Yadollahpour, Payman and {Random Promoter DREAM Challenge Consortium} and Kim, Sun and Albrecht, Jake and Regev, Aviv and Gong, Wuming and Kulakovskiy, Ivan V. and Meyer, Pablo and de Boer, Carl},
        year = {2023},
        month = apr
    }
```