# Arensbergen 2019 dataset

## Main information

The **episomal** Arensbergen's SuRE dataset (Survey of Regulatory Elements) is based on the analysis of genomes from 4 individuals from 4 different populations ([van Arensbergen et al. 2017](https://pubmed.ncbi.nlm.nih.gov/28024146/)) and was scaled up by [van Arensbergen et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6609452/). The genomes of these individuals are fragmented into 150–500 bp sequences, each cloned into a reporter plasmid. These fragments can drive expression if they contain a functional promoter. Approximately 2.4B and 1.2B fragments were tested (assayed) in K562 and HepG2 cells, respectively.

Preprocessed data and code were integrated from the work of [Reddy et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/) ([GitHub](https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/SuRE.py). Following their subsampling methodology, separate datasets are created for each individual. This subsampling controls for GC content and expression level distribution. The final datasets contain approximately 400-600K training sequences and 50-70K test and validation sequences per individual.

## Tasks

### Classification Task

The classification task involves predicting two **independent class labels (0 to 4)** for each sequence. These labels represent **expression bins** based on normalized read counts in K562 and HepG2 cell lines, respectively.

**Definition of Bins (for each cell line):**

- **Bin 0**: 0 reads.

- **Bin 1**: (0, 10] reads.

- **Bin 2**: (10, 20] reads.

- **Bin 3**: (20, 30] reads.

- **Bin 4**: 30+ reads.

Most sequences fall into Bin 0. The datasets are balanced across all 25 possible combinations of K562 and HepG2 bins.

```
    chr     start       end     strand      split       seq                      K562_bin	HepG2_bin
    ---------------------------------------------------------------------------------------------------------
    X	    130598969	130599266	-       train    ATAAGCTTTTTGA...            1           4
    6	    41888832	41889084	-	    train    GTTAGCTTCTCTCA...           4	         2
    16	    47945205	47945676	-	    train    CTTATGAAGCTTAG...           0 	         1
```

### Regression Task

The target variables for the regression task are **the average expression levels** of genomic elements in two cell lines, K562 and HepG2.

These averages are derived from the raw SuRE-seq count data.

Each element is associated with two continuous values representing its average regulatory activity in the two cell lines, which can then be used as prediction targets in regression models.

```
    chr     start       end     strand      split       seq                  avg_K562_exp  avg_HepG2_exp
    ---------------------------------------------------------------------------------------------------------
    3       80783739    80783991    -       train    TGGTTGCCCATTTT...           22.333      0.5
    9       73103533    73103851    +       train    GTAAAGTACTCAGT...           28.0	     3.5
    19      19483672	19484047    +       train    ACAAAAGACTCTGA...           22.666 	 6.5
```

See [Arensbergen Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Arensbergen2019/ArensbergenDataset_example.ipynb) for detailed information.

## Parameters

### **`split : str`**

Defines which split to use (e.g., `'train'`, `'val'`, `'test'`, or list of fold indices).

### **`genome_id : str`**

Identifier of the genome to use. Must be one of:

- "SuRE42_HG02601"
- "SuRE43_GM18983" 
- "SuRE44_HG01241"
- "SuRE45_HG03464"

Specifies which genomic dataset to load.

###  **task : str**

Type of machine learning task. Must be one of:
- `"classification"`: for multi-class classification tasks
- `"regression"`: for continuous value prediction tasks

Determines how target values are processed and interpreted.

### **`genomic_regions : str | List[Dict]`, optional**

Genomic regions to include/exclude. Can be:
- Uses hg19 reference genome
- Path to BED file
- List of dictionaries with `'chrom'`, `'start'`, `'end'` keys
- Uses 0-based indexing for genomic coordinates

### **`exclude_regions : bool`**

If `True`, exclude the specified regions instead of including them

### **`transform : callable`, optional**

Transformation applied to each sequence object.

### **`target_transform : callable`, optional**

Transformation applied to the target data.

### **`root : str`, optional**

Root directory where data is stored. If None, uses default data path.

## Data Handling Considerations

1) **Variable Sequence Lengths**: The main characteristic of this data is that sequence lengths vary. To handle this, we use an approach where sequences in each batch are padded with "N" nucleotides to match the length of the longest sequence in the batch. The `pad_collate` function is used for implementation. However, to enable this function to work properly, the shape of sequence tensors needs to be changed, which is achieved by setting the `t.Seq2Tensor(sequence_first=True)` parameter.

2) **Genomic Coordinates**: Use the `genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes in the dataset. *Uses 0-based indexing for genomic coordinates.*

3) **Example Usage**: See [Sure Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Arensbergen2019/ArensbergenDataset_example.ipynb) for detailed usage example and training

## Examples

### 1)  Import Important Packages and Create Padding Function

```python
    from torch.nn.utils.rnn import pad_sequence

    from mpramnist.Arensbergen2019.dataset import ArensbergenDataset
    from mpramnist.Arensbergen2019.trainer import LitModel_Arensbergen_Reg

    import mpramnist.transforms as t

    from torch.utils.data import DataLoader

    # pad_collate to handle batches of variable-length sequences. It pads shorter sequences with N so all items in the batch have the same shape
    def pad_collate(batch):
        (seq, targets) = zip(*batch)
        seq = pad_sequence(seq, 
                        batch_first=True, 
                        padding_value=0.25  # padding with "N" nucleotides
                        )
        return seq, torch.vstack(targets)
```

### 2) Initialize transforms

```python
    train_transform = t.Compose(
        [
            t.ReverseComplement(0.5),
            t.Seq2Tensor(sequence_first=True),
        ]
    )
    test_transform = t.Compose(
        [
            t.Seq2Tensor(sequence_first=True),
        ]
)
   
```

### 3) Dataset Creation

```python
    # Load training data for classification from one genome
    train_dataset = ArensbergenDataset(
        split="train",
        genome_id="SuRE42_HG02601", 
        task="regression",
        transform = train_transform
    )

    # Load regression data with genomic region filtering
    dataset = SureDataset(
        split="test",
        genome_id="SuRE43_GM18983",
        task="regression",
        genomic_regions="promoters.bed",
        transform = test_transform
    )

    # Load data excluding specific genomic regions
    regions = [{"chrom": "chr1", "start": 1000000, "end": 2000000}]
    dataset = SureDataset(
        split="val",
        genome_id="SuRE44_HG01241",
        task="classification", 
        genomic_regions=regions,
        exclude_regions=True,
        transform = test_trasnform
    )
```
### 4) Dataloader Creation

```python
    # Create DataLoader for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,  # shuffle for training
        num_workers=16,
        pin_memory=True,
        collate_fn=pad_collate,  # pad sequences to max length in each batch
    )

    # Create DataLoader for validation
    val_loader = DataLoader(
        dataset=dataset,
        batch_size=1024,
        shuffle=False,  # no shuffle for validation or test
        num_workers=16,
        pin_memory=True,
        collate_fn=pad_collate,
    )
```

## Launch Parameters

```bash
    #MPRALegNet
    python3 Arensbergen_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --genome_ids SuRE42_HG02601 SuRE43_GM18983 SuRE44_HG01241 SuRE45_HG03464 --cell_types HepG2 K562 --result_dir ./Arensbergen_legnet.tsv
    #Malinois
    python3 Arensbergen_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --genome_ids SuRE42_HG02601 SuRE43_GM18983 SuRE44_HG01241 SuRE45_HG03464 --cell_types HepG2 K562 --result_dir ./Arensbergen_malinois.tsv
    #MPRAnn
    python3 Arensbergen_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --genome_ids SuRE42_HG02601 SuRE43_GM18983 SuRE44_HG01241 SuRE45_HG03464 --cell_types HepG2 K562 --result_dir ./Arensbergen_mprann.tsv
    #PARM
    python3 Arensbergen_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --genome_ids SuRE42_HG02601 SuRE43_GM18983 SuRE44_HG01241 SuRE45_HG03464 --cell_types HepG2 K562 --result_dir ./Arensbergen_parm.tsv
    #DREAM-RNN
    python3 Arensbergen_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --genome_ids SuRE42_HG02601 SuRE43_GM18983 SuRE44_HG01241 SuRE45_HG03464 --cell_types HepG2 K562 --result_dir ./Arensbergen_dream_rnn.tsv
```

## Original Benchmark Quality

No other study has used this data for pretraining, so we don't have information about the quality metrics achieved by the original authors.

## Achieved Quality Using LegNet Model

| Cell Type | Genome ID | Original Performance | MPRALegNet | MPRAnn | Malinois | PARM | DREAM-RNN |
|-----------|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
| HepG2 | SuRE42_HG02601 | #N/A | 0,359 | 0,2952 | 0.352 | 0.3265 | 0.3345 |
| K562 | SuRE42_HG02601 | #N/A | 0,51 | 0,4352 | 0.5105 | 0.4865 | 0.4831 |
| HepG2 | SuRE43_GM18983 | #N/A | 0,345 | 0,2887 | 0.3442 | 0.3186 | 0.3356 |
| K562 | SuRE43_GM18983 | #N/A | 0,495 | 0,4263 | 0.4977 | 0.4674 | 0.4878 |
| HepG2 | SuRE44_HG01241 | #N/A | 0,307 | 0,2433 | 0.2979 | 0.2804 | 0.318 |
| K562 | SuRE44_HG01241 | #N/A | 0,578 | 0,5253 | 0.5688 | 0.55 | 0.5831 |
| HepG2 | SuRE45_HG03464 | #N/A | 0,322 | 0,2611 | 0.3161 | 0.2809 | 0.2799 |
| K562 | SuRE45_HG03464 | #N/A | 0,617 | 0,5672 | 0.6204 | 0.5998 | 0.5873 |

## Citation

When using this dataset, please cite the original publication:

[van Arensbergen J et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5498152/) 

van Arensbergen J, FitzPatrick VD, de Haas M, Pagie L, Sluimer J, Bussemaker HJ, van Steensel B. Genome-wide mapping of autonomous promoter activity in human cells. Nat Biotechnol. 2017 Feb;35(2):145-153. doi: 10.1038/nbt.3754. Epub 2016 Dec 26. PMID: 28024146; PMCID: PMC5498152.

```bibtex
    @article{arensbergen2017Genome-wide,
        title={Genome-wide mapping of autonomous promoter activity in human cells},
        author={van Arensbergen J, FitzPatrick VD, de Haas M, Pagie L, Sluimer J, Bussemaker HJ, van Steensel B},
        journal={Nat. Biotechnol.},
        volume={35(2)},
        pages={145--153},
        year={2017},
        doi={10.1038/nbt.3754}
    }
```