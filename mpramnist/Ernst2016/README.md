# Sharpr dataset

## Main information

The Sharpr (Systematic High-resolution Activation and Repression Profiling using Reporters) dataset is based on the original Sharpr-MPRA dataset ([Ernst et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5125825/)), which contains expression activity measurements for approximately 487,000 synthetic promoter sequences of 145 bp length. These sequences are derived from DNase I peaks in K-562, HepG2, HUVEC, and H1-hESC cells. Each promoter is cloned upstream of either a minimal TATA promoter or a strong SV40 promoter, and promoter-driven expression is measured in both K562 and HepG2 cell lines with two biological replicates, resulting in 8 raw measurements per sequence (2 cell lines × 2 promoters × 2 replicates).

Following the modeling approach by [Movva et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31206543/), the dataset includes **replicate averaging**: the average of two replicates is computed, resulting in 12 total outputs per input sequence (8 individual replicates + 4 averaged values)

The dataset is partitioned as in the original study ([Reddy et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/)):

- **Training**: All sequences except those from chromosomes 8 and 18

- **Validation**: ~19,000 sequences from chromosome 8

- **Testing**: ~10,000 sequences from chromosome 18

Our implementation and preprocessed data are adapted from:
    https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/Sharpr_MPRA.py

### Data Processing Pipeline

The raw experimental data undergoes the following preprocessing steps:

1) **Read count generation**: DNA and RNA counts are generated following [Kheradpour et al. 2013](https://pubmed.ncbi.nlm.nih.gov/23512712/), with a pseudo-count of 1 added to all counts

2) **Library size normalization**: RNA counts are divided by the total RNA count for the experiment; DNA counts are divided by the total DNA count

3) **Log fold change calculation**: The log₂ ratio of normalized RNA to DNA counts is computed: log₂((RNA/total_RNA) / (DNA/total_DNA))

4) **Quality filtering**: Barcodes with original DNA counts < 20 are treated as missing data

5) **Background subtraction (pilot design)**: For initial experiments, values are normalized by subtracting the average activity of control tiles (positions #1 and #9) to estimate background

6) **Z-score normalization**: Column-wise z-score normalization is applied to the log fold changes for each of the 12 tasks, resulting in outputs with mean 0 and variance 1

## Tasks

### Regression Task

The regression task involves predicting 12 normalized expression values for each 145bp promoter sequence. These values represent the processed log fold changes of expression activity across different experimental conditions:

- **2 cell lines**: K562 and HepG2

- **2 core promoters**: Minimal TATA promoter (minp) and SV40 promoter (sv40p)

- **3 representations**: Two biological replicates (rep1, rep2) and their average (avg)

The 12 target columns are:

    k562_minp_rep1, k562_minp_rep2, k562_minp_avg,
    k562_sv40p_rep1, k562_sv40p_rep2, k562_sv40p_avg,
    hepg2_minp_rep1, hepg2_minp_rep2, hepg2_minp_avg,
    hepg2_sv40p_rep1, hepg2_sv40p_rep2, hepg2_sv40p_avg

### Data Representation

```    
    name	                    chromosome	    start	    end	        strand	seq	            k562_minp_rep1	k562_minp_rep2	k562_minp_avg	...       split
    -----------------------------------------------------------------------------------------------------
    H1hesc_1_0_0_chr20_30310735	    20	        30310588	30310733	+	    GGGAGCCCA...	-1.8394496	    -1.3714164	        -1.860508       ... 	  train
    H1hesc_1_6_0_chr8_128830575	    8	        128830428	128830573	+	    CCCATTTTA...	-0.8249799	    -1.5034798	        -1.3725991      ... 	  val
    H1hesc_1_94_0_chr18_11851975	18	        11851828	11851973	+	    AATCTGGAG...	-1.1480043	    -1.559959	        -1.5876448      ... 	  test

```

See [Sharpr Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Ernst2016/ErnstDataset_example.ipynb) for detailed information.

## Parameters

### **`split : str`**

Defines which split to use (e.g., `'train'`, `'val'`, `'test'`, or list of fold indices).

### **`cell_type : List[str]`**

List of column names with activity data to be used as targets.
Must be a subset of 
    
    CELL_TYPES = [
        "k562_minp_rep1", "k562_minp_rep2", "k562_minp_avg",
        "k562_sv40p_rep1", "k562_sv40p_rep2", "k562_sv40p_avg",
        "hepg2_minp_rep1", "hepg2_minp_rep2", "hepg2_minp_avg",
        "hepg2_sv40p_rep1", "hepg2_sv40p_rep2", "hepg2_sv40p_avg",
    ]

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

Root directory where data is stored. If `None`, uses default data path.

## Data Handling Considerations

1) **Cell Types**: The `cell_type` parameter determines which columns to use for prediction. This allows the task to be formulated as either multi-label regression or single-label regression.

2) **Genomic Coordinates**: Use the `genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes in the dataset. *Uses 0-based indexing for genomic coordinates.*

3) **Example Usage**: See [Sharpr Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Ernst2016/ErnstDataset_example.ipynb) for detailed usage example and training

## Examples

### 1)  Import Important Packages

```python
    import torch
    import mpramnist
    from mpramnist.Ernst2016 import ErnstDataset
    from mpramnist.Ernst2016 import LitModel_Ernst

    import mpramnist.transforms as t

    from torch.utils.data import DataLoader

    print(SharprDataset.CELL_TYPES)

>>> ['k562_minp_rep1',
 ... 'k562_minp_rep2',
 ... 'k562_minp_avg',
 ... 'k562_sv40p_rep1',
 ... 'k562_sv40p_rep2',
 ... 'k562_sv40p_avg',
 ... 'hepg2_minp_rep1',
 ... 'hepg2_minp_rep2',
 ... 'hepg2_minp_avg',
 ... 'hepg2_sv40p_rep1',
 ... 'hepg2_sv40p_rep2',
 ... 'hepg2_sv40p_avg']
```

### 2) Initialize transforms

```python
    train_transform = t.Compose(
        [
            t.ReverseComplement(0.5), # Set it to 0.0 for validation and test
            t.Seq2Tensor(),
        ]
    )
``` 

### 3) Dataset Creation

```python
    train_dataset = ErnstDataset(
        split="train",
        cell_type=ErnstDataset.CELL_TYPES,
        transform = transform
    )   

    # Load regression data with genomic region filtering
    test_dataset = ErnstDataset(
        split="test",
        cell_type = ErnstDataset.CELL_TYPES,
        genomic_regions="promoters.bed",
        transform = transform
    )

    # Load data excluding specific genomic regions
    regions = [{"chrom": "chr1", "start": 1000000, "end": 2000000}]
    dataset = ErnstDataset(
        split="val",
        cell_type = ErnstDataset.CELL_TYPES,
        genomic_regions=regions,
        exclude_regions=True,
        transform = transform
    )
```

### 2) Dataloader Creation

```python
    # Create DataLoader for training
    loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,  # True for training, False for val and test
        num_workers=16,
        pin_memory=True,
    )
```

## Launch Parameters

```bash
#MPRALegNet
python3 Ernst_model_launch.py --model MPRALegNet --lr 1e-2 --wd 1e-1 --result_dir ./ernst_legnet.tsv --batch_size 1024 --epoch_num 50 --runs 5
#Malinois
python3 Ernst_model_launch.py --model Malinois --lr 1e-2 --wd 1e-1 --result_dir ./ernst_malinois.tsv --batch_size 1024 --epoch_num 50 --runs 5
#MPRAnn
python3 Ernst_model_launch.py --model MPRAnn --lr 1e-2 --wd 1e-1 --result_dir ./ernst_mprann.tsv --batch_size 1024 --epoch_num 50 --runs 5
#PARM
python3 Ernst_model_launch.py --model PARM --lr 1e-2 --wd 1e-1 --result_dir ./ernst_parm.tsv --batch_size 1024 --epoch_num 50 --runs 5
#DREAM-RNN
python3 Ernst_model_launch.py --model DREAM-RNN --lr 1e-2 --wd 1e-1 --result_dir ./ernst_dream-rnn_еуые.tsv --batch_size 1024 --epoch_num 1 --runs 5
```

## Original Benchmark Quality

No other study has used this data for pretraining, so we don't have information about the quality metrics achieved by the original authors.

## Achieved Performance Using Basic Models

Pearson correlation, r

| Cell type | Experiment | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM_RNN |
|-----------|:---------------:|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:| :--------------------:|
| HepG2 | minP_avg| #N/A | 0,3336 | 0,2689 | 0.3587 | 0.3897 | 0.3704 |
| HepG2 | sv40_avg| #N/A | 0,3158 | 0,3053 | 0.1907 | 0.2183 | 0.2135 |
| K562 | minP_avg| #N/A | 0,3874 | 0,3124 | 0.2811 | 0.3208 | 0.3197 |
| K562 | sv40_avg| #N/A | 0,2062 | 0,1728 | 0.2953 | 0.3445 | 0.3462 |


## Citation

When using this dataset, please cite the original publication:

[Ernst J et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5125825/) 

Ernst J, Melnikov A, Zhang X, Wang L, Rogov P, Mikkelsen TS, Kellis M. Genome-scale high-resolution mapping of activating and repressive nucleotides in regulatory regions. Nat Biotechnol. 2016 Nov;34(11):1180-1190. doi: 10.1038/nbt.3678. Epub 2016 Oct 3. PMID: 27701403; PMCID: PMC5125825.

```bibtex
    @article{ernst2016Genome-scale,
        title={Genome-scale high-resolution mapping of activating and repressive nucleotides in regulatory regions},
        author={Ernst J, Melnikov A, Zhang X, Wang L, Rogov P, Mikkelsen TS, Kellis M.},
        journal={Nat. Biotechnol.},
        volume={34(11)},
        pages={1180-1190},
        year={2016},
        doi={10.1038/nbt.3678}
    }
```