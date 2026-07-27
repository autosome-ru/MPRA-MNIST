# Fluorescence dataset

## Main Information

The Fluorescence dataset ([Reddy AJ et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/)) is based on a **lentiviral** MPRA screen designed to measure the enhancer activity of synthetic promoters in three immune cell lines: Jurkat (T cells), K562 (lymphoblasts), and THP-1 (monocytes).

The dataset comprises 17,104 designed promoter sequences of 250 bp length, which were strategically selected to maximize the discovery of differentially active promoters. The promoters fall into three classes: sequences from endogenous differentially expressed genes (~50%), sequences tiled with motifs enriched in such genes (~40%), and sequences from constitutively highly expressed genes (~10%).

Each promoter was cloned upstream of a minimal CMV promoter driving an EGFP reporter. Its regulatory strength in each cell line is quantified as the log₂ ratio of cells in the highest vs. lowest EGFP expression quartiles (averaged over two replicates). This value represents the promoter's ability to shift the expression distribution in a cell population.

The data is split into training (70%), validation (10%), and test (20%) sets, stratified by promoter class and GC content.

See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Reddy2023/ReddyDataset_example.ipynb) for detailed usage example and training

### Target Variable: Activity Score Calculation

**Important:** This dataset uses a different quantification method than standard RNA/DNA-based MPRAs.

### Experimental Measurement:
1. Each promoter drives EGFP expression in a lentiviral vector
2. Cells are sorted by fluorescence intensity using **Fluorescence-Activated Cell Sorting**
3. DNA is sequenced from:
   - Top 25% brightest cells (high expression quartile)
   - Bottom 25% dimmest cells (low expression quartile)

### Activity Measurement

Activity = log₂( (reads_from_top_quartile + 1) / (reads_from_bottom_quartile + 1) )

### Biological Interpretation:
This metric quantifies a promoter's ability to **create differential expression within a cell population** rather than its mean transcriptional strength. A high score indicates that the promoter makes some cells very bright while others remain dim.

### Why This Matters for Applications:
This approach is particularly relevant for designing **cell-type-specific promoters** in gene therapy, where it's crucial to have:
- Strong activity in target cells
- Minimal "leakage" in non-target cells
- Clear on/off switching between cell types

### Comparison to Standard MPRA:
- **Standard MPRA:** `log₂(RNA/DNA)` = mean transcriptional output
- **This dataset:** `log₂(high_fluorescence_cells/low_fluorescence_cells)` = expression heterogeneity

## Tasks

### Regression

The regression task involves predicting the enhancer activity score (as defined above) for each of the three cell lines. The target values are continuous, representing the log2 fold-enrichment of cells in high vs. low fluorescence bins.

### Classification

The classification task is not yet implemented and will be added in a future release.

### Data Representation

```
sequence	    JURKAT  	K562	    THP1	numerical_JURKAT	    numerical_K562	        numerical_THP1
GGGGGCGCT...	False	    False	    True	-0.2697505930801609	    -1.1044306691338297	    0.1111939002042786
CCATGCGGC...	True	    True	    True	0.4672150010860452	    0.4805318315873264	    0.5567267144276968
CGCGTTCCA...	False	    True	    False	-0.693749046357636	    1.1414958790310077	    -0.8999900063099485
```

- **sequence**: The 250 bp DNA sequence.

- **JURKAT, K562, THP1 (boolean)**: Binary labels indicating activity above a threshold in the respective cell line.

- **numerical_JURKAT, numerical_K562, numerical_THP1 (float)**: Normalized continuous activity values for regression.

## Parameters

### **`split : str`**

Defines which data split to use. Must be one of: 'train', 'val', 'test'.
Determines which dataset file to load (e.g., 'Fluorescence_train.tsv').

### **`cell_type : str` | List[str]**, optional, default=`["JURKAT", "K562", "THP1"]`

Cell type(s) to include in the dataset. Can be a single cell type string
or list of multiple cell types. All three cell types are included by default.

### **`task : str`, optional**, default="regression"

Specifies the machine learning task. Currently only "regression" is supported.
Classification may be added in future versions.

### **`transform : callable`, optional**

Transformation applied to each sequence object.

### **`target_transform : callable`, optional**

Transformation applied to the target data.

### **`root : str`, optional, default=`None`**

Root directory where dataset files are stored or should be downloaded.
If None, uses the default dataset directory from parent class.


## Data Handling Considerations

1) **Cell Type Selection**: Use the `cell_type` parameter to select one or multiple cell lines. All three cell lines `["JURKAT", "K562", "THP1"]` used by default

2) **Available Tasks**: Currently, only the regression task is supported. Classification will be added in future versions..

3) **Multi-task Learning**: This dataset is designed for multi-task learning, where a single model predicts activities across multiple cell lines simultaneously.

4) **Example Usage**: See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Reddy2023/ReddyDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    from mpramnist.Reddy2023 import ReddyDataset
    from mpramnist.Reddy2023 import LitModel_Reddy_Reg

    import mpramnist.transforms as t

    from torch.utils.data import DataLoader
```

### 2) Initialize transforms

```python
    transform = t.Compose(
        [
            t.ReverseComplement(0.5),       # Reverse-complement augmentation (training only)
            t.Seq2Tensor(),                 # Convert the sequence string to a numerical tensor
        ]
    )
    val_trainsform = t.Compose(
        [
            t.ReverseComplement(0.0),       # for validation and test
            t.Seq2Tensor(),                 # Convert the sequence string to a numerical tensor
        ]
    )
```

### 3) Dataset Creation

```python
    # Basic usage with default settings (all cell types)
    train_dataset = ReddyDataset(split='train', transform = transform)
    
    # Specific cell type for regression
    dataset = ReddyDataset(
        split='val',
        cell_type='K562',    # Use only K562 cell line
        task='regression',
        transform = val_transform,
        root = "../data/"
    )

```

### 4) Dataloader Creation

```python 
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=1024, 
        shuffle=True, # Shuffle is recommended for training
        num_workers=16 
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1024,
        shuffle=False, # No need to shuffle for validation/testing
        num_workers=16 
    )
```

## Launch Parameters

```bash
#MPRALegNet
python3 ReddyReg_model_launch.py --model MPRALegNet --lr 1e-2 --wd 1e-1 --result_dir ./Reddy_legnet.tsv --batch_size 1024 --epoch_num 50 --runs 5
#Malinois
python3 ReddyReg_model_launch.py --model Malinois --lr 1e-2 --wd 1e-1 --result_dir ./Reddy_malinois.tsv --batch_size 1024 --epoch_num 50 --runs 5
#MPRAnn
python3 ReddyReg_model_launch.py --model MPRAnn --lr 1e-2 --wd 1e-1 --result_dir ./Reddy_mprann.tsv --batch_size 1024 --epoch_num 50 --runs 5
#PARM
python3 ReddyReg_model_launch.py --model PARM --lr 1e-2 --wd 1e-1 --result_dir ./Reddy_parm.tsv --batch_size 1024 --epoch_num 50 --runs 5
#DREAM-RNN
python3 ReddyReg_model_launch.py --model DREAM-RNN --lr 1e-2 --wd 1e-1 --result_dir ./Reddy_dream-rnn.tsv --batch_size 1024 --epoch_num 50 --runs 5
```

## Original Benchmark Quality

Pearson correlation, r

| Cel type | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM-RNN |
|-----------|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|
| HepG2 | 0,615 | 0,592 | 0,5202 | 0,5137 | 0,6411 | 0.5999 |
| K562 | 0,599 | 0,604 | 0,5322 | 0,4931 | 0,63 | 0.5867 |
| WTC11 | 0,555 | 0,507 | 0,4623 | 0,4516 | 0,5701 | 0.5347 |

## Citation

When using this dataset, please cite the original publication:

[Reddy AJ et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/)

Reddy AJ, Herschl MH, Geng X, Kolli S, Lu AX, Kumar A, Hsu PD, Levine S, Ioannidis NM. Strategies for effectively modelling promoter-driven gene expression using transfer learning. bioRxiv [Preprint]. 2024 May 19:2023.02.24.529941. doi: 10.1101/2023.02.24.529941. PMID: 36909524; PMCID: PMC10002662.

```bibtex
    
    @article{reddy2024strategies,
        title = {Strategies for effectively modelling promoter-driven gene expression using transfer learning},
        author = {Reddy, Aniketh Janardhan and Herschl, Michael H. and Geng, Xinyang and Kolli, Sathvik and Lu, Amy X. and Kumar, Aviral and Hsu, Patrick D. and Levine, Sergey and Ioannidis, Nilah M.},
        journal = {bioRxiv},
        year = {2023},
        month = feb,
        doi = {10.1101/2023.02.24.529941},
        url = {https://doi.org/10.1101/2023.02.24.529941}
    }   
```