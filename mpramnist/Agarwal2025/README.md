# Agarwal Single and Multi dataset

## Main Information

The Agarwal dataset ([Agarwal et al., 2025](https://www.nature.com/articles/s41586-024-08430-9)) is based on an **optimized lentiMPRA system** (lentiviral MPRA), which provides an "in genome" readout through random genomic integration, offering higher cell-type specificity compared to episomal MPRA systems.

The dataset was designed to:
1. Characterize **tissue-specific regulatory activity** of cis-regulatory elements (CREs)
2. Examine the **relative orientation dependence** of promoters and enhancers
3. Train models to predict regulatory and nucleotide variant effects

### Experimental Design

The study tested **over 200,000 sequences** in a single experiment, including:

*   **Potential enhancers:** Identified from open chromatin regions (cCREs) in corresponding cell types
*   **Core promoter regions:** To characterize promoter activity effects
*   **Canonical promoters:** Centered on transcription start sites (TSS)
*   **Shuffled enhancer sequences:** With preserved dinucleotide composition (negative controls)
*   **Control elements:** With known activity in HepG2, K562, and WTC11 cell lines
*   **60,000 sequences tested in all three cell lines** (**AgarwalMulti** library)

### Dataset Composition

The processed **AgarwalSingle** dataset comprises:
*   **HepG2:** 122,926 sequences
*   **K562:** 196,664 sequences  
*   **WTC11:** 46,185 sequences

The **AgarwalMulti** library was constructed by **selecting and re-testing elements from the individual large-scale libraries** in all three cell lines (HepG2, K562, WTC11). This enables direct comparison of the same sequences' activity across different cellular contexts.
After filtering elements (as described in the original methodology), the final processed dataset contains:

**55,338 sequences** × **3 cell lines** = **166,014 activity measurements**

All sequences are **200 nucleotides long** (excluding constant 15-nt flanks). Data is split into training, validation, and test sets using an 8:1:1 ratio, following the original study.

See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Agarwal2025/AgarwalDataset_example.ipynb) for detailed usage example and training

## Tasks

### Regression

### Calculation of Regulatory Activity

The regression task involves predicting scalar values representing the **activity score** (enhancer/promoter strength) for each cell line.

**Key steps from the original study:**
1. **Replicate measurements:** Three independent biological replicates for both DNA and RNA
2. **Barcode filtering:** Elements measured with <10 independent barcodes were excluded to reduce noise
3. **Activity calculation:** For each replicate: *log₂(RNA reads / DNA reads)*

4. **Specificity score for AgarwalMulti:** Measured the deviation of each element from its mean activity across cell types

### Data Representation

#### AgarwalSingle
```
seq_id	    chromosome  start	    end	        strand	    seq	                expression	    averaged_expression	        fold
---------------------------------------------------------------------------------------------------------
seq10004_F	    10	    89029900	89030100	+	        TAGCTCAACACAAATCC	 0.43	            -0.017	                10
seq10004_R	    10	    89029900	89030100	-	        CATTGTTTCCATAGGGA	-0.464	            -0.017	                10
seq10005_F	    10	    89032143	89032343	+	        GACCCTAAATCAGTATG	-1.231	            -1.6350000000000002	    7
seq10005_R	    10	    89032143	89032343	-	        AAAGGGACTTTCCGCAT	-2.039	            -1.6350000000000002	    7
```

**Column descriptions:**
*   `expression`: Activity score *log₂(RNA reads / DNA reads)*
*   `averaged_expression`: Mean of activity scores of forward and reverse-complement sequences
*   `fold`: Cross-validation fold (1-10)

#### AgarwalMulti

```
seq_id	chromosome_hg19	start_hg19	end_hg19	strand_hg19	chromosome	start	end	strand	seq	HepG2_Specificity_Score	K562_Specificity_Score	WTC11_Specificity_Score	fold	HepG2_log2	K562_log2	WTC11_log2
---------------------------------------------------------------------------------------------------------
ENSG00000000971	1	196620911.0	196621111.0	+	1	196651781	196651981	+	GATATCACCAGCTGCTGATTTGCACAT...	0.0705767810952613	-0.386981332585921	0.26108557591418	2	-0.5749999999999998	-0.788	-1.024
ENSG00000001630	7	91763598.0	91763798.0	-	7	92134284	92134484	-	TGGGTTTAGTAGGAGACCTGGGGCAAG...	-0.250686030152044	-0.167357423155188	0.382690550999934	6	-1.234	-1.126	-1.426
ENSG00000002726	7	150521782.0	150521982.0	+	7	150824694	150824894	+	CAAGGTGGCTGGGGAGAAGGCCGAGGT...	0.547128759230438	-0.49542235153705	-0.103962309065462	10	-0.12	-0.5879999999999999	-0.939
ENSG00000003056	12	9102102.0	9102302.0	-	12	8949506	8949706	-	GGGGTCTGGTGGGAGGAGCGGTTGCCC...	-0.638726825596637	-0.133358216202116	0.726073181797631	1	0.973	1.295	2.165

```

**Column descriptions:**
*   `HepG2_Specificity_Score`, `K562_Specificity_Score`, `WTC11_Specificity_Score` : Mean normalized activity across 3 replicates for the individual sequence for current cell type
*   `HepG2_log2`, `K562_log2`, `WTC11_log2` : Raw activity values *log₂(RNA reads / DNA reads)*
*   `fold`: Cross-validation fold (1-10)

## AgarwalSingle Parameters

```python
Cell_Type = "HepG2" # or K562 or WTC11

train_dataset = AgarwalSingleDataset(cell_type=Cell_Type, split="train", transform=train_transform, root="../data/",)
```

### **`split : Union[str, List[int], int]`**

Defines which data split to use. Options:
- String: `'train'`, `'val'`, `'test'` (uses predefined fold sets)
- List[int]: List of specific fold numbers (`1-10`)
- int: Single fold number (`1-10`)

### **`cell_type : str`**

Cell type for filtering the data. Must be one of: `'HepG2'`, `'K562'`, `'WTC11'`

### **`genomic_regions : Optional[Union[str, List[Dict]]], optional`**

Genomic regions to include or exclude. Options:
- str: Path to BED file containing genomic regions
- List[Dict]: List of dictionaries with `'chrom'`, `'start'`, `'end'` keys
- None: No genomic region filtering
- Uses **0-based** indexing for genomic coordinates in **hg38**

### **`exclude_regions : bool, default=False`**

If `True`, exclude the specified genomic regions instead of including them

### **`averaged_target : bool, default=False`**

If `True`, use `'averaged_expression'` (mean activity between forward and reverse-complement sequences) as target;
otherwise use individual `'expression'` values

### **`root : optional`**

Root directory for data storage

### **`transform : callable, optional`**

Transformation function applied to each sequence

### **`target_transform : callable, optional`**

Transformation function applied to target values

## AgarwalMulti parameters

```python
Cell_Type = "HepG2" # or K562 or WTC11

train_dataset = AgarwalMultiDataset(cell_type=Cell_Type, split="train", transform=train_transform, root="../data/",)
```

All the parameters listed below are applicable to **AgarwalMulti** dataset, including:

### **`use_specificity_score: bool = True`**
For the AgarwalMulti dataset, two representations are available:
- specificity score (default) – measures each element’s deviation from its 
    mean activity across cell types (as in the original publication).
- raw activity – the direct log₂(RNA / DNA) read count ratio.
Switch to False to use the raw activity instead of the specificity score.

## Data Handling Considerations

1) **Cell Type Selection**: Use the `cell_type` parameter to select specific cell lines. The data is not multi-label, as sequences measured in HepG2 were not measured in K562.

2) **Genomic Region Filtering**: Use `the genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes. Uses **0-based** indexing for genomic coordinates in **hg38**.

3) **Constant Flanks**: Original sequences in the study had constant 15-nucleotide flanks on each side. These flanks have been removed from the provided sequences, but for optimal or comparable results, we recommend adding them back before training, validation, and testing. Use `AgarwalDataset.CONSTANT_LEFT_FLANK` and `AgarwalDataset.CONSTANT_RIGHT_FLANK` as shown in the examples below.

4) **LegNet Shift Augmentation**: The LegNet model uses embedded flanks for shift augmentation. These flanks are stored in `AgarwalDataset.LEFT_FLANK` and `AgarwalDataset.RIGHT_FLANK` attributes. Use them as shown in the examples below. Shift augmentation involves shifting sequences by a certain number of nucleotides left or right.

5) **Target Selection**: You can use either individual sequence activity measurements (`averaged_target = False`, using the *expression* column) or averaged activities between forward and reverse-complement sequences (`averaged_target = True`, using the *averaged_expression* column).

6) **Specificity score for AgarwalMulti:** The authors defined specificity as the deviation of each element's activity from its mean across all cell types, highlighting cell‑type‑specific signals. Setting `use_specificity_score` parameter to False instead returns the raw activity values, computed as the log2‑ratio of RNA reads to DNA reads for each element. For all MPRA-MNIST computations we used `use_specificity_score`=True.

7) **Example Usage**: See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Agarwal2025/AgarwalDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
# Single
from mpramnist.Agarwal2025.dataset import AgarwalSingleDataset
from mpramnist.Agarwal2025.trainer import LitModel_AgarwalSingle

# Multi
from mpramnist.Agarwal2025.dataset import AgarwalMultiDataset
from mpramnist.Agarwal2025.trainer import LitModel_AgarwalMulti

import mpramnist.transforms as t

from torch.utils.data import DataLoader
```

### 2) Initialize trannsforms

```python

# required for each sequence. flanks from original study
constant_left_flank = AgarwalDataset.CONSTANT_LEFT_FLANK 
constant_right_flank = AgarwalDatase.CONSTANT_RIGHT_FLANK  

# original flanks from human MPRAlegnet. Using for shifting augmentation
left_flank = AgarwalDataset.LEFT_FLANK  
right_flank = AgarwalDataset.RIGHT_FLANK

# Training transform with augmentations
transform = t.Compose(
    [
        t.AddFlanks(constant_left_flank, constant_rigtht_flank), # Add constant flanks

        # Transforms for shift augmentation (use only for training)
        t.AddFlanks("", right_flank),   # these transforms are used to the shift augmentation.
        t.RightCrop(230, 260),          # Shift parameters are (length, len(right_flank))
        t.LeftCrop(230, 230),           # 

        t.ReverseComplement(0.5),       # Reverse-complement augmentation (training only)
        t.Seq2Tensor(),
    ]
```

### 3) Dataset Creation

```python
# Load training data for HepG2 cell type
dataset = AgarwalSingleDataset(split='train', cell_type='HepG2')
dataset = AgarwalMultiDataset(split='train', cell_type='HepG2')

# Load data filtered by genomic regions from BED file
dataset = AgarwalSingleDataset(
    split='train',
    cell_type='K562',
    transform = transform,
    genomic_regions='path/to/regions.bed'
)

# Load data excluding specific genomic regions
regions = [{'chrom': '1', 'start': 1000, 'end': 2000}]
dataset = AgarwalSingleDataset(
    split=[1, 2, 3],
    cell_type='WTC11',
    genomic_regions=regions,
    transform = transform,
    exclude_regions=True
)

val_dataset = AgarwalSingleDataset(
    cell_type="HepG2",
    split=[9], # or 'val'
    transform = validation_transform, # validation transforms should not use shift and reverse-complement
    root="../data/",
)
```

### 4) Dataloader Creation

```python 
val_loader = DataLoader(
    dataset=val_dataset, batch_size=1024, shuffle=False, num_workers=16
)
```

## Launch Parameters

### AgarwalSingle
```bash
#MPRALegNet
python3 AgarwalSingle_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalsingle_legnet.tsv
#Malinois
python3 AgarwalSingle_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalsingle_malinois.tsv
#MPRAnn
python3 AgarwalSingle_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalsingle_mprann.tsv
#PARM
python3 AgarwalSingle_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalsingle_parm.tsv
#DREAM-RNN
python3 AgarwalSingle_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalsingle_dream_rnn.tsv
```

### AgarwalMulti
```bash
#MPRALegNet
python3 AgarwalMulti_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalmulti_legnet.tsv
#Malinois
python3 AgarwalMulti_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalmulti_malinois.tsv
#MPRAnn
python3 AgarwalMulti_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalmulti_mprann.tsv
#PARM
python3 AgarwalMulti_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalmulti_parm.tsv
#DREAM-RNN
python3 AgarwalMulti_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 K562 WTC11 --result_dir ./Agarwalmulti_dream_rnn.tsv
```

## Achieved Performance Using Basic Models

### AgarwalSingle

Pearson correlation, r

| Cell type | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM-RNN |
|-----------|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|
| HepG2 | **0,8189** | 0,806 | 0,7736 | 0.7274 | 0.7985 | 0.7767 |
| K562 | **0,8514** | 0,83 | 0,7901 | 0.7816 | 0.823 | 0.8012 |
| WTC11 | **0,7354** | 0,718 | 0,6796 | 0.6254 | 0.7237 | 0.6724 |

### AgarwalMulti

Pearson correlation, r

| Cell type | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM_RNN |
|-----------|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:| :--------------------:|
| HepG2 | 0,78 | **0,798** | 0,7633 | 0.708170 | 0.7886 | 0.742 |
| K562 | 0,75 | **0,759** | 0,7248 | 0.666216 | 0.755 | 0.6883 |
| WTC11 | 0,77 | **0,77** | 0,738 | 0.690842 | 0.7658 | 0.7034 |

## Citation

When using this dataset, please cite the original publication:

[Agarwal et al. 2025](https://www.nature.com/articles/s41586-024-08430-9) 

Agarwal, V., Inoue, F., Schubach, M. et al. Massively parallel characterization of transcriptional regulatory elements. Nature 639, 411–420 (2025). https://doi.org/10.1038/s41586-024-08430-9

```bibtex
    @article{agarwal2025massively,
        title={Massively parallel characterization of transcriptional regulatory elements},
        author={Agarwal, V. and Inoue, F. and Schubach, M. and others},
        journal={Nature},
        volume={639},
        pages={411--420},
        year={2025},
        doi={10.1038/s41586-024-08430-9}
    }
```