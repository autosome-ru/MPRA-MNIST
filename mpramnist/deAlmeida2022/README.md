# DeepSTARR dataset

## Main Information

The DeepSTARR dataset ([de Almeida et al. 2022](https://www.nature.com/articles/s41588-022-01048-5)) is based on a **genome-wide STARR-seq screen** in D. melanogaster S2 cells to quantitatively measure the activity of **thousands of native genomic enhancers** towards two distinct promoter types: developmental (*sim*) and housekeeping (*hsp70*).

**The dataset comprises 249 bp sequences**, which correspond to enhancer regions identified as peaks in the original screen, resized to a fixed length. The target variable (`Dev_log2` and `Hk_log2`) represents the quantitative enhancer activity, calculated as **log2((RNA+1)/(DNA+1))**, where RNA and DNA are the read counts from the STARR-seq reporter RNA and input DNA libraries, respectively. This log2 fold-change (log2FC) metric quantifies the transcriptional activation strength of each enhancer sequence for the corresponding promoter type.

The data is split into training (native sequences, ~80%), validation (sequences from the first half of chromosome 2R, ~8.4%), and test (sequences from the second half of chromosome 2R, ~8.5%) sets. **In the original study, the training set was expanded by adding reverse-complemented copies of all sequences, effectively doubling its size**. The `use_original_reverse_complement` parameter controls this behavior in the dataset loader.

Therefore, we recommend using the corresponding transform: `ReverseComplement(0.5)` if you disable the built-in augmentation (see parameter `use_original_reverse_complement`).

See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/deAlmeida2022/deAlmeida_example.ipynb) for detailed usage example and training

## Tasks

### Regression

The regression task is to predict the **quantitative enhancer activity** (log2 enrichment or log2((RNA+1)/(DNA+1))) of a DNA sequence for one or both promoter types, as measured in the STARR-seq assay.

### Data Representation

```
name	                            chromosome	start	    end	    strand	sequence	    Dev_log2	Hk_log2	    split
chr2L_5587_5835_+_positive_peaks	    2L	    5586	    5835	   +	ATTCAGATTGC...	5.711541	1.3625218	train
chr2R_584101_584349_-_negative	        2R	    584100	    584349	   -	TTAAGGCAATG...	1.2294666	1.1338974	val
chr2R_21143801_21144049_-_negative	    2R	    21143800	21144049   -	TTTCTTGAACT...	0.61593497	-1.2059526	test
```

## Parameters

### **`split : str | List[str]`**

Data split specification. Can be:
- Standard splits: `'train'`, `'val'`, `'test'`
- Chromosome names: any from `LIST_OF_CHR`
- List of chromosome names for custom splits

### **`cell_type : str | List[str]`, default: `["Dev_log2", "Hk_log2"]`**

Promoter type(s) for target data.   Can be:
- `"Dev_log2"`: Developmental activity
- `"Hk_log2"`: Housekeeping activity
- List containing both for multi-task learning
- Named `cell_type` for other dataset's attribute consistency

### **`use_original_reverse_complement : bool | None`, optional**

Whether to apply reverse complement augmentation as in original study.
- If `None` (default): Automatically set to `True` for the `'train'` split and `False` for '`val'`/`'test'`.

- If `True`: The dataset internally returns an augmented set for training (original + reverse complement). **Do not apply an external `ReverseComplement` transform in this case, as it will cause data leakage.**

If `False`: The dataset returns the original sequences. You can then apply your own `ReverseComplement` transform via the `transform` parameter.

### **`genomic_regions : str | List[Dict]`, optional**

Genomic regions to include or exclude. Can be:
- Path to BED file
- List of dictionaries with `'chrom'`, `'start'`, `'end'` keys

### **`exclude_regions : bool`, default: `False`**

If `True`, exclude the specified genomic regions instead of including them

### **`transform : callable`, optional**

Transformation function applied to each sequence.

### **`target_transform : callable`, optional**

Transformation function applied to target values.

### **`root : str`, optional**

Root directory for data storage.

## Data Handling Considerations

1) **Promoter Type Selection**: Use the `cell_type` parameter to select one or multiple promoter lines. `Dev_log2` is Developmental activity, `Hk_log2` is Housekeeping activity.

2) **Multi-task Learning**: This dataset is designed for multi-task learning, where a single model predicts activities across multiple promoter types simultaneously.

3) **Genomic Region Filtering**: Use `the genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes. The dataset uses **0-based** indexing for genomic coordinates in **D. melanogaster genome**.

4) **Reverse Complement Augmentation**: The `use_original_reverse_complement` parameter controls the built-in augmentation logic from the original paper. If set to `True`/`None` for the training split, the dataset handles augmentation internally. In this case, do not apply a `ReverseComplement` transform manually, as it will duplicate data incorrectly. Use the `transform` parameter only for other sequence manipulations (e.g., `Seq2Tensor`).

5) **Example Usage**: See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/deAlmeida2022/deAlmeida_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
from mpramnist.deAlmeida2022.dataset import deAlmeidaDataset
from mpramnist.deAlmeida2022.trainer import LitModel_deAlmeida

from torch.utils.data import DataLoader
import mpramnist.transforms as t
```

### 2) Initialize transforms

```python
# For use with `use_original_reverse_complement=True/None` (default)
train_transform = t.Compose([t.Seq2Tensor()]) # t.ReverseComplement is NOT needed here
val_test_transform = t.Compose([t.Seq2Tensor()])

# For use with `use_original_reverse_complement=False` (custom augmentation)
train_transform_custom = t.Compose([t.ReverseComplement(0.5), t.Seq2Tensor()])
``` 

### 3) Dataset Creation

```python
# --- Using the ORIGINAL paper's augmentation strategy (RECOMMENDED) ---
# Training data: augmentation is applied internally, dataset size is 2x.
train_data = deAlmeidaDataset(split="train", cell_type="Dev_log2", transform=train_transform)
# Validation/Test data: no internal augmentation(because validation data).
val_data = deAlmeidaDataset(split="2L", cell_type="Dev_log2", transform=val_test_transform)

# Multi-task data (both activities)
multi_data = deAlmeidaDataset(split="test", cell_type=["Dev_log2", "Hk_log2"], transform=val_test_transform)

# --- Using CUSTOM augmentation (disable internal logic) ---
train_data_custom = deAlmeidaDataset(
    split="train",
    use_original_reverse_complement=False, # Disable internal augmentation
    cell_type="Dev_log2",
    transform=train_transform_custom # Apply your own transform
)

# Load data filtered by genomic regions (exclude regions in BED file)
region_data = deAlmeidaDataset(
    split="train",
    genomic_regions="path/to/regions.bed",
    exclude_regions=True,
    transform=train_transform
)

# Load a custom chromosome split
custom_split_data = deAlmeidaDataset(
    split=["2L", "2R", "3L"],
    cell_type="Dev_log2",
    transform=train_transform
)
```

### 4) Dataloader Creation

```python 
    train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=1024, 
    shuffle=True,           # Shuffle is recommended for training
    num_workers=8
)

val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=1024, 
    shuffle=False,          # No need to shuffle for validation/testing
    num_workers=8
)
```

## Launch Parameters

```bash
#MPRALegNet
python3 deAlmeida_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 1 --runs 5 --promoter_types Dev_log2 Hk_log2 --result_dir ./deAlmeida_legnet_test.tsv
#Malinois
python3 deAlmeida_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --promoter_types Dev_log2 Hk_log2 --result_dir ./deAlmeida_malinois.tsv
#MPRAnn
python3 deAlmeida_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --promoter_types Dev_log2 Hk_log2 --result_dir ./deAlmeida_mprann.tsv
#PARM
python3 deAlmeida_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --promoter_types Dev_log2 Hk_log2 --result_dir ./deAlmeida_parm.tsv
#DeepStarr
python3 deAlmeida_model_launch.py --model DeepStarr --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --promoter_types Dev_log2 Hk_log2 --result_dir ./deAlmeida_deepstarr.tsv
#DREAM-RNN
python3 deAlmeida_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 1 --runs 1 --promoter_types Dev_log2 Hk_log2 --result_dir ./deAlmeida_dream_rnn.tsv
```

## Achieved Performance Using Basic Models

| Cel type | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM-RNN |
|-----------|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|
| Developmental | 0,68 | 0,718 | 0,7121 | 0.6871 | 0.7117 | 0.7295 |
| Housekeeping | 0,74 | 0,786 | 0,7815 | 0.7751 | 0.7845 | 0.7986 |

## Citation

When using this dataset, please cite the original publication:

[de Almeida et al. 2022](https://www.nature.com/articles/s41588-022-01048-5)

de Almeida, B.P., Reiter, F., Pagani, M. et al. DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers. Nat Genet 54, 613–624 (2022). https://doi.org/10.1038/s41588-022-01048-5

```bibtex
    @article{deAlmeida2022deepstarr,
    title = {DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers},
    volume = {54},
    ISSN = {1546-1718},
    url = {http://dx.doi.org/10.1038/s41588-022-01048-5},
    DOI = {10.1038/s41588-022-01048-5},
    number = {5},
    journal = {Nature Genetics},
    publisher = {Springer Science and Business Media LLC},
    author = {de Almeida, Bernardo P. and Reiter, Franziska and Pagani, Michaela and Stark, Alexander},
    year = {2022},
    month = may,
    pages = {613–624}
}
```