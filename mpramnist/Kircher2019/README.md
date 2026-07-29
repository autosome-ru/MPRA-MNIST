# Kircher dataset

## Main Information

The Kircher dataset is based on results from **episomal** saturating mutagenesis MPRA experiments ([Kircher et al. 2019](https://www.nature.com/articles/s41467-019-11526-w)). The study characterized 44,647 regulatory element variants, including 11 enhancers and 10 promoters across 12 cell lines.

These experimentally characterized sequences are proposed as a benchmark dataset for validating machine learning model quality. Specifically, models can be trained on independent data (e.g., Agarwal dataset) and their predictive power can be evaluated on the Kircher saturation mutagenesis data (see [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Kircher2019/KircherDataset_example.ipynb)).

### Promoters

| Name | Element Type | Cell type |
| ----------- | ----------- | ----------- |
| F9 | Hemophilia B | HepG2 |
| LDLR | Familial hypercholesterolemia | HepG2 |
| FOXE1 | Thyroid cancer | HeLa |
| GP1BB | Bernard-Soulier Syndrome |  HEL 92.1.7 |
| HBB | Thalassemia | HEL 92.1.7 |
| HBG1 | Hereditary persistence of fetal hemoglobin | HEL 92.1.7 |
| HNF4A (P2) | Maturity-onset diabetes of the young (MODY) | HEK293T |
| MSMB | Prostate cancer | HEK293T |
| TERT | Various types of cancer | HEK293T, SF7996 |
| PKLR | Pyruvate kinase deficiency | K562 |


### Enhancers 

| Name | Element Type | Cell type |
| ----------- | :----------- | ----------- |
| SORT1 | Plasma low-density lipoprotein cholesterol & myocardial infarction | HepG2 |
| BCL11A+58 | Sickle cell disease | HEL 92.1.7 |
| MYC (rs6983267) | Various types of cancer | HEK293T | 
| IRF6 | Cleft lip | HaCaT |
| IRF4 | Human pigmentation | SK-MEL-28 |
| MYC (rs11986220) | Various types of cancer | LNCaP + 100nM DHT |
| RET | Hirschsprung | Neuro-2a |
| UC88 | - | Neuro-2a |
| TCF7L2 | Type 2 diabetes | MIN6 |
| ZFAND3 | Type 2 diabetes | MIN6 | 
| ZRS | Limb malformations | NIH-3T3 (with HOXD13/ HOXD13+HAND2) |

## Tasks

### Regression

Measured activity values represent the **difference between** *alternative* and *reference* sequence **activities**.

But the regression task involves predicting scalar values of **regulatory activity** of *alternative* and *reference* sequences for the corresponding cell line.

Therefore, the difference between the predicted alternative and reference sequence activities must be computed.

### Data Representation

```
Element     Cell_Type       Chromosome      Position        Ref     Alt     Value
BCL11A	    HEL92.1.7	        2	           60494939	    C	    -	    -0.34
BCL11A	    HEL92.1.7	        2	           60494939	    C	    A	    -0.05
BCL11A	    HEL92.1.7	        2	           60494939	    C	    G	    -0.13
...
PKLR-24h	K562	            1	           155301804	A	    G   	-0.09
PKLR-24h	K562	            1	           155301804	A	    T   	-0.04
...
TERT-HEK	HEK293T	            5	           1295069	    G	    C   	-0.26	
TERT-HEK	HEK293T	            5	           1295069	    G	    T   	-0.4
...
```

## Parameters

### **split : str, optional**

Specifies how to split the data. Currently only "test" is supported.
Default is "test".

### **length : int, optional**  

Length of the sequence for the differential expression experiment. 
Must be positive integer. Default is 200.

### **elements : Union[list[str], str], optional**

List of promoter-enhancer elements to include. If None, includes all elements.
Can be a single string or list of strings.

### **cell_type : Union[list[str], str], optional**

List of cell types to filter by. If None, includes all elements.
Can be a single string or list of strings.

### **genomic_regions : str | List[Dict], optional**

Genomic regions to include/exclude. Can be:
- Path to BED file
- List of dictionaries with 'chrom', 'start', 'end' keys
- Uses 0-based indexing for genomic coordinates

### **exclude_regions : bool**

If True, exclude the specified regions instead of including them.

### **transform : callable, optional**

Transformation applied to each sequence object.

### **target_transform : callable, optional**

Transformation applied to the target data (**expression values).

### **root : str, optional**

Root directory where data is stored. If None, uses default data directory.

## Data Handling Considerations

1) The data is intended exclusively for validation of machine learning models.

2) The dataset contains information about nucleotide positions in the hg38 genome, including reference and alternative nucleotide variants. For your specific task, use the `length` parameter (default: 200) to extract nucleotide sequences with specified length and the variant nucleotide at the center.

3) When using the dataset, the hg38 genome is automatically loaded if not previously available, and nucleotide sequences of the specified length are extracted with the variant nucleotide positioned at the center.

4) Measured activity values represent the difference between alternative and reference sequence activities.

5) Use the `cell_type` parameter to filter elements from specific cell types.

6) Use the `elements` parameter to select specific regulatory elements.

7) Use the `genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes in the dataset. *Uses 0-based indexing for genomic coordinates.*

8) **Example Usage**:   See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/KircherDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    import mpramnist
    from mpramnist.Kircher.dataset import KircherDataset
    import torch.utils.data as data

    # Print detailed information about available cell types and elements
    KircherDataset.Cell_Type_Map()

>>> {'HepG2': ['F9', 'LDLR', 'LDLR.2', 'SORT1', 'SORT1-flip', 'SORT1.2'],
... 'HeLa': ['FOXE1'],
... 'HEL92.1.7': ['BCL11A', 'GP1BA', 'HBB', 'HBG1'],
... 'HEK293T': ['HNF4A', 'MSMB', 'MYCrs6983267', 'TERT-HEK'],
... 'K562': ['PKLR-24h', 'PKLR-48h'],
... 'SF7996': ['TERT-GAa', 'TERT-GBM', 'TERT-GSc'],
... 'SK-MEL-28': ['IRF4'],
... 'HaCaT': ['IRF6'],
... 'LNCaP': ['MYCrs11986220'],
... 'Neuro-2a': ['RET', 'UC88'],
... 'MIN6': ['TCF7L2', 'ZFAND3'],
... 'NIH-3T3': ['ZRSh-13', 'ZRSh-13h2']
... }
```

### 2) Dataset Creation

```python
     # Load data for specific regulatory elements
     dataset = KircherDataset(elements=['F9', 'HBB', 'LDLR'])
    
     # Load data for specific cell types
     dataset = KircherDataset(cell_type=['HepG2', 'K562'])
    
     # Load data with custom sequence length
     dataset = KircherDataset(length=300, elements='HBB')
    
    # Load data for SORT enhancer
     kircher_dataset = KircherDataset(
         length=200,
         elements=["SORT1.2", "SORT1", "SORT1-flip"],
         transform=forw_transform,
         root="../data/",
     )

     # Load data filtered by genomic regions
     dataset = KircherDataset(
         genomic_regions='path/to/regions.bed',
         elements=['BCL11A', 'IRF4']
     )
```

### 3) Dataloader Creation

```python
    kircher_forw = data.DataLoader(
         dataset=kircher_dataset,
         batch_size=128,
         shuffle=False,
         num_workers=16,
         pin_memory=True,
    )
```

See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Kircher2019/KircherDataset_example.ipynb) for detailed usage example and training

## Launch Parameters

```bash
#MPRALegNet
python3 Kircher_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements LDLR LDLR.2 --result_dir ./Kircher_legnet_ldlr.tsv
python3 Kircher_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements F9 --result_dir ./Kircher_legnet_f9.tsv
python3 Kircher_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements SORT1 SORT1-flip SORT1.2 --result_dir ./Kircher_legnet_sort.tsv
python3 Kircher_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types K562 --elements PKLR-24h PKLR-48h --result_dir ./Kircher_mlegnetpklr.tsv
#Malinois
python3 Kircher_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements LDLR LDLR.2 --result_dir ./Kircher_malinois_ldlr.tsv
python3 Kircher_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements F9 --result_dir ./Kircher_malinois_f9.tsv
python3 Kircher_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements SORT1 SORT1-flip SORT1.2 --result_dir ./Kircher_malinois_sort.tsv
python3 Kircher_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types K562 --elements PKLR-24h PKLR-48h --result_dir ./Kircher_malinois_pklr.tsv
#MPRAnn
python3 Kircher_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements LDLR LDLR.2 --result_dir ./Kircher_mprann_ldlr.tsv
python3 Kircher_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements F9 --result_dir ./Kircher_mprann_f9.tsv
python3 Kircher_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements SORT1 SORT1-flip SORT1.2 --result_dir ./Kircher_mprann_sort.tsv
python3 Kircher_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types K562 --elements PKLR-24h PKLR-48h --result_dir ./Kircher_mprann_pklr.tsv
#PARM
python3 Kircher_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements LDLR LDLR.2 --result_dir ./Kircher_parm_ldlr.tsv
python3 Kircher_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements F9 --result_dir ./Kircher_parm_f9.tsv
python3 Kircher_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements SORT1 SORT1-flip SORT1.2 --result_dir ./Kircher_parm_sort.tsv
python3 Kircher_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types K562 --elements PKLR-24h PKLR-48h --result_dir ./Kircher_parm_pklr.tsv
#DREAM-RNN
python3 Kircher_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements LDLR LDLR.2 --result_dir ./Kircher_dream_rnn_ldlr.tsv
python3 Kircher_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements F9 --result_dir ./Kircher_dream_rnn_f9.tsv
python3 Kircher_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types HepG2 --elements SORT1 SORT1-flip SORT1.2 --result_dir ./Kircher_dream_rnn_sort.tsv
python3 Kircher_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types K562 --elements PKLR-24h PKLR-48h --result_dir ./Kircher_dream_rnn_pklr.tsv
```

## Achieved Performance Using Basic Models

Pearson correlation, r 

| Cell type | Promoter Type | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM-RNN |
|-----------|:---------------:|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|
| HepG2 | SORT1 | #N/A | 0,428 | 0,3901 | 0,314 | 0,4628 | 0,4956 |
| HepG2 | LDLR | #N/A | 0,442 | 0,3024 | 0,3715 | 0,3998 | 0,4298 |
| HepG2 | F9 | #N/A | 0,416 | 0,4314 | 0,4918 | 0,4865 | 0,2417 |
| K562 | PKLR | #N/A | 0,425 | 0,3781 | 0,3812 | 0,425 | 0,134 |


## Citation

When using this dataset, please cite the original publication:

[Kircher et al. 2019](https://www.nature.com/articles/s41467-019-11526-w) 

Kircher, M., Xiong, C., Martin, B. et al. Saturation mutagenesis of twenty disease-associated regulatory elements at single base-pair resolution. Nat Commun 10, 3583 (2019). https://doi.org/10.1038/s41467-019-11526-w

```bibtex
    @article{kircher2019saturation,
        title={Saturation mutagenesis of twenty disease-associated regulatory elements at single base-pair resolution},
        author={Kircher, M., Xiong, C., Martin, B. and others},
        journal={Nat. Commun.},
        volume={10},
        pages={3583},
        year={2019},
        doi={10.1038/s41467-019-11526-w}
    }
```

