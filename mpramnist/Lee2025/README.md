# Lee dataset

## Main Information

The Lee dataset is based on results from massively parallel reporter assay (MPRA) experiments from [Lee et al. 2023](https://doi.org/10.1016/j.cell.2024.12.022). The study characterized **13,261 non-coding single-nucleotide variants (SNVs)** connected to risk for eight psychiatric disorders, conducted in human neural progenitor cells (HNPs). The experiment identified **1,461 expression-modulating variants (emVars)** with significant allelic activity differences.

These experimentally characterized sequences are proposed as a benchmark dataset for validating machine learning model quality on regulatory variant effect prediction. Specifically, models can be trained on independent data (e.g., other MPRA datasets) and their predictive power can be evaluated on the Lee MPRA data.


## Experimental Design

The study tested **13,261 non-coding variants** associated with eight psychiatric disorders, including:

*   **Cell type:** Human neural progenitor cells (HNPs) derived from fetal brain tissue
*   **Variant selection:** Variants were chosen from 8 psychiatric GWAS datasets
*   **Negative controls:** scrambled DNA sequences with matching GC content to the MPRA library
*   **Positive controls:** sections of the cytomegalovirus (CMV) and elongation factor 1 alpha (EF1a) promoters a
*   **Interval length:** 150 bp fragments centered on each variant for both risk and protective alleles
*   **emVar:** 683 variants showing significant allelic regulatory activity (FDR < 0.05) located in MPRA‑active elements


## Types of Variants identified in the assay

| **Variant class** | **Number of variants** | **Description** | 
| :-----------: | :-----------: | :-----------: |
| emVar | 683 | Variants that exhibit significant differential regulatory activity between risk and protective alleles (FDR<0.05) and are located in MPRA-active elements |
| MPRA-Allelic | 3789 | Variants that exhibit significant differential regulatory activity between risk and protective alleles (FDR<0.05) |
| MPRA-nonallelic | 6396 | Variants that do not exhibit significant differential regulatory activity between risk and protective alleles (p>0.1) |
| Uncertain | 2393 | others |


## Tasks

### Regression

Measured variant activity is represented as **log2 fold change (logFC)** between alternative and reference allele activities: `logFC = log₂(alt / ref)`.

Activity of individual alleles (`ref` and `alt`) are measured as `log₂(RNA reads / DNA reads)`

Therefore, the difference between the predicted alternative and reference sequence activities must be computed.

### Data Representation


```
chromosome	position_hg19	position_hg38	RSID	ref	alt	reverse_prediction	MPRA_logFC	MPRA_FDR	Variant_Class	MPRA_AveExpr	MPRA_t	MPRA_P	MPRA_B	GWAS_P	GWAS_OR	GWAS_SE	GWAS_frq
chr4	42086449	42084432	rs10001295	T	C	1	0.0183765	0.864804	MPRA-nonallelic	-0.9206216	0.267494	0.789765	-7.7310295	7.57e-08	0.967436	0.0061568	0.182
chr4	42087546	42085529	rs10002107	A	G	1	-0.128632	0.337966	MPRA-nonallelic	-1.00618709	-1.258928	0.211651	-6.538996	5.08e-09	0.970612	0.0051038	0.207
chr4	80225200	79304046	rs10004612	C	A	-1	-0.5140376	2.616e-09	MPRA-Allelic	-1.01212085	-7.584262	4.776e-11	14.201189	6.71e-06	1.020668	0.0045405	0.474
chr4	42178259	42176242	rs10005662	C	T	-1	-0.1746818	0.018632	MPRA-Allelic	-0.8873500  -2.88519	0.0050041	-3.790045	1.88e-06	0.973096	0.0057230	0.141
```


**Column descriptions:**
*   `reverse_prediction`: If the prediction sign should be inverted (ref allele is non-effect for GWAS)
*   `Variant_Class`: Different classes of variants
*   `MPRA_logFC`: Variant activity score  *log₂(alt/ref)*
*   `MPRA_FDR`: MPRA p-value after FDR correction
*   `MPRA_AveExpr`: MPRA average expression value
*   `MPRA_t`: MPRA t value
*   `MPRA_P`: MPRA P value
*   `MPRA_B`: MPRA B value
*   `GWAS_P`: MPRA p-value, mean across significant tissues
*   `GWAS_OR`: MPRA p-value, mean across significant tissues
*   `GWAS_SE`: MPRA p-value, mean across significant tissues
*   `GWAS_frq`: Frequency of the Major allele in GWAS



## Parameters


### **split : str, optional**

Specifies how to split the data. Currently only "test" is supported.
Default is "test".

### **length : int, optional**  

Length of the sequence for the differential expression experiment. 
Must be positive integer. Default is 150.


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

2) The dataset contains information about nucleotide positions in the hg38 genome, including reference and alternative nucleotide variants. For your specific task, use the `length` parameter (default: 150) to extract nucleotide sequences with specified length and the variant nucleotide at the center.

3) When using the dataset, the hg38 genome is automatically loaded if not previously available, and nucleotide sequences of the specified length are extracted with the variant nucleotide positioned at the center.

4) Measured activity values represent the difference between alternative and reference sequence activities.

5) Use the `genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes in the dataset. *Uses 0-based indexing for genomic coordinates.*

6) **Example Usage**:   See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/LeeDataset_example.ipynb) for detailed usage example and training


## Examples

### 1) Import Important Packages

```python
    import mpramnist
    from mpramnist.Lee2025.dataset import LeeDataset
    import torch.utils.data as data
```

### 2) Dataset Creation

```python
     # Load whole dataset
     dataset = dataset = LeeDataset()
    
     # Load data with custom sequence length
     dataset = dataset = LeeDataset(length=200)
    
    # Load data with specified transform
     Lee_dataset = LeeDataset(
         length=200,
         transform=forw_transform,
         root="../data/",
     )

     # Load data filtered by genomic regions
     dataset = LeeDataset(
         genomic_regions='path/to/regions.bed',
         root="../data/",
     )
```

### 3) Dataloader Creation

```python
    lee_forw = data.DataLoader(
         dataset=Lee_dataset,
         batch_size=128,
         shuffle=False,
         num_workers=16,
         pin_memory=True,
    )
```

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/LeeDataset_example.ipynb) for detailed usage example and training

## Launch Parameters

```bash
    #MpraLegNet
    python3 Lee_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --result_dir ./Lee_legnet.tsv
    #Malinois
    python3 Lee_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --result_dir ./Lee_malinois.tsv
    #MPRAnn
    python3 Lee_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --result_dir ./Lee_mprann.tsv
    #PARM
    python3 Lee_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --result_dir ./Lee_parm.tsv
    #DREAM_RNN
    python3 Lee_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --result_dir ./Lee_dreamrnn.tsv
```


## Achieved Quality Using LegNet Model in MPRA-MNIST

Pearson correlation, r

    r = 0.4 for SORT1 (HepG2)
    
    r = 0.54 for PKLR (K562)

    r = 0.66 for LDLR (HepG2)
    
    r = 0.52 for F9 (HepG2)


## Citation

When using this dataset, please cite the original publication:

[Lee et al. 2025](https://doi.org/10.1016/j.cell.2024.12.022)

Lee S, McAfee J, Lee J et al. Massively parallel reporter assay investigates shared genetic variants of eight psychiatric disorders Cell, 2025; 188, 1409-1424.e21 . https://doi.org/10.1016/j.cell.2024.12.022

```bibtex
    @article{lee2025massively,
        title={Massively parallel reporter assay investigates shared genetic variants of eight psychiatric disorders},
        author={Lee S, McAfee J, Lee J and others},
        journal={Cell},
        volume={188},
        pages={1409--1424},
        year={2025},
        doi={10.1016/j.cell.2024.12.022}
    }
```

