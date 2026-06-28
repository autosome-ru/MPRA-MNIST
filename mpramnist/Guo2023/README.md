# Guo Multi and Single dataset

## Main Information

The Guo dataset is based on results from lentiMPRA experiments from [Guo et al. 2023](https://www.nature.com/articles/s41588-023-01533-5). The study characterized **2,221 non-coding single-nucleotide variants (SNVs)** connected to risk for ten neuropsychiatric disorders across multiple neural cell types, identifying 892 differentially-active SNVs (daSNVs) with significant allelic activity differences.

These experimentally characterized sequences are proposed as a benchmark dataset for validating machine learning model quality on regulatory variant effect prediction. Specifically, models can be trained on independent data (e.g., other MPRA datasets) and their predictive power can be evaluated on the Guo MPRA data.




## Experimental Design

The study tested **2,221 non-coding variants** associated with ten neuropsychiatric disorders, including:

*   **MPRA experiment type:** lentivirus-based MPRA (lentiMPRA)
*   **Variant selection:** Variant were chosen from 10 neuropsychiatric GWAS datasets
*   **Negative controls:** 22 blacklisted regions by ENCODE hg19
*   **Interval length:** 145 bp fragments centered on each variant for both reverence and alternate alleles
*   **Variant logFC activity:** estimated with MPRAnalyze
*   **daSNP:** SNVs with allele specific activity were defined as those achieve a | log2(fold-change)| > 0.05 and an FDR-corrected p-value < 0.05


## Data Filtration
1.  Results from MPRA experiment in P-NPC cells were removed due to the lack of the variant ids in the original data
2.  Variants with no provided reference or alternate allele were removed
3.  Variants with provided several alternate alleles were removed


## Available Cell Types

| **Cell Type** | **Description** | **Number of variants tested** | 
| :-----------: | :-----------: | :-----------: |
| AST | Astrocytes | 2088 |
| ES | Embryonic Stem Cell | 2070 |
| N-D2 | ES-derived neural cells, day 2 | 2108 |
| N-D4 | ES-derived neural cells, day 4 | 2112 |
| N-D10 | ES-derived neural cells, day 10 | 2093 |
| A-NPC | Anterior Neural Progenitor Cells | 2122 |
| HEK293T | Immortalized human embryonic kidney cells (control) | 2150 |
| D283 | Medulloblastoma cell line | 1806 |
| D341 | Medulloblastoma cell line | 1812 |
| IMR.prog | Nondifferentiated IMR-32 neuroblastoma cells | 1812 |
| IMR.diff | Differentiated IMR-32 neuroblastoma cells | 1808 |
| SHSY5Y.prog | Nondifferentiated SH-SY5Y neuroblastoma cells | 1810 |
| SHSY5Y.diff | Differentiated SH-SY5Y neuroblastoma cells | 1812 |




## Tasks

### Regression

Measured variant activity is represented as **log2 fold change (logFC)** between alternative and reference allele activities: `logFC = log₂(alt / ref)`.

Activity of individual alleles (`ref` and `alt`) are measured as `log₂(RNA reads / DNA reads)`

Therefore, the difference between the predicted alternative and reference sequence activities must be computed.

### Data Representation

#### GuoSingle

```
rowname	        chrom	  pos	 ref alt	logFC	  pval	       fdr	     statistic	 orig_seq	is_mpra_daSNP	mpra_tissue	            mpra_logfc_mean	    mpra_pval_mean
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
chr4_103750316	chr4   103750316  G	  A	    4.84	2.78e-23	6.13e-20	98.8114838	GTAGTGTCAC	    True	    ND4|P-NPC|HEK293|AST1	-2.24	            0.00067
chr5_159609391	chr5   159609391  G   A	    1.78	5.41e-18    2.98e-15	74.7260713	AAGCAGTAAG	    True	    ND10|AST1	            -1.662	            1.871e-10
chr11_47365014	chr11  47365014	  C	  T	    2.91	0.000488	0.0224593	12.1591861	GCCACACCGA	    True	    ND4|P-NPC|AST1	        -2.1577	            0.00263
```


**Column descriptions:**
*   `logFC`: Variant activity score estimated with MPRAnalyze
*   `pval`: MPRA p-value
*   `fdr`: MPRA p-value after FDR correction
*   `orig_seq`: original sequence from the assay
*   `is_mpra_daSNP`: if the variant has allele specific activity in any of the cell lines
*   `mpra_tissue`: tissues MPRA was significant in
*   `mpra_logfc_mean`: MPRA log2 fold-change (alt/ref), mean across significant tissues
*   `mpra_pval_mean`: MPRA p-value, mean across significant tissues



#### GuoMulti

```
rowname	 chromosome	position	ref	alt	orig_seq	logFC_AST	pval_AST	fdr_AST	statistic_AST    ... logFC_SHSY5Y.diff	pval_SHSY5Y.diff	fdr_SHSY5Y.diff	statistic_SHSY5Y.diff 	is_mpra_daSNP	mpra_tissue	mpra_logfc_mean	mpra_pval_mean
---------------------------------------------------------------------------------------------------------
chr10_104359350	chr10	104359350	T	C	GGCCTCGCCC	-0.00477	0.991	1.0	    0.00013	...	0.4528313	0.173935	0.486600	1.848695	True	ES	        -2.0667	1.966e-07
chr10_104426177	chr10	104426177	G	A	ATTGTGGTTC	-0.47585	0.266	0.671	1.22430 ... -0.164683	0.486945	0.782585	0.483270	True	P-NPC|A-NPC	0.91957	0.003051
```

**Column descriptions:**
*   `logFC_CELL_TYPE`: Variant activity score, measured in `CELLTYPE` and estimated with MPRAnalyze
*   `pval_CELL_TYPE`: MPRA p-value measured in `CELL_TYPE`
*   `fdr_CELL_TYPE`: MPRA p-value after FDR correction measured in `CELL_TYPE`
*   `orig_seq`: original sequence from the assay
*   `is_mpra_daSNP`: if the variant has allele specific activity in any of the cell lines
*   `mpra_tissue`: tissues MPRA was significant in
*   `mpra_logfc_mean`: MPRA log2 fold-change (alt/ref), mean across significant tissues
*   `mpra_pval_mean`: MPRA p-value, mean across significant tissues




## GuoSingle Parameters

### **`cell_type : str`**

Cell type for filtering the data. Must be one of the listed in `Available Cell Types`

### **split : str, optional**

Specifies how to split the data. Currently only "test" is supported.
Default is "test".

### **length : int, optional**  

Length of the sequence for the differential expression experiment. 
Must be positive integer. Default is 145.


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



## GuoMulti Parameters

### **split : str, optional**

Specifies how to split the data. Currently only "test" is supported.
Default is "test".

### **length : int, optional**  

Length of the sequence for the differential expression experiment. 
Must be positive integer. Default is 145.

### **cell_types : Union[list[str], str], optional**

List of cell types to filter by. If None, includes all cell_types.
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

2) The dataset contains information about nucleotide positions in the hg19 genome, including reference and alternative nucleotide variants. For your specific task, use the `length` parameter (default: 145) to extract nucleotide sequences with specified length and the variant nucleotide at the center. However, please note that the effects are measured in an MPRA assay using sequences of length 145. Thus, changing this parameter will likely result in a biologically irrelevant prior for the model.

3) When using the dataset, the hg19 genome is automatically loaded if not previously available, and nucleotide sequences of the specified length are extracted with the variant nucleotide positioned at the center.

4) Measured activity values represent the difference between alternative and reference sequence activities.

5) Use the `cell_types` parameter to filter elements from specific cell types.

6) Use the `genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes in the dataset. *Uses 0-based indexing for genomic coordinates.*

7) **Example Usage**:   See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/GuoDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    import mpramnist
    from mpramnist.Guo2023.dataset import GuoMultiDataset
    import torch.utils.data as data

    # Get list ov available cell types
    GuoMultiDataset.CELL_TYPES

>>> ['AST', 'ES', 'N-D2', 'N-D4', 'N-D10', 'A-NPC', 'D283', 'D341', 'IMR.diff', 'IMR.prog', 'SHSY5Y.diff', 'SHSY5Y.prog', 'HEK293T']
```

### 2) Dataset Creation

```python
     # Load whole dataset
     dataset = GuoMultiDataset()
    
     # Load data for specific cell types
     dataset = GuoMultiDataset(cell_types=['AST', 'SHSY5Y.diff'])
    
     # Load data with custom sequence length
     dataset = GuoMultiDataset(length=200, cell_types='AST')
    
    # Load data with specified transform
     Guo_dataset = GuoMultiDataset(
         length=200,
         cell_types=['AST', 'SHSY5Y.diff'],
         transform=forw_transform,
         root="../data/",
     )

     # Load data filtered by genomic regions
     dataset = GuoMultiDataset(
         genomic_regions='path/to/regions.bed',
         cell_types=['AST', 'SHSY5Y.diff']
     )
```

### 3) Dataloader Creation

```python
    guo_forw = data.DataLoader(
         dataset=Guo_dataset,
         batch_size=128,
         shuffle=False,
         num_workers=16,
         pin_memory=True,
    )
```

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/GuoDataset_example.ipynb) for detailed usage example and training

## Launch Parameters

```bash
    #MpraLegNet
    python3 Guo_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types AST --result_dir ./Guo_AST_legnet.tsv
    #Malinois
    python3 Guo_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types AST --result_dir ./Guo_AST_malinois.tsv
    #MPRAnn
    python3 Guo_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types AST --result_dir ./Guo_AST_mprann.tsv
    #PARM
    python3 Guo_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types AST --result_dir ./Guo_AST_parm.tsv
    #DREAM_RNN
    python3 Guo_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types AST --result_dir ./Guo_AST_dreamrnn.tsv
```

## AlphaGenome Quality


| **cell_type** | **n_all** | **pearsonr_all** | **n_daSNP** | **pearsonr_daSNP** |
|-----------|-------|--------------|---------|----------------|
| AST       | 2087  | -0.0270    | 65      | -0.1653      |
| ES        | 2069  | -0.0211    | 104     | -0.1860      |
| N-D2      | 2107  | -0.0019    | 84      | 0.02943       |
| N-D4      | 2111  | -0.0158    | 186     | -0.0670      |
| N-D10     | 2092  | -0.0295    | 149     | -0.0955      |
| A-NPC     | 2121  | -0.0175    | 71      | -0.0494      |
| D283      | 1779  | 0.0432     | 70      | 0.2248       |
| D341      | 1787  | 0.0093     | 150     | 0.1489       |
| IMR.diff  | 1783  | 0.0037     | 188     | 0.0744       |
| IMR.prog  | 1789  | 0.0038     | 269     | 0.1366       |
| SHSY5Y.diff | 1790 | 0.0093    | 173     | 0.1154       |
| SHSY5Y.prog | 1785 | 0.0014    | 112     | 0.0780       |
| HEK293T   | 2149  | -0.0422    | 269     | -0.0972      |


## Achieved Quality Using LegNet Model trained on Gosai SK-N-SH data in MPRA-MNIST

| **cell_type** | **n_all** | **pearsonr_all** | **n_daSNP** | **pearsonr_daSNP** |
|---------------|-----------|------------------|-------------|--------------------|
| AST           | 2087      | -0.0459          | 65          | -0.1084            |
| ES            | 2069      | -0.0134          | 104         | -0.0632            |
| N-D2          | 2107      | -0.0495          | 84          | -0.1539            |
| N-D4          | 2111      | -0.0386          | 186         | -0.1316            |
| N-D10         | 2092      | -0.0652          | 149         | -0.1559            |
| A-NPC         | 2121      | -0.0297          | 71          | -0.0729            |
| D283          | 1779      | 0.0842           | 70          | 0.1808             |
| D341          | 1787      | 0.0384           | 150         | 0.1125             |
| IMR.diff      | 1783      | 0.0538           | 188         | 0.0718             |
| IMR.prog      | 1789      | 0.0224           | 269         | 0.1056             |
| SHSY5Y.diff   | 1790      | 0.0369           | 173         | 0.0320             |
| SHSY5Y.prog   | 1785      | 0.0244           | 112         | 0.0818             |
| HEK293T       | 2149      | -0.0460          | 269         | -0.0987            |

## Citation

When using this dataset, please cite the original publication:

[Guo et al. 2023](https://www.nature.com/articles/s41588-023-01533-5)

Guo, M.G., Reynolds, D.L., Ang, C.E. et al. Integrative analyses highlight functional regulatory variants associated with neuropsychiatric diseases. Nat Genet 55, 1876–1891 (2023). https://doi.org/10.1038/s41588-023-01533-5

```bibtex
    @article{guo2023integrative,
        title={Integrative analyses highlight functional regulatory variants associated with neuropsychiatric diseases},
        author={Guo, M.G., Reynolds, D.L., Ang, C.E and others},
        journal={Nature Genetics},
        volume={55},
        pages={1876--1891},
        year={2023},
        doi={10.1038/s41588-023-01533-5}
    }
```
