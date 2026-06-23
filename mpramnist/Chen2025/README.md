# Chen Multi and Single dataset

## Main Information

The Chen dataset is based on results from MPRA experiments from [Chen et al. 2025](https://doi.org/10.1101/2025.07.11.659973). The study characterized **599 non-coding single-nucleotide variants (SNVs)** associated with Late-onset Alzheimer's disease (LOAD), tested in **855 constructs** across immune and neural contexts. The study identified expression-modulating variants (emVars) with significant allelic regulatory activity in THP-1 macrophages, HMC3 microglia-like cells, and mouse brain tissue (cortex, hippocampus, and striatum).

These experimentally characterized sequences are proposed as a benchmark dataset for validating machine learning model quality on regulatory variant effect prediction. Specifically, models can be trained on independent data (e.g., other MPRA datasets) and their predictive power can be evaluated on the Chen MPRA data.

## Experimental Design

The study tested **599 non-coding variants** associated with Late-onset Alzheimer's disease, including:

- **Variant selection:** 599 variants (186 rare, 106 low-frequency, 307 common) from LOAD GWAS and WGS studies
- **Negative controls:** sequences with 25bp motif-shuffled versions of the CREs
- **Interval length:** 227 bp fragments centered on either the variant or the open chromatin peak summit
- **emVars:** Expression-modulating variants with significant allelic regulatory activity (FDR < 0.05, MPRAnalyze comparative analysis)


## Available Cell Types and States

| **Cell Type** | **State** | **Description** |
| :-----------: | :-----------: | :-----------: |
| THP1 | Naive | no stimulation |
| THP1 | IFNB | stimulation with interferon-β (IFN-β), modeling antiviral state |
| THP1 | IFNG | stimulation with interferon-γ (IFN-γ), modeling M1-polarized state |
| THP1 | LPSIFNG | stimulation with interferon-γ (IFN-γ) and lipopolysaccharide (LPS), modeling hyperinflammatory state |
| THP1 | aggregated | averaged data from all other states |
| HMC3 | Naive | no stimulation |
| HMC3 | IFNB | stimulation with interferon-β (IFN-β), modeling antiviral state |
| HMC3 | IFNG | stimulation with interferon-γ (IFN-γ), modeling M1-polarized state |
| HMC3 | LPSIFNG | stimulation with interferon-γ (IFN-γ) and lipopolysaccharide (LPS), modeling hyperinflammatory state |
| HMC3 | aggregated | averaged data from all other states |
| Brain | Cortex | microdissected mouse brain region |
| Brain | Hippocampus | microdissected mouse brain region |
| Brain | Striatum | microdissected mouse brain region |
| Brain | aggregated | averaged data from all other states |


## Types if constructions used in the assay

*   **`PEAKCENTER`: (n = 611)** if the variant overlapped an open chromatin peak (identified via ATAC-seq assay), in immune cells or neurons, the summit served as the center of the CREs
*   **`SNPCENTER`: (n = 222)** if the variant did not overlap any peak, SNP-centered CREs were designed


## Tasks

### Regression

Measured variant activity is represented as **log2 fold change (logFC)** between alternative and reference allele activities: `logFC = log₂(alt / ref)`.

Activity of individual alleles (`ref` and `alt`) are measured as `log₂(RNA reads / DNA reads)`

Therefore, the difference between the predicted alternative and reference sequence activities must be computed.

### Data Representation

#### GuoSingle

```
RSID	interval_type	chromosome	pos_hg38	ref	alt	hg	snp_position	interval_center	reverse_prediction	logFC	pval	fdr	statistic	orig_seq
----------------------------------------------------------------------------------------------------------------------------------------------------------
cg03073402	SNPCENTER	chr19	42423524	C	G	hg19	42927676	42927676	1	0.302907503	    0.031393473	0.220011637	4.631316097	GCGCTCACCTTTGGC
cg03169557	SNPCENTER	chr16	89532542	C	G	hg19	89598950	89598950	1	-0.271034107	0.09562602	0.354794992	2.77701875	TGCCCATTTCCTGAT
cg05030077	SNPCENTER	chr16	2205198	    C	G	hg19	2255199     2255199	    1	0.229937977	    0.218614771	0.510698441	1.513440994	CTGGATATACTTACA
cg05066959	SNPCENTER	chr8	41661790	C	G	hg19	41519308	41519308	1	-0.211632217	0.201382476	0.486389879	1.632319882	AGAAGAGAGACTGGA
```


**Column descriptions:**
*   `interval_type`: type of the construction with the variant (PEAKCENTER or SNPCENTER)
*   `logFC`: Variant activity score  *log₂(alt/ref)*
*   `pval`: MPRA p-value
*   `fdr`: MPRA p-value after FDR correction
*   `orig_seq`: original sequence from the assay
*   `pos_hg38`:  SNP position in hg38 human genome version
*   `hg`: human genome version used in `snp_position` and `interval_center`
*   `snp_position`: SNP position in human genome version specified in `hg`
*   `interval_center`: interval center position in human genome version specified in `hg`
*   `reverse_prediction`: If the prediction sign should be inverted (ref allele is non-effect for GWAS)



#### GuoMulti

```
RSID	interval_type	chromosome	pos_hg38	ref	alt	hg	snp_position	interval_center	reverse_prediction	logFC_THP1_aggregated	pval_THP1_aggregated	fdr_THP1_aggregated	statistic_THP1_aggregated	...	logFC_Brain_Striatum	pval_Brain_Striatum	fdr_Brain_Striatum	statistic_Brain_Striatum	orig_seq
cg03073402	SNPCENTER	chr19	42423524	C	G	hg19	42927676	42927676	1	 0.037138205	0.112539914	0.198395106	2.518186894	...	0.465511869	0.006795306	0.106106103	7.32629303	GCGCTCACCT
cg03169557	SNPCENTER	chr16	89532542	C	G	hg19	89598950	89598950	1	-0.132102404	6.44e-08	5.92e-07	29.22552823	...	-0.426873955	0.0098101	0.12518859	6.669058466	TGCCCATTTC
rs10030602	PEAKCENTER	chr4	112086886	A	G	hg38	112086886	112086918	1	-0.000870249	0.971510189	0.983007351	0.001275509	...	-0.171230307	0.26644094	0.587033649	1.234978908	GCACCTCGCT
```


**Column descriptions:**

*   `interval_type`: type of the construction with the variant (PEAKCENTER or SNPCENTER)
*   `logFC_CELLTYPE_STATE`: Variant activity score  *log₂(alt/ref)* measured in `CELLTYPE` in the state `STATE`
*   `pval_CELLTYPE_STATE`: MPRA p-value measured in `CELLTYPE` in the state `STATE`
*   `fdr_CELLTYPE_STATE`: MPRA p-value after FDR correction measured in `CELLTYPE` in the state `STATE`
*   `orig_seq`: original sequence from the assay
*   `pos_hg38`:  SNP position in hg38 human genome version
*   `hg`: human genome version used in `snp_position` and `interval_center`
*   `snp_position`: SNP position in human genome version specified in `hg`
*   `interval_center`: interval center position in human genome version specified in `hg`
*   `reverse_prediction`: If the prediction sign should be inverted (ref allele is non-effect for GWAS)



## ChenSingle Parameters

### **`cell_type : str`**

Cell type for filtering the data. Must be one of the listed in `Available Cell Types`

### **`state : str`**

State of the specified cell type. Defalt is `aggregated`.

### **split : str, optional**

Specifies how to split the data. Currently only "test" is supported.
Default is "test".

### **length : int, optional**  

Length of the sequence for the differential expression experiment. 
Must be positive integer. Default is 227.

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



## ChenMulti Parameters

### **split : str, optional**

Specifies how to split the data. Currently only "test" is supported.
Default is "test".

### **length : int, optional**  

Length of the sequence for the differential expression experiment. 
Must be positive integer. Default is 227.

### **cell_types : Union[list[str], str], optional**

List of cell types to filter by. If None, includes all cell_types.
Can be a single string or list of strings.

### **states: Union[list[list[str]], list[str]], optional**

List of states to use for every cell type. If `cell_types` are
specified, list of states must be provided for every cell type. 
If one wants to use all available states for the specified cell 
type, use `state` = `all`.

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

2) The dataset contains information about nucleotide positions in the hg19 genome, including reference and alternative nucleotide variants. For your specific task, use the `length` parameter (default: 145) to extract nucleotide sequences with specified length and the variant nucleotide at the center.

3) When using the dataset, the hg19 genome is automatically loaded if not previously available, and nucleotide sequences of the specified length are extracted with the variant nucleotide positioned at the center.

4) Measured activity values represent the difference between alternative and reference sequence activities.

5) Use the `cell_types` parameter to filter specific cell types. Use `states` to filter specific state of the cell types.

6) Use the `genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes in the dataset. *Uses 0-based indexing for genomic coordinates.*

7) **Example Usage**:   See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/GuoDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    import mpramnist
    from mpramnist.Chen2025.dataset import ChenMultiDataset
    import torch.utils.data as data

    # Get list ov available cell types
    ChenMultiDataset.CELL_TYPES

>>> {'THP1': ['aggregated', 'Naive', 'IFNB', 'IFNG', 'LPSIFNG'],
... 'HMC3': ['aggregated', 'Naive', 'IFNB', 'IFNG', 'LPSIFNG'],
... 'Brain': ['aggregated', 'Cortex', 'Hippocampus', 'Striatum']
... }
```

### 2) Dataset Creation

```python
     # Load whole dataset
     dataset = ChenMultiDataset()
    
     # Load data for specific cell types
     dataset = ChenMultiDataset(cell_types=['THP1', 'HMC3'])

     # Load data for specific cell types and state
     dataset = ChenMultiDataset(cell_types=['THP1', 'HMC3'],
                                states = ['IFNB', 'IFNG'])
    
     # Load data with custom sequence length
     dataset = ChenMultiDataset(length=200, cell_types='Brain')
    
    # Load data with specified transform
     Chen_dataset = ChenMultiDataset(
         length=200,
         cell_types='Brain',
         targets='Cortex'
         transform=forw_transform,
         root="../data/",
     )

     # Load data filtered by genomic regions
     dataset = ChenMultiDataset(
         genomic_regions='path/to/regions.bed',
         cell_types='Brain'
     )
```

### 3) Dataloader Creation

```python
    chen_forw = data.DataLoader(
         dataset=Chen_dataset,
         batch_size=128,
         shuffle=False,
         num_workers=16,
         pin_memory=True,
    )
```

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/ChenDataset_example.ipynb) for detailed usage example and training

## Launch Parameters

```bash
    #MpraLegNet
    python3 Chen_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types Brain --states aggregated --result_dir ./Guo_AST_legnet.tsv
    #Malinois
    python3 Chen_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types Brain --states aggregated --result_dir ./Guo_AST_malinois.tsv
    #MPRAnn
    python3 Chen_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types Brain --states aggregated --result_dir ./Guo_AST_mprann.tsv
    #PARM
    python3 Chen_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types Brain --states aggregated --result_dir ./Guo_AST_parm.tsv
    #DREAM_RNN
    python3 Chen_model_launch.py --model DREAM_RNN --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_types Brain --states aggregated --result_dir ./Guo_AST_dreamrnn.tsv
```

## Original Benchmark Quality

Pearson correlation, r 

    r = 0.49 for SORT1 (HepG2)
    
    r = 0.65 for PKLR (K562)

    r = 0.66 for LDLR (HepG2)
    
    r = 0.51 for F9 (HepG2)


## Achieved Quality Using LegNet Model in MPRA-MNIST

Pearson correlation, r

    r = 0.4 for SORT1 (HepG2)
    
    r = 0.54 for PKLR (K562)

    r = 0.66 for LDLR (HepG2)
    
    r = 0.52 for F9 (HepG2)


## Citation

When using this dataset, please cite the original publication:

[Chen et al. 2025](https://doi.org/10.1101/2025.07.11.659973)

Chen, Ziheng and Liu, Yaxuan and Brown, Ashley R. et al. Context-dependent regulatory variants in Alzheimer's disease. bioRxiv 2025.07.11.659973. ; doi: https://doi.org/10.1101/2025.07.11.659973

```bibtex
    @article{chen2025context,
        title={Context-dependent regulatory variants in Alzheimer{\textquoteright}s disease},
        author={Chen, Ziheng and Liu, Yaxuan and Brown, Ashley R. and others},
        journal={bioRxiv},
        year={2025},
        doi={10.1101/2025.07.11.659973}
    }
```
