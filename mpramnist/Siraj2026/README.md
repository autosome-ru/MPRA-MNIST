# Siraj MPRA and SatMut dataset

## Main Information

The Siraj dataset is based on results from saturating mutagenesis MPRA experiments ([Siraj et al. 2026](https://doi.org/10.1038/s41586-026-10121-6)). The study focuses on functional dissection of complex trait variants at single-nucleotide resolution across relevant cell types.

These experimentally characterized sequences are proposed as a benchmark dataset for validating machine learning model quality. Specifically, models can be trained on independent data (e.g., Agarwal dataset) and their predictive power can be evaluated on the Siraj MPRA data or SatMut data (see [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/mpramnist/Siraj2026/SirajDataset_example.ipynb)).

## Tasks

### Regression

Measured activity values represent the **difference between** *alternative* and *reference* sequence **activities**.

But the regression task involves predicting scalar values of **regulatory activity** of *alternative* and *reference* sequences for the corresponding cell line.

Therefore, the difference between the predicted alternative and reference sequence activities must be computed.

## Data Representation

### SirajMPRA
```
Variant	Active (any cell-type)	emVar (any cell-type)	allele1 log2FC activity in A549	allele1 log2FC activity in HEPG2	allele1 log2FC activity in K562	allele1 log2FC activity in SKNSH	allele1 log2FC activity in HCT116	allele1 log2FC activity SE in A549	allele1 log2FC activity SE in HEPG2	allele1 log2FC activity SE in K562	allele1 log2FC activity SE in SKNSH	allele1 log2FC activity SE in HCT116	allele2 log2FC activity in A549	allele2 log2FC activity in HEPG2	allele2 log2FC activity in K562	allele2 log2FC activity in SKNSH	allele2 log2FC activity in HCT116	allele2 log2FC activity SE in A549	allele2 log2FC activity SE in HEPG2	allele2 log2FC activity SE in K562	allele2 log2FC activity SE in SKNSH	allele2 log2FC activity SE in HCT116	log2FCactivity in A549	log2FC activity in HEPG2	log2FC activity in K562	log2FC activity in SKNSH	log2FC activity in HCT116	active in A549	active in HEPG2	active in K562	active in SKNSH	active in HCT116	logPadj activity in A549	logPadj activity in HEPG2	logPadj activity in K562	logPadj activity in SKNSH	logPadj acitivty in HCT116	log2Skew in A549	log2Skew in HEPG2	log2Skew in K562	log2Skew in SKNSH	log2Skew in HCT116	emVar in A549	emVar in HEPG2	emVar in K562	emVar in SKNSH	emVar in HCT116	log2Skew SE in A549	log2Skew.SE_HEPG2	log2Skew SE in K562	log2Skew SE in SKNSH	log2Skew SE in HCT116	log2Skew FDR in A549	log2Skew FDR in HEPG2	log2Skew FDR in K562	log2Skew FDR in SKNSH	log2Skew FDR in HCT116	mean Plasmid counts ref A549	mean Plasmid counts ref HEPG2	mean Plasmid counts ref K562	mean Plasmid counts ref SKNSH	mean Plasmid counts ref HCT116	mean Plasmid counts alt A549	mean Plasmid counts alt HEPG2	mean Plasmid counts alt K562	mean Plasmid counts alt SKNSH	mean Plasmid counts alt HCT116	mean RNA counts ref A549	mean RNA counts ref HEPG2	mean RNA counts ref K562	mean RNA counts ref SKNSH	mean RNA counts ref HCT116	mean RNA counts alt A549	mean RNA counts alt HEPG2	mean RNA counts alt K562	mean RNA counts alt SKNSH	mean RNA counts alt HCT116
---------------------------------------------------------------------------------------------------------
chr10:100025816:G:A	1	0	NA	1.62368480591055	2.0447887371547702	1.22237033929226	1.28057510025318	NA	0.15552063234708599	0.18549814525563499	0.35952673474110503	0.230448001150133	NA	1.6492259754337699	1.95042069404111	1.80756932748472	1.16212678747709	NA	0.15107758716860101	0.18309336533447301	0.185257477445082	0.15192520413653199	NA	1.6492259754337699	2.0447887371547702	1.80756932748472	1.28057510025318	NA	1	1	1	1	NA	21.923012460420502	22.140538950908301	16.670578312332601	8.6008661627640208	NA	0.020770509382126701	-0.094830509150156797	0.72445387843656595	-0.047430893063247402	NA	0	0	0	0	NA	0.19577697320728399	0.19607772193113501	0.43927852130596101	0.233934791800915	NA	0.010412947018157301	0.063060032697131604	0.42445802862322302	0.018760560812404299	NA	242.80000000000001	290.80000000000001	242.80000000000001	242.80000000000001	NA	378.39999999999998	417.60000000000002	378.39999999999998	378.39999999999998	NA	453	593.16666666666595	305.39999999999998	459	NA	719.60000000000002	796	599.39999999999998	656.20000000000005
chr10:100029561:C:T	1	1	NA	3.3905244404747599	0.29639518683826499	3.40250320796825	2.16276590335222	NA	0.088808119715101899	0.15257185411103999	0.127544137240267	0.095213468266553303	NA	2.6104014956772601	0.44305532506765799	2.6516649802313901	1.61986898958444	NA	0.095685635897149401	0.17796094440857199	0.16487899957972399	0.090494022581615499	NA	3.3905244404747599	0.44305532506765799	3.40250320796825	2.16276590335222	NA	1	0	1	1	NA	301.853225082721	0	150.967330753016	108.402368064279	NA	-0.78361729078232101	0.16183141918763899	-0.76991334513551102	-0.53497948027586695	NA	1	0	1	1	NA	0.089438731574886396	0.22753940111524201	0.14855671727530501	0.10847148228207699	NA	15.7296722340875	0.10601151836632	5.0877297798017196	4.4391647450985499	NA	1023	1091.4000000000001	1023	1023	NA	600.79999999999995	660	600.79999999999995	600.79999999999995	NA	6677.1999999999998	665.16666666666595	5250	3554.4000000000001	NA	2298.8000000000002	439.33333333333297	1873.8	1458.4000000000001
chr10:100098716:G:T	1	1	NA	1.17430855457731	0.85327903412445605	1.2839107990541101	1.00021764287257	NA	0.0714215329408769	0.11293629510818499	0.066360872711100694	0.080770916442450796	NA	1.06394914464357	0.74897789550029004	1.68878410377106	0.92836768873056397	NA	0.0673321641958263	0.20826050485299599	0.090971143878774002	0.10602699509963801	NA	1.17430855457731	0.85327903412445605	1.68878410377106	1.00021764287257	NA	1	0	1	1	NA	54.934452604337501	7.9900281187092403	77.584686841156397	29.408679296800901	NA	-0.094788411111564302	-0.16773707670065499	0.413191246900944	-0.063968588736469101	NA	0	0	1	0	NA	0.112230966125328	0.16984376588506001	0.13769224182221601	0.13776367107508899	NA	0.13337562656482899	0.172399573473765	1.46270074665943	0.046877501411075502	NA	1664	1276.8	1389.5999999999999	1664	NA	652	448.80000000000001	482.19999999999999	652	NA	3417.4000000000001	1131.6666666666599	2423	2850	NA	1243.4000000000001	374.33333333333297	1114.2	1060.4000000000001
```

### SirajSatMut
```
ID	Cell Type	sat_ref_parent	sat_ref	allele	var1	pos1	var2	pos2	centervar	window	chr	pos	ref	alt	ref2	alt2	mut_pos	mut_base	is_haplo	oligomut	indel	DNA_mean	log2FC	log2FC_SE	padj	var1_emVar	var2_emVar	is_var1	is_var2	var_emVar	log2FC_baseline	log2FC_SE_baseline	log2Skew	log2Skew_SE	post_log2Skew	post_log2FC	log2Skew_pval	log2Skew_fdr	phylop_241m	conserved	phylop_241m_trunc
---------------------------------------------------------------------------------------------------------
1:11713563:G:A:A:wC:m0	HEPG2	1:11713563:G:A:wC	1:11713563:G:A:A:wC	A	chr1:11713563:G:A	11713563	NA	NA	var1	wC	1	11713563	G	A	NA	NA	100	G	0	m0	NA	157.40000000000001	0.90405001121226303	0.68096176627523397	0.208489590299588	1	0	1	NA	chr1:11713563:G:A	1.6404904147024899	0.0056577765346911097	-0.73644040349023099	0.680985269711469	-0.164633713225254	1.4758567014772399	0.37216845657893499	0.99916892637583699	-1.3819999694824201	0	0
1:11713563:G:A:A:wC:mA100C	HEPG2	1:11713563:G:A:wC	1:11713563:G:A:A:wC	A	chr1:11713563:G:A	11713563	NA	NA	var1	wC	1	11713563	G	A	NA	NA	100	C	0	mA100C	NA	639	1.7341712070173501	0.207854804785812	1.12e-16	1	0	1	NA	chr1:11713563:G:A	1.6404904147024899	0.0056577765346911097	0.093680792314855196	0.20793179244133	0.060492391924824099	1.7009828066273101	0.83722648644443898	0.99916892637583699	-1.3819999694824201	0	0
1:11713563:G:A:A:wC:mA100G	HEPG2	1:11713563:G:A:wC	1:11713563:G:A:A:wC	A	chr1:11713563:G:A	11713563	NA	NA	var1	wC	1	11713563	G	A	NA	NA	100	G	0	mA100G	NA	235.59999999999999	2.8518988393587001	0.24129898544786799	5.40e-32	1	0	1	NA	chr1:11713563:G:A	1.6404904147024899	0.0056577765346911097	1.2114084246562	0.24136530573694001	0.78939683651510395	2.4298872512175902	0.013671778474534	0.59039156435409701	-1.3819999694824201	0	0
1:11713563:G:A:A:wC:mA100T	HEPG2	1:11713563:G:A:wC	1:11713563:G:A:A:wC	A	chr1:11713563:G:A	11713563	NA	NA	var1	wC	1	11713563	G	A	NA	NA	100	T	0	mA100T	NA	796.79999999999995	1.9145378110734499	0.185437598837429	9.05e-25	1	0	1	NA	chr1:11713563:G:A	1.6404904147024899	0.0056577765346911097	0.27404739637095499	0.18552388929166899	0.20381831869915201	1.8443087334016399	0.52461543150583101	0.99916892637583699	-1.3819999694824201	0	0
```

## SirajMPRA Parameters

### **cell_type : str**

Cell type to filter by. Must be one of: `'K562'`, `'HEPG2'`, `'A549'`, `'SKNSH'`, `'HCT116'`.

### **split : str, optional**

Specifies how to split the data. Currently only "test" is supported.
Default is "test".

### **length : int, optional**

Length of the sequence for the differential expression experiment.
Must be positive integer. Default is 200.

### **genomic_regions : str | List[Dict], optional**

Genomic regions to include/exclude. Can be:
- Path to BED file
- List of dictionaries with 'chrom', 'start', 'end' keys

### **exclude_regions : bool, optional**

If True, exclude the specified regions instead of including them.

### **filter_not_active : bool, optional**

If True, exclude sequences not active in all cell lines instead of including them.

### **filter_not_emVar : bool, optional**

If True, exclude variants not expression-modulating in all cell lines instead of including them.

### **filter_not_active_in_cellline : bool, optional**

If True, exclude sequences not active in target cell line instead of including them.

### **filter_not_emVar_in_cellline : bool, optional**

If True, exclude variants not expression-modulating in target cell line instead of including them.

### **filter_mnp : bool, optional**

If True, exclude multiple nucleotide polymorphisms instead of including them.

### **transform : callable, optional, optional**

Transformation applied to each sequence object.

### **target_transform : callable, optional, optional**

Transformation applied to the target data (expression values).

### **root : str, optional, optional**

Root directory where data is stored. If None, uses default data directory.

## SirajSatMut Parameters

### **cell_type : str**

Cell type to filter by. Must be one of: `'K562'`, `'HEPG2'`.

### **mut_num : int**

Number of baseline mutations. Must be `'1'` or `'2'`.

### **split : str, optional**

Specifies how to split the data. Currently only "test" is supported.
Default is "test".

### **length : int, optional**

Length of the sequence for the differential expression experiment.
Must be positive integer. Default is 200.

### **log2Skew_pval: float, optional**

Threshold p-value for log2Skew.

### **genomic_regions : str | List[Dict], optional**

Genomic regions to include/exclude. Can be:
- Path to BED file
- List of dictionaries with 'chrom', 'start', 'end' keys

### **exclude_regions : bool, optional**

If True, exclude the specified regions instead of including them.

### **transform : callable, optional**

Transformation applied to each sequence object.

### **target_transform : callable, optional**

Transformation applied to the target data (expression values).

### **root : str, optional**

Root directory where data is stored. If None, uses default data directory.

## Data Handling Considerations

1) The data is intended exclusively for validation of machine learning models.

2) The dataset contains information about nucleotide positions in the hg19 genome, including reference and alternative nucleotide variants. For your specific task, use the `length` parameter (default: 200) to extract nucleotide sequences with specified length and the variant nucleotide at the center.

3) When using the dataset, the hg19 genome is automatically loaded if not previously available, and nucleotide sequences of the specified length are extracted with the variant nucleotide positioned at the center.

4) Measured activity values represent the difference between alternative and reference sequence activities.

5) Use the `cell_type` parameter to select specific cell type. The data is multi-label, as sequences measured in HepG2 were also measured in other availble cell types.

6) For SirajSatMut use the `mut_num` parameter to select number of baseline mutations. Reference sequence contains this mutation as well as alternative sequence.

7) Use the `genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes in the dataset. *Uses 0-based indexing for genomic coordinates.*

8) **Example Usage**:   See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/KircherDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
import mpramnist
from mpramnist.Siraj2026.dataset import SirajMPRADataset, SirajSatMutDataset

import torch.utils.data as data
```

### 2) Dataset Creation

```python
# Load SirajMPRA data for specific cell type
dataset = SirajMPRADataset(cell_type="K562")

# Load SirajSatMut data for specific cell type and baseline mutation number
dataset = SirajSatMutDataset(cell_type="HEPG2", mut_num=2)

# Load data with custom sequence length
dataset = SirajMPRADataset(length=300, cell_type="HEPG2")

# Load data filtered by genomic regions from BED file
siraj_dataset = SirajMPRADataset(
    cell_type="A549", transform=transform, genomic_regions="path/to/regions.bed"
)

# Load data excluding specific genomic regions
regions = [{"chrom": "1", "start": 1000, "end": 2000}]
dataset = SirajMPRADataset(
    cell_type="SKNSH",
    genomic_regions=regions,
    transform=transform,
    exclude_regions=True,
)
```

### 3) Dataloader Creation

```python
siraj_forw = data.DataLoader(
    dataset=siraj_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)
```

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/mpramnist/Siraj2026/SirajDataset_example.ipynb) for detailed usage example and training

## Launch Parameters

### SirajMPRA

```bash
#MpraLegNet
python3 SirajMPRA_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_type HepG2 --result_dir ./sirajmpra_hepg2_legnet.tsv
#Malinois
python3 Kircher_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_type HepG2 --result_dir ./sirajmpra_hepg2_malinois.tsv
#MPRAnn
python3 Kircher_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_type HepG2 --result_dir ./sirajmpra_hepg2_mprann.tsv
#PARM
python3 Kircher_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_type HepG2 --result_dir ./sirajmpra_hepg2_parm.tsv
```

### SirajSatMut

```bash
#MpraLegNet
python3 SirajSatMut_model_launch.py --model MPRALegNet --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_type HepG2 --mut_num 1 --result_dir ./sirajsatmut_hepg2_legnet.tsv
#Malinois
python3 SirajSatMut_model_launch.py --model Malinois --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_type HepG2 --mut_num 1 --result_dir ./sirajsatmut_hepg2_malinois.tsv
#MPRAnn
python3 SirajSatMut_model_launch.py --model MPRAnn --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_type HepG2 --mut_num 1 --result_dir ./sirajsatmut_hepg2_mprann.tsv
#PARM
python3 SirajSatMut_model_launch.py --model PARM --lr 0.01 --wd 0.1 --epoch_num 50 --runs 5 --cell_type HepG2 --mut_num 1 --result_dir ./sirajsatmut_hepg2_parm.tsv
```

## Citation

When using this dataset, please cite the original publication:

[Siraj et al. 2026](https://www.nature.com/articles/s41586-026-10121-6)

Siraj, L., Castro, R.I., Dewey, H.B. et al. Functional dissection of complex trait variants at single-nucleotide resolution. Nature (2026). https://doi.org/10.1038/s41586-026-10121-6

```bibtex
@article{siraj2026functional,
    title={Functional dissection of complex trait variants at single-nucleotide resolution},
    author={Siraj, Layla and Castro, Rodrigo I and Dewey, Hannah B and Kales, Susan and Butts, John C and Nguyen, Thanh Thanh L and Kanai, Masahiro and Berenzy, Daniel and Mouri, Kousuke and Wang, Qingbo S and others},
    journal={Nature},
    pages={1--11},
    year={2026},
    publisher={Nature Publishing Group UK London}
}
```
