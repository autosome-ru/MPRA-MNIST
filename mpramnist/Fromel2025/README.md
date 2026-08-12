# Fromel dataset

## Main Information TODO

(see [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/tree/main/mpramnist/Fromel2025)).


## Tasks TODO


### Data Representation TODO

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

## Parameters TODO

### **split : str, optional**

Specifies how to split the data. Currently only "test" is supported.
Default is "test".

### **transform : callable, optional**

Transformation applied to each sequence object.

### **target_transform : callable, optional**

Transformation applied to the target data (**expression values).

### **root : str, optional**

Root directory where data is stored. If None, uses default data directory.

## Data Handling Considerations TODO


8) **Example Usage**:   See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/tree/main/mpramnist/Fromel2025) for detailed usage example and training

## Examples TODO


See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/tree/main/mpramnist/Fromel2025) for detailed usage example and training

## Launch Parameters

```bash
#MPRALegNet
python3 Fromel_model_launch.py --model MPRALegNet --lr 0.005 --wd 2e-1 --result_dir ./fromel_legnet.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types HSPC
python3 Fromel_model_launch.py --model MPRALegNet --lr 0.005 --wd 2e-1 --result_dir ./fromel_legnet.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types K562
#Malinois
python3 Fromel_model_launch.py --model Malinois --lr 0.005 --wd 2e-1 --result_dir ./fromel_malinois.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types HSPC
python3 Fromel_model_launch.py --model Malinois --lr 0.005 --wd 2e-1 --result_dir ./fromel_malinois.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types K562
#MPRAnn
python3 Fromel_model_launch.py --model MPRAnn --lr 0.005 --wd 2e-1 --result_dir ./fromel_mprann.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types HSPC
python3 Fromel_model_launch.py --model MPRAnn --lr 0.005 --wd 2e-1 --result_dir ./fromel_mprann.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types K562
#PARM
python3 Fromel_model_launch.py --model PARM --lr 0.005 --wd 2e-1 --result_dir ./fromel_parm.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types HSPC
python3 Fromel_model_launch.py --model PARM --lr 0.005 --wd 2e-1 --result_dir ./fromel_parm.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types K562
#DREAM-RNN
python3 Fromel_model_launch.py --model DREAM-RNN --lr 0.0005 --wd 2e-2 --result_dir ./fromel_dream-rnn.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types HSPC
python3 Fromel_model_launch.py --model DREAM-RNN --lr 0.005 --wd 2e-1 --result_dir ./fromel_dream-rnn.tsv --batch_size 1024 --epoch_num 50 --runs 5 --cell_types K562
```

## Achieved Performance Using Basic Models

Pearson correlation, r 

| Cell type | Promoter Type | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM-RNN |
|-----------|:---------------:|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|
| HSPC | Synthetic | 0,5985210255 | 0,6217104685 | 0,6401711542 | 0,4739 | 0,5116 | -0,0058 |
| HSPC | Complex synthetic | 0,5127548973 | 0,5290360861 | 0,5479937443 | 0,5662 | 0,6268 | 0,0019 |
| HSPC | Genome | -0,02053 | 0,1031066064 | 0,005271940564 | 0,0495 | 0,0677 | 0,0169 |
| HSPC | Generated | 0,6440018327 | 0,68409 | 0,6647506665 | 0,6407 | 0,6988 | -0,018 |
| K562 | Synthetic | #N/A | 0,8542356168 | 0,8366548958 | 0,8045 | 0,8452 | 0,8037 |

## Citation TODO

When using this dataset, please cite the original publication:
