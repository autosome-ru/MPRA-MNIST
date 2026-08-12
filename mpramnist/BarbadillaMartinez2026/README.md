# Barbadilla-Martinez dataset

## Main Information TODO

See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/BarbadillaMartinez2026/BarbadillaMartinez_example.ipynb) for detailed usage example and training

## Tasks TODO

### Regression TODO

### Data Representation TODO

## Parameters TODO

### **`split : str | List[str]`**

Data split specification. Can be:
- Standard splits: `'train'`, `'val'`, `'test'`
- Chromosome names: any from `LIST_OF_CHR`
- List of chromosome names for custom splits

### **`transform : callable`, optional**

Transformation function applied to each sequence.

### **`target_transform : callable`, optional**

Transformation function applied to target values.

### **`root : str`, optional**

Root directory for data storage.

## Data Handling Considerations TODO

5) **Example Usage**: See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/BarbadillaMartinez2026/BarbadillaMartinez_example.ipynb) for detailed usage example and training

## Examples TODO

## Launch Parameters

```bash
#MPRALegNet
python3 BarbadillaMartinez_model_launch.py --model MPRALegNet --result_dir ./Barb_legnet_focused_v1.tsv --epoch_num 5 --cell_types AGS HAP1
python3 BarbadillaMartinez_model_launch.py --model MPRALegNet --result_dir ./Barb_legnet_focused_v2.tsv --epoch_num 5 --cell_types HepG2 K562 MCF7 U2OS HCT116 HEK293 LNCaP
python3 BarbadillaMartinez_model_launch.py --model MPRALegNet --result_dir ./Barb_legnet_genomewide.tsv --epoch_num 5 --cell_types HepG2_genomewide K562_genomewide
#Malinois
python3 BarbadillaMartinez_model_launch.py --model Malinois --result_dir ./Barb_malinois_focused_v1.tsv --epoch_num 5 --cell_types AGS HAP1
python3 BarbadillaMartinez_model_launch.py --model Malinois --result_dir ./Barb_malinois_focused_v2.tsv --epoch_num 5 --cell_types HepG2 K562 MCF7 U2OS HCT116 HEK293 LNCaP
python3 BarbadillaMartinez_model_launch.py --model Malinois --result_dir ./Barb_malinois_genomewide.tsv --epoch_num 5 --cell_types HepG2_genomewide K562_genomewide
#MPRAnn
python3 BarbadillaMartinez_model_launch.py --model MPRAnn --result_dir ./Barb_mprann_focused_v1.tsv --epoch_num 5 --cell_types AGS HAP1
python3 BarbadillaMartinez_model_launch.py --model MPRAnn --result_dir ./Barb_mprann_focused_v2.tsv --epoch_num 5 --cell_types HepG2 K562 MCF7 U2OS HCT116 HEK293 LNCaP
python3 BarbadillaMartinez_model_launch.py --model MPRAnn --result_dir ./Barb_mprann_genomewide.tsv --epoch_num 5 --cell_types HepG2_genomewide K562_genomewide
#PARM
python3 BarbadillaMartinez_model_launch.py --model PARM --result_dir ./Barb_parm_focused_v1.tsv --epoch_num 5 --cell_types AGS HAP1
python3 BarbadillaMartinez_model_launch.py --model PARM --result_dir ./Barb_parm_focused_v2.tsv --epoch_num 5 --cell_types HepG2 K562 MCF7 U2OS HCT116 HEK293 LNCaP
python3 BarbadillaMartinez_model_launch.py --model PARM --result_dir ./Barb_parm_genomewide.tsv --epoch_num 5 --cell_types HepG2_genomewide K562_genomewide
#DREAM-RNN
python3 BarbadillaMartinez_model_launch.py --model DREAM-RNN --result_dir ./Barb_dream-rnn_focused_v1.tsv --epoch_num 5 --cell_types AGS HAP1
python3 BarbadillaMartinez_model_launch.py --model DREAM-RNN --result_dir ./Barb_dream-rnn_focused_v2.tsv --epoch_num 5 --cell_types HepG2 K562 MCF7 U2OS HCT116 HEK293 LNCaP
python3 BarbadillaMartinez_model_launch.py --model DREAM-RNN --result_dir ./Barb_dream-rnn_genomewide.tsv --epoch_num 5 --cell_types HepG2_genomewide K562_genomewide
```

## Achieved Performance Using Basic Models

Pearson correlation, r

| Cell type | Experiment | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM_RNN |
|-----------|:---------------:|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:| :--------------------:|
| AGS | Focused v1 | 0,93 | 0,951661 | 0,926447 | 0,9288 | 0,9482 | 0,9271 |
| HAP1 | Focused v1 | 0,93 | 0,95578 | 0,940324 | 0,9343 | 0,9498 | 0,9351 |
| HepG2 | Focused v2 | 0,91 | 0,933652 | 0,909658 | 0,9019 | 0,9256 | 0,8867 |
| K562 | Focused v2 | 0,93 | 0,949594 | 0,92888 | 0,9268 | 0,9458 | 0,9143 |
| MCF7 | Focused v2 | 0,85 | 0,90721 | 0,87123 | 0,8922 | 0,9179 | 0,8839 |
| U2OS | Focused v2 | 0,86 | 0,908959 | 0,879971 | 0,8854 | 0,9112 | 0,8701 |
| HCT116 | Focused v2 | 0,89 | 0,922598 | 0,885624 | 0,9035 | 0,9258 | 0,8898 |
| HEK293 | Focused v2 | 0,91 | 0,934728 | 0,914232 | 0,9122 | 0,9331 | 0,8975 |
| LNCaP | Focused v2 | 0,84 | 0,887756 | 0,84962 | 0,8694 | 0,8999 | 0,8529 |
| HepG2 | Genome-wide | 0,89 | 0,918224 | 0,889625 | 0,8584 | 0,8991 | 0,8692 |
| K562 | Genome-wide | 0,93 | 0,943673 | 0,92749 | 0,9074 | 0,9352 | 0,9163 |


## Citation TODO

When using this dataset, please cite the original publication:
