# Vaishnav dataset

## Main Information

The Vaishnav dataset is based on the research by [Vaishnav et al., 2022](https://www.nature.com/articles/s41586-022-04506-6). It contains measurements of **episomal** promoter-driven gene expression (acting as a proxy for regulatory activity) in two environments: a standard rich medium (YPD, containing yeast extract, peptone, and glucose) and a minimal defined medium (SD-Ura, a synthetic medium lacking uracil).

The core dataset comprises approximately **40 million unique DNA promoter sequences**, each measured in one or both environments, resulting in **30,722,376 sequence-environment pairs for YPD** and **20,616,659 pairs for SD-Ura**.

**Input:** DNA sequences of variable length (originally 80bp, presented as 78 to 139 nucleotides with flanks).
**Output:** A single scalar value representing the **expression level / regulatory activity**. This value is calculated as the **weighted average of expression bins (0-17)**, where cells were sorted based on the fluorescence intensity of a YFP reporter gene (GPRA system). The weight for each bin corresponds to the number of times the sequence was observed in that bin.

To standardize sequence length for neural network processing, constant flanking sequences from the original plasmid context are added to both ends.

The training data (random sequences) is split into training and validation sets (9:1). The independent test data includes:
1.  **Native sequences:** Endogenous yeast promoters (3,928 in YPD, 3,977 in SD-Ura), measured with high precision (~100 cells per sequence).
2.  **Drift sequences:** Algorithmically generated sequences from the evolution experiment.
3.  **Paired sequences:** Reference and mutated sequence pairs to predict the effect of single mutations.

See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Vaishnav2024/VaishnavDataset_example.ipynb) for detailed usage example and training

## Tasks

### Regression

The regression task involves predicting the regulatory activity value from the DNA sequence.

### Data Representation

```
===== drift and native datasets
            seq	                                                                        label
CTTTCAATTGGGTGGGGACGCGACGGCGCCCCGACTAGGATGCTAGCGTACTATGCTGCCTGAAAGTCTATAGGAGCATT	12.616153717041016
CTTTAAATTCGGTGGGGACGCGTCGGCGCCCCGGCTAGGATGCTAGCGTACTATGCTGCCTGAAAGTCTATAGGAGCATT	12.998969078063965
CTTTCAATTGGGTGGGGACGCGACCGCGCCCCGGCTAGGATGCTAGCCTACTATGCTGCCTGAAAGTCTATAGGAGCATT	13.037347316741943
```

```
===== paired datasets
    seq	                        seq_alt	        n_mut	ref_measured	    alt_measured	    delta_measured
CTTTCAATTGGGTGGGGA...	CTTTCAATTGGGTGGGGA...	1	    11.559163950076	    11.278011168248	    -0.281152781828
CTTTCAATTGGGTGGGGA...	CTTTAAATTCGGTGGGGA...	3	    11.559163950076	    11.7254605738169	0.1662966237409
CTTTCAATTGGGTGGGGA...	CTTTCAATTGGGTGGGGA...	2	    11.559163950076	    11.7250527297309	0.1658887796549
```

## Parameters

### **`split : Literal["train", "val", "test"]`**

Data split specification:
- `"train"`: Training data
- `"val"`: Validation data
- `"test"`: Test data

### **`dataset_env_type : Literal["defined", "complex"]`**

Environmental context type:
- `"defined"`: Synthetic Defined medium lacking uracil (SD-Ura). This dataset contains ~20.6 million sequences.
- `"complex"`: Yeast Extract, Peptone, and Dextrose medium (YPD). This dataset contains ~30.7 million sequences.

### **`test_dataset_type : Literal["drift", "native", "paired"]`**, optional

Required for test split only. Specifies test scenario:
- `"drift"`: The sequences designed by the genetic algorithm 
- `"native"`: Native yeast promoter test sequences
- `"paired"`: Paired sequences for predicting the effect of mutations. The task is to predict the difference (`delta_measured`) between the reference (`seq`) and alternative (`seq_alt`) sequences.

### **`transform : callable`**, optional

Transformation function applied to each sequence. Useful for data augmentation
or sequence encoding.

### **`target_transform : callable`**, optional

Transformation function applied to target values. Useful for normalization
or target processing.

### **`root : str`**, optional

Root directory for data storage. If `None`, uses default data directory.

## Data Handling Considerations

1) The `dataset_env_type` parameter selects data from a specific growth medium. The `"defined"` (SD-Ura) environment contains approximately 20.6 million sequences, while the `"complex"` (YPD) environment contains approximately 30.7 million.

2) The `test_dataset_type` parameter defines which sequence set is used for evaluation in the test (or validation) split. `"drift"` sequences designed by the genetic algorithm, `"native"` sequences originate from the yeast genome, and `"paired"` requires predicting the difference between reference and mutated sequences.

3) When loading paired data, the dataset returns tuples `(seq, seq_alt, delta_measured)`.

4) Using the `transform` argument is recommended to apply necessary preprocessing, such as adding constant flanking sequences and converting sequences to tensors.

5) **Example Usage**: See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Vaishnav2024/VaishnavDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    from mpramnist.Vaishnav2024 import VaishnavDataset
    from mpramnist.Vaishnav2024 import LitModel_Vaishnav
    
    import mpramnist.transforms as t
    
    from torch.utils.data import DataLoader

    length = 110
    plasmid = VaishnavDataset.PLASMID.upper()
    insert_start = plasmid.find("N" * 80)
    right_flank = VaishnavDataset.RIGHT_FLANK
    left_flank = plasmid[insert_start - length : insert_start]
```

### 2) Initialize transforms

```python
    train_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.LeftCrop(length, length),
            t.ReverseComplement(0.5), # Apply reverse complement with 50% probability for augmentation
            t.Seq2Tensor(),           # Convert sequence to tensor (e.g., one-hot)
        ]
    )
    val_test_transform = t.Compose(
        [
            t.AddFlanks(left_flank, right_flank),
            t.LeftCrop(length, length),
            t.ReverseComplement(0),   # No augmentation for validation/test
            t.Seq2Tensor(),
        ]
    )
   
```

### 3) Dataset Creation

```python
    # Load training data from defined environment
    train_dataset = VaishnavDataset(
        split="train",
        dataset_env_type="defined",
        transform=train_transform
    )
    
    # Load validation data from complex environment
    val_dataset = VaishnavDataset(
        split="val", 
        dataset_env_type="complex",
        transform=val_test_transform
    )
    
    # Load test data for distribution drift scenario
    test_drift = VaishnavDataset(
        split="test",
        dataset_env_type="defined",
        test_dataset_type="drift",
        transform=val_test_transform
    )
    
    # Load test data for paired sequence analysis
    test_paired = VaishnavDataset(
        split="test",
        dataset_env_type="complex", 
        test_dataset_type="paired",
        transform=val_test_transform
    )
```

### 4) Dataloader Creation

```python 
    train_loader = data.DataLoader(
        dataset=train_dataset, 
        batch_size=1024, 
        shuffle=True, # True for training, False for val and test
        num_workers=8
    )

    val_loader = data.DataLoader(
        dataset=val_dataset, 
        batch_size=1024, 
        shuffle=False, # Do not shuffle for validation and testing
        num_workers=8
    )
```

## Launch Parameters

```bash
#MPRALegNet
python3 Vaishnav_model_launch.py --model MPRALegNet --lr 1e-2 --wd 1e-2 --result_dir ./Vaishnav_legnet.tsv --batch_size 1024 --epoch_num 1
#Malinois
python3 Vaishnav_model_launch.py --model Malinois --lr 1e-2 --wd 1e-2 --result_dir ./Vaishnav_malinois.tsv --batch_size 1024 --epoch_num 1
#MPRAnn
python3 Vaishnav_model_launch.py --model MPRAnn --lr 1e-2 --wd 1e-2 --result_dir ./Vaishnav_mprann.tsv --batch_size 1024 --epoch_num 1
#PARM
python3 Vaishnav_model_launch.py --model PARM --lr 1e-2 --wd 1e-2 --result_dir ./Vaishnav_parm.tsv --batch_size 1024 --epoch_num 1
#DREAM-RNN
python3 Vaishnav_model_launch.py --model DREAM-RNN --lr 1e-2 --wd 1e-2 --result_dir ./Vaishnav_dream-rnn.tsv --batch_size 1024 --epoch_num 1
```


## Original Benchmark Quality

Pearson correlation, r

| Cell type | Experiment | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM_RNN |
|-----------|:---------------:|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:| :--------------------:|
| strains Y8205, S288C::ura3, etc | SD-Ura (defined) Native | 0,931 | 0,979 | 0,9697 | 0,9711 | 0,964 | 0,9643 |
| strains Y8205, S288C::ura3, etc | SD-Ura (defined) Drift | 0,968 | 0,986 | 0,984 | 0,9736 | 0,9734 | 0,9779 |
| strains Y8205, S288C::ura3, etc | SD-Ura (defined) Paired | 0,847 | 0,853 | 0,8507 | 0,831 | 0,8338 | 0,8421 |
| strains Y8205, S288C::ura3, etc | YPD (complex) Native | 0,948 | 0,975 | 0,9672 | 0,9686 | 0,9587 | 0,9628 |
| strains Y8205, S288C::ura3, etc | YPD (complex) Drift | 0,98 | 0,985 | 0,9856 | 0,9752 | 0,9745 | 0,9801 |
| strains Y8205, S288C::ura3, etc | YPD (complex) Paired | 0,869 | 0,879 | 0,8793 | 0,8566 | 0,86 | 0,8674 |
 

## Citation

When using this dataset, please cite the original publication:

[Vaishnav et al. 2022](https://www.nature.com/articles/s41586-022-04506-6)

Vaishnav, E.D., de Boer, C.G., Molinet, J. et al. The evolution, evolvability and engineering of gene regulatory DNA. Nature 603, 455–463 (2022). https://doi.org/10.1038/s41586-022-04506-6

```bibtex
    @article{vaishnav2022evolution,
        title = {The evolution, evolvability and engineering of gene regulatory DNA},
        volume = {603},
        ISSN = {1476-4687},
        url = {http://dx.doi.org/10.1038/s41586-022-04506-6},
        DOI = {10.1038/s41586-022-04506-6},
        number = {7901},
        journal = {Nature},
        publisher = {Springer Science and Business Media LLC},
        author = {Vaishnav, Eeshit Dhaval and de Boer, Carl G. and Molinet, Jennifer and Yassour, Moran and Fan, Lin and Adiconis, Xian and Thompson, Dawn A. and Levin, Joshua Z. and Cubillos, Francisco A. and Regev, Aviv},
        year = {2022},
        month = mar,
        pages = {455--463}
    }
```