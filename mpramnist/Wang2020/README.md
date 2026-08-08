# DeepPromoter dataset

## Main Information

The DeepPromoter dataset is based on the **episomal** experimental data from [Wang et al. 2020](https://academic.oup.com/nar/article/48/12/6403/5837049?login=false)., which includes 14,098 promoter sequences from E. coli that were experimentally validated for activity. This collection comprises:

- **Natural promoters** from the E. coli K12 MG1655 genome

- **AI-generated synthetic promoters** created using a Generative Adversarial Network (GAN) trained on natural promoter features

Each sequence is **50 nucleotides** long and represents the region upstream of the transcription start site (TSS), containing key promoter elements including the -10 and -35 boxes.

After removing duplicates, the dataset was split into: **9,000 sequences for training, 1,000 for validation, and 1,884 for testing**.

See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Wang2020/WangDataset_example.ipynb) for detailed usage example and training

## Tasks

### Regression

The regression task is to predict a single scalar value representing normalized promoter activity. The activity values in the dataset are calculated using the following experimental measurements from the original study:

1) **Experimental measurement**: Each promoter variant ("clone") was inserted upstream of the sfGFP reporter gene, and its expression was measured as fluorescence per cell (`F/OD600`).

2) **Normalization**: Promoter strength (S) was calculated relative to positive and negative controls using the formula:

S = ( (F/OD600)_clone - (F/OD600)_blank ) / ( (F/OD600)_BBA_J23_119 - (F/OD600)_blank )

Or in methematical notation:

$$S = \frac{(F/OD600)_{\text{clone}} - (F/OD600)_{\text{blank}}}{(F/OD600)_{\text{BBA\_J23\_119}} - (F/OD600)_{\text{blank}}}$$

Where:

 - **(F/OD600)**= Fluorescence normalized by cell density

 - **blank** = Control with a 10-nt random sequence (GGGCTCTGTA)

 - **BBA_J23_119** = Reference wild-type promoter (positive control)

3) **Interpretation**: 
   - S = 1: Activity equal to reference promoter
   - S < 1: Weaker than reference
   - S > 1: Stronger than reference
   - Many AI-generated promoters showed S values >> 1, indicating significantly higher activity.


4) **Final values**: Reported activities are averages of three independent biological experiments.

The target values in this dataset represent these normalized promoter activity scores, which quantify transcriptional strength relative to experimental controls.

### Data Representation

```
                sequence	                        target	split
taatttttatctgtctgtgcgctatgcctatattggttaaagtatttagt	36.84	train
cgcgggaatcgcgcaggcagcactaccccagagcgtggtagcctgggagt	94.32	val
gaatcactttttcgttgccgccttctttgaatttatcaacgatattaccc	70.45	test

```

## Parameters

### **`split : str`**
Defines which data split to use. Must be one of: `'train'`, `'val'`, `'test'`.
The dataset filters sequences based on the 'split' column in the data file.
### **`transform : callable`**, optional
Transformation applied to each sequence object.
### **`target_transform : callable`**, optional
Transformation applied to the target data.
### **`root : str`**, optional, `default=None`
Root directory where dataset files are stored or should be downloaded.
If `None`, uses the default dataset directory from parent class.

## Data Handling Considerations

1) **Example Usage**: See [Usage Example](https://github.com/autosome-ru/MPRA-MNIST/blob/main/mpramnist/Wang2020/WangDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    from mpramnist.Wang2020.dataset import WangDataset
     mpramnist.Wang2020.trainer import LitModel_Wang

    import mpramnist.transforms as t

    from torch.utils.data import DataLoader
```

### 2) Initialize trannsforms

```python
    # Define set of transforms
    train_transform = t.Compose(
        [
            t.ReverseComplement(0.5),  # Apply reverse complement with 50% probability
            t.Seq2Tensor(),
        ]
    )
    val_test_transform = t.Compose(
        [
            t.Seq2Tensor(),  # No reverse complement for validation/test
        ]
    )
```

### 3) Dataset Creation

```python
    # Basic usage for training
    dataset = WangDataset(
        split='train', 
        transform = train_transform,
        root="../data/"
    )

    val_dataset = WangDataset(
        split="val", 
        transform=val_test_transform, 
        root="../data/"
    )
    
```

### 4) Dataloader Creation

```python 
    train_loader = data.DataLoader(
        dataset=dataset, 
        batch_size=1024, 
        shuffle=True,   # Shuffle for the training set
        num_workers=8
    )

    val_loader = data.DataLoader(
        dataset=val_dataset, 
        batch_size=1024, 
        shuffle=False,   # No need to shuffle for validation/testing
        num_workers=8
    )
```

## Launch Parameters

```bash
#MPRALegNet
python3 Wang_model_launch.py --model MPRALegNet --lr 1e-3 --wd 1e-4 --result_dir ./wang_legnet.tsv --batch_size 1024 --epoch_num 50 --runs 5
#Malinois
python3 Wang_model_launch.py --model Malinois --lr 1e-3 --wd 1e-4 --result_dir ./wang_malinois.tsv --batch_size 1024 --epoch_num 50 --runs 5
#MPRAnn
python3 Wang_model_launch.py --model MPRAnn --lr 1e-3 --wd 1e-4 --result_dir ./wang_mprann.tsv --batch_size 1024 --epoch_num 50 --runs 5
#PARM
python3 Wang_model_launch.py --model PARM --lr 1e-3 --wd 1e-4 --result_dir ./wang_parm.tsv --batch_size 1024 --epoch_num 50 --runs 5
#DREAM-RNN
python3 Wang_model_launch.py --model DREAM-RNN --lr 1e-3 --wd 1e-4 --result_dir ./wang_dream-rnn.tsv --batch_size 1024 --epoch_num 50 --runs 5
```

## Original Benchmark Quality

**Pearson correlation, r**

| Cell type | Original performance | MPRALegnet | Mprann | Malinois | PARM | DREAM_RNN |
|-----------|:---------------:|:----------------:|:-------------------:|:--------------------:|:--------------------:| :--------------------:|
| The DH5α E. coli strain | 0,25 | 0,269 | 0,0549 | 0,1044 | 0,2584 | 0,0466 |


## Citation

When using this dataset, please cite the original publication:

[Ye Wang et al. 2020](https://academic.oup.com/nar/article/48/12/6403/5837049?login=false)

Ye Wang, Haochen Wang, Lei Wei, Shuailin Li, Liyang Liu, Xiaowo Wang, Synthetic promoter design in Escherichia coli based on a deep generative network, Nucleic Acids Research, Volume 48, Issue 12, 09 July 2020, Pages 6403–6412, https://doi.org/10.1093/nar/gkaa325

```bibtex
    @article{Wang2020synthetic,
        title = {Synthetic promoter design in Escherichia coli based on a deep generative network},
        volume = {48},
        issn = {1362-4962},
        url = {https://doi.org/10.1093/nar/gkaa325},
        doi = {10.1093/nar/gkaa325},
        number = {12},
        journal = {Nucleic Acids Research},
        publisher = {Oxford University Press},
        author = {Wang, Ye and Wang, Haochen and Wei, Lei and Li, Shuailin and Liu, Liyang and Wang, Xiaowo},
        year = {2020},
        month = {05},
        pages = {6403–6412}
    }
```