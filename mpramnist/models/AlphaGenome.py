from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.variant_scoring import *
from alphagenome_pytorch.config import DtypePolicy

import torch
import re
import numpy as np

from mpramnist.Chen2025.dataset import ChenMultiDataset
from mpramnist.Chen2025.dataset import ChenSingleDataset

from mpramnist.Guo2023.dataset import GuoMultiDataset
from mpramnist.Guo2023.dataset import GuoSingleDataset

from mpramnist.Lee2025.dataset import LeeDataset

import torch
from torch.utils.data import DataLoader


from alphagenome_research.model.metadata import metadata as metadata_lib
from alphagenome_research.model import dna_model
jax_metadata = metadata_lib.load(dna_model.Organism.HOMO_SAPIENS)

experiments_to_use = {'atac': 1, 'dnase': 1, 'procap': 1, 'cage': 1, 'rna_seq': 1, 'chip_tf': 128, 'chip_histone': 128}

jax_meta_order = {'atac': jax_metadata.atac, 'dnase': jax_metadata.dnase, 'procap' : jax_metadata.procap,
                  'cage': jax_metadata.cage, 'rna_seq': jax_metadata.rna_seq, 'chip_tf': jax_metadata.chip_tf, 
                  'chip_histone': jax_metadata.chip_histone}


def predict_variants_AlphaGenome(weights_path, dataset, batch_size, device = 'cpu', filter_tracks = None, exact_match = False):
    """
    Predict variant effects using AlphaGenome model (pytorch realisation).

    For each batch, computes reference and alternative predictions across multiple
    experiments (ATAC, DNase, etc.), and aggregates them per channel.
    Supports all MPRA-MNIST SNP Datasets.

    Parameters
    ----------
    weights_path : str
        Path to the AlphaGenome model weights (.safetensors).
    dataset : MpraDataset
        MpraDataset with SNP data. Currently supports:
        GuoMultiDataset, GuoSingleDataset, LeeDataset, 
        ChenMultiDataset, ChenSingleDataset. Initialize 
        MpraDataset using sequence length (length parameter)
        supported by AlphaGenome model (from 2048 to 2**20).
    batch_size : int
        Batch size for the internal DataLoader.
    device : str, default='cpu'
        Device for inference ('cuda:0', 'cpu', etc.).
    filter_tracks : list of str, optional
        If given, only prediction tracks whose biosample name matches any of
        these strings (substring or exact) are kept. Available tracks
        can be found in jax_metadata biosample_name columns.
        If None, all non-padding tracks are used. 
    exact_match : bool, default=False
        If True, `filter_tracks` requires exact (case sensitive) matches.
        If False, uses case insensitive substring matching.

    Returns
    -------
    list of dicts matching corresponding LitModel output for the provided dataset.
    Prediction is made with MEAN among selected tracks
    """
    
    # init the model
    model = AlphaGenome.from_pretrained(weights_path, device=device, dtype_policy=DtypePolicy.mixed_precision())
    model.eval()

    # wrap with new dataloader, small batch_size
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    N = len(dataset)
    all_results = []

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for i, batch in enumerate(dataloader):
    
            # Use lists of torch tensors (no numpy)
            batch_output_ref = []
            batch_output_alt = []

            # get batch data
            if isinstance(dataset, GuoSingleDataset) or isinstance(dataset, GuoMultiDataset):
                seqs, labels, fdrs = batch
                var_type = None
                rev_pred = None
                
            elif isinstance(dataset, LeeDataset):
                seqs, labels, var_type, rev_pred = batch
                fdrs = None
            elif isinstance(dataset, ChenSingleDataset) or isinstance(dataset, ChenMultiDataset):
                seqs, labels, fdrs, rev_pred = batch
                var_type = None

            pred_ref, pref_alt = [model.predict(seqs.get('seq').transpose(1, 2).to(device), 0), model.predict(seqs.get('seq_alt').transpose(1, 2).to(device), 0)]

            for i, (expname, res) in enumerate(experiments_to_use.items()):
                exp_meta = jax_meta_order[expname]
                mask =  exp_meta['name'] != 'Padding'
                batch_output_ref.append(torch.log1p(pred_ref[expname][res][:, :, mask].to(torch.float64).sum(axis=1).cpu()))
                batch_output_alt.append(torch.log1p(pref_alt[expname][res][:, :, mask].to(torch.float64).sum(axis=1).cpu()))

            result = {
                "ref_predicted": torch.cat(batch_output_ref, dim=1),
                "alt_predicted": torch.cat(batch_output_alt, dim=1),
                "target": labels
                }

            if fdrs is not None:
                result["fdr"] = fdrs.cpu()
            if var_type is not None:
                result["var_type"] = var_type.cpu()
            if rev_pred is not None:
                result["reverse_prediction"] = rev_pred.cpu()


            all_results.append(result)
            if i % 100 == 0:
                print(f'Processed samples: {i * batch_size} / {N}')
        
        return all_results



def filter_tracks(data, biosample_names, exact_match = True):
    """
    data: torch.Tensor of shape (L, total_channels)
          where channels are ordered as in experiments_to_use.
    biosample_names: list of strings to filter.
    exact_match: bool.

    Returns filtered data with only channels that are non-padding and match biosample.
    """
    mask = []
    for expname in experiments_to_use.keys():
        exp_meta = jax_meta_order[expname]

        exp_meta =  exp_meta[exp_meta['name'] != 'Padding']

        if not exact_match:
            pattern = '|'.join(re.escape(w) for w in biosample_names)
            exp_mask = exp_meta['biosample_name'].str.contains(pattern, case=False, na=False)
        else:
            exp_mask = exp_meta['biosample_name'].isin(biosample_names)
        
        mask.append(exp_mask)

    mask = np.concatenate(mask)
    
    return data[:, torch.tensor(mask, dtype=torch.bool, device=data.device)]
