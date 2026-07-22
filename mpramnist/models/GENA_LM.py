import torch 
import torch.nn as nn 

from typing import List, Dict
import torch.nn.functional as F

from transformers import AutoTokenizer, BertModel
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

def initialize_head(module: nn.Module):
    """
    Initialize only newly created layers.
    Pretrained BERT weights are not touched.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MLPHead(nn.Module):
    """
    Classification / regression head.

    hidden_dim=None:
        LayerNorm -> Linear

    hidden_dim=int:
        LayerNorm -> Linear -> GELU -> Dropout -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        self.classifier.apply(initialize_head)

    def forward(self, x):
        return self.classifier(x)


class GENA_LM(nn.Module):
    """
    GENA-LM wrapper compatible with Lightning trainers.
    Supports
        - regression
        - classification
    Input:
        list[str]
    or
        tuple/list of DNA sequences
    or
        already tokenized dict
    Output:
        regression:
            (B,)
        classification:
            (B, num_labels)
    """
    
    def __init__(
        self,
        model_name="AIRI-Institute/gena-lm-bert-base-t2t",
        task="regression",
        num_labels=1,
        seqsize=300,
        pooling="cls",
        hidden_dim=None,
        dropout=0.1,
    ):
        super().__init__()

        if task not in ["regression", "classification"]:
            raise ValueError(
                "task must be 'regression' or 'classification'"
            )

        if pooling not in ["cls", "mean"]:
            raise ValueError(
                "pooling must be 'cls' or 'mean'"
            )

        self.task = task
        self.pooling = pooling
        self.seqsize = seqsize
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        output_dim = 1 if task == "regression" else num_labels

        self.classifier = MLPHead(
            input_dim=hidden_size,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _tokenize(self, sequences):
        """
        Convert a batch of DNA strings into HuggingFace tensors.
        """

        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            seqsize=self.seqsize,
            return_tensors="pt",
        )

        return {k: v.to(self.device)
            for k, v in tokens.items()}

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):

        if self.pooling == "cls":
            return hidden_states[:, 0]

        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, x):
        if isinstance(x, dict):
            batch = {k: v.to(self.device)
                     for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            batch = self._tokenize(list(x))
        
        outputs = self.encoder(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"])

        embedding = self._pool(outputs.last_hidden_state,
                               batch["attention_mask"])

        logits = self.classifier(embedding)

        if self.task == "regression":
            return logits.squeeze(-1)

        return logits

    @property
    def hidden_size(self):
        return self.encoder.config.hidden_size

    @property
    def vocab_size(self):
        return self.encoder.config.vocab_size

    def save_pretrained(self, path: str):
        """
        Save encoder, tokenizer and prediction head.
        """

        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        torch.save(
            {"classifier": self.classifier.state_dict(),
             "task": self.task,
             "pooling": self.pooling,
             "seqsize": self.seqsize},
            f"{path}/classifier.pt")

    @classmethod
    def from_pretrained(
        cls,
        path,
        task="regression",
        num_labels=1,
        **kwargs):

        model = cls(model_name=path,
                    task=task,
                    num_labels=num_labels,
                    **kwargs)

        checkpoint = torch.load(f"{path}/classifier.pt",
                                map_location="cpu")

        model.classifier.load_state_dict(checkpoint["classifier"])

        return model

    def extra_repr(self):
        return (f"task={self.task}, "
                f"pooling={self.pooling}, "
                f"seqsize={self.seqsize}")
