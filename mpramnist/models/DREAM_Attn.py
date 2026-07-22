import torch 
import torch.nn as nn 

from typing import List, Any, Dict
import torch.nn.functional as F

from torch import Generator

from abc import ABCMeta, abstractmethod

from collections import OrderedDict

import math

def initialize_weights(m: nn.Module, generator: Generator):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n), generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001, generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class FirstLayersBlock(nn.Module, metaclass=ABCMeta):
    """
    Network first layers performing low-resolution feature generation
    """
    @abstractmethod
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 seqsize: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seqsize = seqsize
    
    @abstractmethod
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Usual forward pass of torch nn.Module
        """
        ...
        
    def train_step(self,
                   batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Modification of the forward pass. Required to train properly different combinations of blocks
        Receives batch with required "x" and "y" keys and optional keys, required for blocks from some teams
        Returns tuple containing:
            1. modified "x"
            2. auxiliary loss if it is computed by the block or `None` otherwise 
            
        Default realization simply call forward and return None as an auxiliary loss
        """
        return self.forward(batch["x"].to(self.device)), None
        
    @property
    def dummy(self) -> torch.Tensor:
        """
        return dummy input data to test model correctness and infer output seqsize
        """
        return torch.zeros(size=(1, self.in_channels, self.seqsize), dtype=torch.float32)
    
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration: # model has no parameters
            return torch.device("cpu") # it safe to return cpu in such case
    
    def infer_outseqsize(self) -> int:
        """
        return output seqsize by running model
        """
        x = self.forward(self.dummy.to(self.device))
        return x.shape[-1]
    
    def check(self) -> None:
        """
        Run model on dummy object
        """
        self.forward(self.dummy.to(self.device))
        
    def weights_init(self, generator: Generator) -> None:
        """
        Weight initializations for block. Should use provided generator to generate new weights
        By default do nothing
        """
        pass

class CoreBlock(nn.Module, metaclass=ABCMeta):
    """
    Network core layers performing complex feature extraction
    """
    @abstractmethod
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 seqsize: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seqsize = seqsize

    @abstractmethod
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Usual forward pass of torch nn.Module
        """
        ...
        
    def train_step(self, 
                   batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Modification of the forward pass. Required to train properly different combinations of blocks
        Receives batch with required "x" and "y" keys and optional keys, required for blocks from some teams
        Returns tuple containing:
            1. modified "x"
            2. auxiliary loss if it is computed by the block or `None` otherwise  

        Default realization simply call forward and return None as an auxiliary loss
        """
        return self.forward(batch["x"].to(self.device)), None
    
    def weights_init(self, generator: Generator) -> None:
        """
        Weight initializations for block. Should use provided generator to generate new weights
        By default do nothing
        """
        pass
        
        
    @property
    def dummy(self) -> torch.Tensor:
        """
        return dummy input data to test model correctness and infer output seqsize
        """
        return torch.zeros(size=(1, self.in_channels, self.seqsize), dtype=torch.float32)
    
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration: # model has no parameters
            return torch.device("cpu") # it safe to return cpu in such case
    
    def infer_outseqsize(self) -> int:
        """
        return output seqsize by running model
        """
        x = self.forward(self.dummy.to(self.device))
        return x.shape[-1]
    
    def check(self) -> None:
        """
        Run model on dummy object
        """
        self.forward(self.dummy.to(self.device))
    
class FinalLayersBlock(nn.Module, metaclass=ABCMeta):
    """
    Network final layers performing final prediction and (optionally) loss calculation
    """
    
    @abstractmethod
    def __init__(self,
                 in_channels: int,
                 seqsize: int):
        super().__init__()
        self.in_channels = in_channels
        self.seqsize = seqsize
    
    @abstractmethod
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Usual forward pass of torch nn.Module
        """
        ...
    
    @abstractmethod
    def train_step(self,
                   batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Final block must return predicted y value and loss to be used by Trainer instance
        This block MUST work in case batch contains only 'x' and 'y' keys. 
        It, however, can change behaviour (e.g loss calculated) if batch contains additional keys,
        beneficial for team implementation (see autosome final block)
        """
        ...
        
    @property
    def dummy(self) -> torch.Tensor:
        """
        return dummy input data to test model correctness
        """
        return torch.zeros(size=(1, self.in_channels, self.seqsize), dtype=torch.float32)
    
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration: # model has no parameters
            return torch.device("cpu") # it safe to return cpu in such case
        
    def check(self) -> None:
        """
        Run model on dummy object
        """
        self.forward(self.dummy.to(self.device))
        
    def weights_init(self, generator: Generator) -> None:
        """
        Weight initializations for block. Should use provided generator to generate new weights
        By default do nothing
        """
        pass

class SwiGLULayer(nn.Module):
    def __init__(self, dim):
        super(SwiGLULayer, self).__init__()
        self.dim = dim
        self.swish = nn.SiLU() # same as swish

    def forward(self, x):
        out, gate = torch.chunk(x, 2, dim = self.dim)
        return out * self.swish(gate)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, embedding_dim, mult=4, rate = 0.0, use_bias = True):
        super(FeedForwardSwiGLU, self).__init__()
        swiglu_out = int(embedding_dim * mult/2)
        self.layernorm = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.linear1 = nn.Linear(embedding_dim,embedding_dim * mult, bias = use_bias)
        self.swiglulayer = SwiGLULayer(dim = 1)
        self.drop = nn.Dropout(rate)
        self.linear2 = nn.Linear(swiglu_out,embedding_dim, bias = use_bias)

    def forward(self, inputs):
        x = self.layernorm(inputs.transpose(1,2)) # Swap dimensions and make channel dim=2
        x = self.linear1(x) 
        x = self.swiglulayer(x.transpose(1,2)) # Swap dimensions again and make channel dim =1
        x = self.drop(x)
        x = self.linear2(x.transpose(1,2)) # Swap dimensions and make channel dim=2
        out = self.drop(x.transpose(1,2)) # Swap dimensions again and make channel dim =1
        return out


class ConformerSASwiGLULayer(nn.Module):
    def __init__(self, embedding_dim,  ff_mult = 4, kernel_size = 15, rate = 0.2, num_heads = 4, use_bias = False):
        super(ConformerSASwiGLULayer, self).__init__()
        self.ff1 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)
        self.layernorm1 = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size, groups=embedding_dim, padding='same', bias = False),
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, bias = True),
            nn.ReLU(),
            nn.Dropout(rate),
        )
        self.layernorm2 = nn.LayerNorm(embedding_dim,eps = 1e-6)    
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first = True)
        self.ff2 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)

    def forward(self, x):
        x = x.float()
        x = x + 0.5 * self.ff1(x)
        
        x1 = x.transpose(1,2)
        x1 = self.layernorm1(x1) #channel dim = 2
        x1 = x1.transpose(1, 2)
        x1 = x1 + self.conv(x1)
        
        x = x + x1
        x = x.transpose(1, 2) # output channel dim = 2
        x = self.layernorm2(x)
        x = x + self.attn(x, x, x)[0]
        x = x.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        
        return x

class AutosomeFirstLayersBlock(FirstLayersBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seqsize: int  # for compatibity. Isn't used by block itself
    ):
        super().__init__(in_channels=in_channels, 
                         out_channels=out_channels, 
                         seqsize=seqsize)
        ks = 7
        activation = nn.SiLU
        self.bn_momentum = .1
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ks,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(out_channels,
                            momentum=self.bn_momentum),
            activation()
        )

    def forward(self, x) -> torch.Tensor:
        if len(x.shape) < 3:
            x = F.one_hot(x.to(torch.int64), self.in_channels)
            x = x.float().permute(0,2,1)
        x = self.block(x)
        return x
    
    def weights_init(self, generator: Generator) -> None:
        self.apply(lambda x: initialize_weights(x, generator))

class UnlockDNACoreBlock(CoreBlock):
    def __init__(
        self,
        in_channels: int=512,
        out_channels: int=64,
        seqsize: int = 200,
        num_heads: int=4,
        kernel_size = 15,
        rate = 0.1,
        n_blocks = 4

    ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         seqsize=seqsize)     


        self.blocks = nn.ModuleList([ConformerSASwiGLULayer(embedding_dim = in_channels,
                                    kernel_size = kernel_size, rate = rate, num_heads = num_heads) for _ in range(n_blocks)])
        self.n_blocks = n_blocks
        self.out_channels = out_channels
        self.pos_embedding = nn.Embedding(self.seqsize, out_channels)
        
    def forward(self, x):
        x = x.transpose(1,2)
        
        pos = torch.arange(start=0, end = self.seqsize, step=1).to(self.device)
        pos = pos.unsqueeze(0)
        pos = self.pos_embedding(pos.long())
        x = x + pos
        x = x.transpose(1,2)

        for i in range(self.n_blocks) :
            x = self.blocks[i](x)
        return x
    
#    def weights_init(self, generator: Generator) -> None:
#        self.apply(lambda x: initialize_weights(x, generator))

class AutosomeFinalLayersBlock(FinalLayersBlock):
    def __init__(self, in_channels=64, seqsize = 230, out_channels=1):
        super(AutosomeFinalLayersBlock, self).__init__(
            in_channels=in_channels,
            seqsize=seqsize)
        self.mapper = nn.Conv1d(
            in_channels=in_channels,  # Assuming the input channels to be the same as output
            out_channels=256,
            kernel_size=1,
            padding='same'
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(256, out_channels)
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, 1)
        )
        # self.activation = nn.SiLU()
        # self.predictions = nn.Linear(256, 1)
        self.regression_criterion = nn.MSELoss()
        self.classification_criterion = nn.BCELoss()

    def forward(self, x):
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2) 
        x = self.linear(x).squeeze()
        # x = self.activation(x)
        # x = self.predictions(x)
        return x

    def train_step(self, batch: dict[str, Any]):
        x = batch["x"].to(self.device)
       
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2) 
        x = self.linear(x)

        x = x.squeeze(-1)
        y = batch["y"].to(self.device).squeeze(-1)

        loss = self.regression_criterion(x, y)
        return x, loss
    
    def weights_init(self, generator: Generator) -> None:
        self.apply(lambda x: initialize_weights(x, generator))


class DREAM_Attn(nn.Module):
    """
    DREAM_Attn model class with attention mechanism.
    Uses AutosomeFirstLayersBlock, UnlockDNACoreBlock (with ConformerSA), 
    and AutosomeFinalLayersBlock.
    """
    def __init__(
        self,
        in_channels: int = 5,
        seqsize: int = 200,
        out_channels: int = 1,
        first_out_channels: int = 256,
        core_out_channels: int = 256,
        n_blocks: int = 4,
        kernel_size: int = 15,
        rate: float = 0.1,
        num_heads: int = 8,
        generator: Generator | None = None
    ):
        super().__init__()
        
        if generator is None:
            generator = torch.Generator()
        
        # Build first block
        self.first = AutosomeFirstLayersBlock(
            in_channels=in_channels,
            out_channels=first_out_channels,
            seqsize=seqsize
        )
        
        # Build core block
        self.core = UnlockDNACoreBlock(
            in_channels=self.first.out_channels,
            out_channels=core_out_channels,
            seqsize=seqsize,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            rate=rate,
            num_heads=num_heads
        )
        
        # Build final block
        self.final = AutosomeFinalLayersBlock(
            in_channels=self.core.out_channels,
            seqsize=self.core.infer_outseqsize(),
            out_channels=out_channels
        )
        
        self.generator = generator
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire model.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seqsize)
            
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        x = self.first(x)
        x = self.core(x)
        x = self.final(x)
        return x
    
    def train_step(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training step of the model.
        
        Args:
            batch: Dictionary containing 'x' and 'y' keys
            
        Returns:
            Tuple of (predictions, loss)
        """
        loss = 0
        
        # First block
        x, ax_loss = self.first.train_step(batch)
        batch["x"] = x
        if ax_loss is not None:
            loss += ax_loss
        
        # Core block
        x, ax_loss = self.core.train_step(batch)
        batch["x"] = x
        if ax_loss is not None:
            loss += ax_loss
        
        # Final block
        y, pred_loss = self.final.train_step(batch)
        loss += pred_loss
        
        return y, loss
    
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    
    @property
    def dummy(self) -> torch.Tensor:
        """Return dummy input data to test model correctness."""
        return self.first.dummy
    
    @property
    def dummy_expression(self) -> torch.Tensor:
        return torch.FloatTensor([[10]])
    
    def check(self) -> None:
        """Run model on dummy object to verify correctness."""
        print("Checking forward pass")
        self.forward(self.dummy.to(self.device))
        print("Forward is OK")
        print("Checking training step")
        batch = {"x": self.dummy, "y": self.dummy_expression}
        self.train_step(batch)
        print("Training step is OK")
    
    def weights_init(self, generator: Generator) -> None:
        """Initialize weights using the provided generator."""
        self.first.weights_init(generator)
        self.core.weights_init(generator)
        self.final.weights_init(generator)
    
    def apply(self, fn):
        """
        Apply a function to all submodules.
        This enables usage like: model.apply(initialize_weights)
        """
        super().apply(fn)
        return self