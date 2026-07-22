import torch 
import torch.nn as nn 

from typing import List
import torch.nn.functional as F

from torch import Generator

from typing import Any, Dict
from abc import ABCMeta, abstractmethod

from typing import List
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

class ConvBlock(nn.Module):
    """
    Basic convolutional block.
    Consists of a convolutional layer, a max pooling layer and a dropout layer.
    """
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        pool_size: int, 
        dropout: float
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.mp = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        x = F.relu(self.conv(x))  # (batch_size, out_channels, seq_len)
        x = self.mp(x)  # (batch_size, out_channels, seq_len // pool_size)
        x = self.do(x)  # (batch_size, out_channels, seq_len // pool_size)
        return x

class BHIFirstLayersBlock(FirstLayersBlock):
    """
    The firstLayersBlock of the BHI model.
    Consists of multiple ConvBlocks with different kernel sizes.
    Output of each ConvBlock is concatenated along the channel dimension.
    """
    def __init__(
        self, 
        in_channels: int = 5,
        out_channels: int = 512,
        seqsize: int = 110,
        kernel_sizes: List[int] = [9, 15],
        pool_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__(in_channels, out_channels, seqsize)
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
        each_out_channels = out_channels // len(kernel_sizes)

        self.conv_list = nn.ModuleList([
            ConvBlock(in_channels, each_out_channels, k, pool_size, dropout) for k in kernel_sizes
        ])

    
    def forward(self, x):
        # x: (batch_size, 4, seq_len), 4 channels: A, C, G, T
        if len(x.shape) < 3:
            x = F.one_hot(x.to(torch.int64), self.in_channels)
            x = x.float().permute(0,2,1)

        # get the output of each convolutional layer
        conv_outputs = [conv(x) for conv in self.conv_list]  # [(batch_size, each_out_channels, seq_len // pool_size), ...]

        # concatenate the outputs along the channel dimension
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, out_channels, seq_len // pool_size)

        return x

class BHICoreBlock(CoreBlock):
    """
    The coreBlock of the BHI model.
    Consists of a bidirectional LSTM layer, multiple ConvBlocks with different kernel sizes and a dropout layer.
    LSTM layer is used for capturing long-range dependencies.
    ConvBlocks consolidate the soft-dependencies into hard-dependencies.
    Output of each ConvBlock is concatenated along the channel dimension same as the FirstCNNBlock.
    """
    def __init__(
        self, 
        in_channels: int = 512,
        out_channels: int = 320,
        seqsize: int = 110,
        lstm_hidden_channels: int = 320,
        kernel_sizes: List[int] = [9, 15],
        pool_size: int = 1,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__(in_channels, out_channels, seqsize)
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
        each_conv_out_channels = out_channels // len(kernel_sizes)

        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_channels, batch_first=True, bidirectional=True)
        self.conv_list = nn.ModuleList([
            ConvBlock(2 * lstm_hidden_channels, each_conv_out_channels, k, pool_size, dropout1) for k in kernel_sizes
        ])
        self.do = nn.Dropout(dropout2)

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, in_channels)
        x, _ = self.lstm(x)  # (batch_size, seq_len, 2 * lstm_hidden_channels)
        x = x.permute(0, 2, 1)  # (batch_size, 2 * lstm_hidden_channels, seq_len)
        
        # get the output of each convolutional layer
        conv_outputs = [conv(x) for conv in self.conv_list]  # [(batch_size, each_conv_out_channels, seq_len // pool_size), ...]

        # concatenate the outputs along the channel dimension
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, conv_out_channels, seq_len // pool_size)

        x = self.do(x)  # (batch_size, conv_out_channels, seq_len // pool_size)

        return x


class BHIFinalLayersBlock(FinalLayersBlock):
    def __init__(
        self,
        in_channels: int = 320,
        seqsize: int = 110,
        hidden_dim: int = 64,
        out_channels=1
    ):
        super().__init__(in_channels, seqsize)
        self.flat = nn.Flatten()
        self.main = nn.Sequential(
            nn.Linear(in_channels * seqsize, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

        self.criterion = nn.HuberLoss()

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        x = self.flat(x)  # (batch_size, in_channels * seq_len)
        x = self.main(x)  # (batch_size, output_dim)
        return x.squeeze(-1)  # (batch_size,)

    def train_step(self, batch: Dict[str, Any]):
        x = batch["x"].to(self.device)
        pred = self.forward(x)
       
        y = batch["y"].to(self.device).to(torch.float32)
        loss = self.criterion(pred, y.squeeze(-1))
            
        return y, loss

class BHI(nn.Module):
    """
    BHI model class that encapsulates the entire architecture.
    Initializes with specified in_channels, seqsize, and out_channels.
    """
    def __init__(
        self,
        in_channels: int = 5,
        seqsize: int = 110,
        out_channels: int = 1,
        first_out_channels: int = 320,
        core_out_channels: int = 320,
        lstm_hidden_channels: int = 320,
        kernel_sizes: List[int] = [9, 15],
        pool_size: int = 1,
        dropout1: float = 0.2,
        dropout2: float = 0.5,
        final_hidden_dim: int = 64,
        generator: Generator | None = None
    ):
        super().__init__()
        
        if generator is None:
            generator = torch.Generator()
        
        # Build first block
        self.first = BHIFirstLayersBlock(
            in_channels=in_channels,
            out_channels=first_out_channels,
            seqsize=seqsize,
            kernel_sizes=kernel_sizes,
            pool_size=pool_size,
            dropout=dropout1
        )
        
        # Build core block
        self.core = BHICoreBlock(
            in_channels=self.first.out_channels,
            out_channels=core_out_channels,
            seqsize=self.first.infer_outseqsize(),
            lstm_hidden_channels=lstm_hidden_channels,
            kernel_sizes=kernel_sizes,
            pool_size=pool_size,
            dropout1=dropout1,
            dropout2=dropout2
        )
        
        # Build final block
        self.final = BHIFinalLayersBlock(
            in_channels=self.core.out_channels,
            seqsize=self.core.infer_outseqsize(),
            hidden_dim=final_hidden_dim,
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