import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(
        self, seq_len, block_sizes=[16, 24, 32, 40, 48], out_ch=64, output_dim=1, kernel_size=3
    ):
        super().__init__()
        self.block_sizes = block_sizes
        self.seq_len = seq_len
        self.out_ch = out_ch
        self.output_dim = output_dim
        nn_blocks = []

        for in_bs, out_bs in zip([4] + block_sizes, block_sizes):
            block = nn.Sequential(
                nn.Conv1d(
                    in_bs, out_bs, kernel_size=kernel_size, padding=kernel_size // 2
                ),
                nn.SiLU(),
                nn.BatchNorm1d(out_bs),
                nn.Dropout(0.3),
            )
            nn_blocks.append(block)

        final_feature_size = seq_len

        self.conv_net = nn.Sequential(
            *nn_blocks,
            nn.Flatten(),
        )
        self.after_conv = nn.Sequential(
            nn.Linear(block_sizes[-1] * final_feature_size, self.out_ch),
            nn.SiLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(self.out_ch, self.out_ch),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(self.out_ch),
            nn.Linear(self.out_ch, self.output_dim),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.after_conv(x)
        if self.output_dim == 1:
            out = self.head(x).squeeze()
        else:
            out = self.head(x)

        return out