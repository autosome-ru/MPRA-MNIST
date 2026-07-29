import math
import torch
import torch.nn as nn
from collections import OrderedDict
import lightning.pytorch as L

class L1KLmixed(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0, beta=5.0):
        super().__init__()

        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta

        self.MSE = nn.L1Loss(reduction=reduction.replace("batch", ""))
        self.KL = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, preds, targets):
        preds_log_prob = preds - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)

        MSE_loss = self.MSE(preds, targets)
        KL_loss = self.KL(preds_log_prob, target_log_prob)

        combined_loss = MSE_loss.mul(self.alpha) + KL_loss.mul(self.beta)

        return combined_loss.div(self.alpha + self.beta)
    
class Conv1dNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        batch_norm=True,
        weight_norm=True,
    ):
        super(Conv1dNorm, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(
                out_channels,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )

    def forward(self, input):
        try:
            return self.bn_layer(self.conv(input))
        except AttributeError:
            return self.conv(input)


class LinearNorm(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, batch_norm=True, weight_norm=True
    ):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(
                out_features,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )

    def forward(self, input):
        try:
            return self.bn_layer(self.linear(input))
        except AttributeError:
            return self.linear(input)


class GroupedLinear(nn.Module):
    def __init__(self, in_group_size, out_group_size, groups):
        super().__init__()

        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups

        # initialize weights
        self.weight = torch.nn.Parameter(
            torch.zeros(groups, in_group_size, out_group_size)
        )
        self.bias = torch.nn.Parameter(torch.zeros(groups, 1, out_group_size))

        # change weights to kaiming
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        reorg = (
            x.permute(1, 0)
            .reshape(self.groups, self.in_group_size, -1)
            .permute(0, 2, 1)
        )
        hook = torch.bmm(reorg, self.weight) + self.bias
        reorg = (
            hook.permute(0, 2, 1)
            .reshape(self.out_group_size * self.groups, -1)
            .permute(1, 0)
        )

        return reorg


class RepeatLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.repeat(*self.args)


class BranchedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_group_size,
        out_group_size,
        n_branches=1,
        n_layers=1,
        activation="ReLU",
        dropout_p=0.5,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_group_size = hidden_group_size
        self.out_group_size = out_group_size
        self.n_branches = n_branches
        self.n_layers = n_layers

        self.branches = OrderedDict()

        self.nonlin = getattr(nn, activation)()
        self.dropout = nn.Dropout(p=dropout_p)

        self.intake = RepeatLayer(1, n_branches)
        cur_size = in_features

        for i in range(n_layers):
            if i + 1 == n_layers:
                setattr(
                    self,
                    f"branched_layer_{i + 1}",
                    GroupedLinear(cur_size, out_group_size, n_branches),
                )
            else:
                setattr(
                    self,
                    f"branched_layer_{i + 1}",
                    GroupedLinear(cur_size, hidden_group_size, n_branches),
                )
            cur_size = hidden_group_size

    def forward(self, x):
        hook = self.intake(x)

        i = -1
        for i in range(self.n_layers - 1):
            hook = getattr(self, f"branched_layer_{i + 1}")(hook)
            hook = self.dropout(self.nonlin(hook))
        hook = getattr(self, f"branched_layer_{i + 2}")(hook)

        return hook


def get_padding(kernel_size):
    """
    Calculate padding values for convolutional layers.

    Args:
        kernel_size (int): Size of the convolutional kernel.

    Returns:
        list: Padding values for left and right sides of the kernel.
    """
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    return [max(0, x) for x in [left, right]]


class BassetBranched(L.LightningModule):
    """BassetBranched model from SJ Gosai et al., 2023;
    see <https://pmc.ncbi.nlm.nih.gov/articles/PMC10441439/>
    and original git-code: https://github.com/sjgosai/boda2/blob/main/boda/model/basset.py
    """

    ######################
    # Model construction #
    ######################

    def __init__(
        self,
        input_len=600,
        n_channels=4,  
        conv1_channels=300,
        conv1_kernel_size=19,
        conv2_channels=200,
        conv2_kernel_size=11,
        conv3_channels=200,
        conv3_kernel_size=7,
        n_linear_layers=1,
        linear_channels=1000,
        linear_activation="ReLU",
        linear_dropout_p=0.11625456877954289,
        n_branched_layers=3,
        branched_channels=140,
        branched_activation="ReLU",
        branched_dropout_p=0.5757068086404574,
        n_outputs=3,
        use_batch_norm=True,
        use_weight_norm=False,
        loss_criterion="L1KLmixed",
        loss_args={},
    ):
        super().__init__()

        self.input_len = input_len
        self.n_channels = n_channels

        self.conv1_channels = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = get_padding(conv1_kernel_size)

        self.conv2_channels = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = get_padding(conv2_kernel_size)

        self.conv3_channels = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = get_padding(conv3_kernel_size)

        self.n_linear_layers = n_linear_layers
        self.linear_channels = linear_channels
        self.linear_activation = linear_activation
        self.linear_dropout_p = linear_dropout_p

        self.n_branched_layers = n_branched_layers
        self.branched_channels = branched_channels
        self.branched_activation = branched_activation
        self.branched_dropout_p = branched_dropout_p

        self.n_outputs = n_outputs

        self.loss_criterion = loss_criterion
        self.loss_args = loss_args

        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm

        self.pad1 = nn.ConstantPad1d(self.conv1_pad, 0.0)
        self.conv1 = Conv1dNorm(
            self.n_channels,
            self.conv1_channels,
            self.conv1_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=self.use_batch_norm,
            weight_norm=self.use_weight_norm,
        )
        self.pad2 = nn.ConstantPad1d(self.conv2_pad, 0.0)
        self.conv2 = Conv1dNorm(
            self.conv1_channels,
            self.conv2_channels,
            self.conv2_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=self.use_batch_norm,
            weight_norm=self.use_weight_norm,
        )
        self.pad3 = nn.ConstantPad1d(self.conv3_pad, 0.0)
        self.conv3 = Conv1dNorm(
            self.conv2_channels,
            self.conv3_channels,
            self.conv3_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=self.use_batch_norm,
            weight_norm=self.use_weight_norm,
        )

        self.pad4 = nn.ConstantPad1d((1, 1), 0.0)

        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)

        next_in_channels = self.conv3_channels * self.get_flatten_factor(self.input_len)

        for i in range(self.n_linear_layers):
            setattr(
                self,
                f"linear{i + 1}",
                LinearNorm(
                    next_in_channels,
                    self.linear_channels,
                    bias=True,
                    batch_norm=self.use_batch_norm,
                    weight_norm=self.use_weight_norm,
                ),
            )
            next_in_channels = self.linear_channels

        self.branched = BranchedLinear(
            next_in_channels,
            self.branched_channels,
            self.branched_channels,
            self.n_outputs,
            self.n_branched_layers,
            self.branched_activation,
            self.branched_dropout_p,
        )

        self.output = GroupedLinear(self.branched_channels, 1, self.n_outputs)

        self.nonlin = getattr(nn, self.linear_activation)()

        self.dropout = nn.Dropout(p=self.linear_dropout_p)

    # original method. not using because of different sequence batch length in some datasets
    """ 
    def get_flatten_factor(self, input_len):
        hook = input_len
        assert hook % 3 == 0
        hook = hook // 3
        assert hook % 4 == 0
        hook = hook // 4
        assert (hook + 2) % 4 == 0

        return (hook + 2) // 4
    """

    def get_flatten_factor(self, input_len):
        x = torch.zeros(1, self.n_channels, input_len)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.maxpool_3(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.maxpool_4(x)
        x = self.pad3(x)
        x = self.conv3(x)
        x = self.pad4(x)
        x = self.maxpool_4(x)
        return x.shape[2] 

    ######################
    # Model computations #
    ######################

    def encode(self, x):
        hook = self.nonlin(self.conv1(self.pad1(x)))
        hook = self.maxpool_3(hook)
        hook = self.nonlin(self.conv2(self.pad2(hook)))
        hook = self.maxpool_4(hook)
        hook = self.nonlin(self.conv3(self.pad3(hook)))
        hook = self.maxpool_4(self.pad4(hook))
        hook = torch.flatten(hook, start_dim=1)
        return hook

    def decode(self, x):
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout(self.nonlin(getattr(self, f"linear{i + 1}")(hook)))
        hook = self.branched(hook)
        return hook

    def classify(self, x):
        output = self.output(x)
        return output

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        output = self.classify(decoded)
        output = output.squeeze(-1)
        return output