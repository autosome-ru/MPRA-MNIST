"""
PyTorch port of ReporterNet (kundajelab/reporterNet, reporterNet/model/mpra_model.py).

Оригинал написан на Keras: свёрточный "стем" -> N residual-блоков с растущей
дилатацией -> GlobalAveragePooling1D -> Dense(1).

Отличия/соответствия с Keras-версией:
  * Keras Conv1D(activation=...) применяет активацию ДО residual-сложения,
    после сложения активации нет — здесь так же.
  * Keras BatchNormalization по (B, L, C) == nn.BatchNorm1d(C) по (B, C, L).
  * Keras Dropout на 3D-тензоре зануляет отдельные элементы (не каналы),
    nn.Dropout делает то же самое.
  * Keras Conv1D по умолчанию инициализируется glorot_uniform + zeros bias:
    для этого есть initialize_weights_reporternet().

Вход: (B, 4, L) — как и у остальных моделей mpramnist (Seq2Tensor даёт [4, L]).
Модель полностью свёрточная + global average pooling, поэтому работает
с любой длиной L (в статье использовалось 600 с плазмидными фланками).
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ReporterNet",
    "initialize_weights_reporternet",
    "MSEMinusPearsonLoss",
    "pearson_r",
]


def _get_activation(activation):
    """'relu' / 'silu' / 'gelu' / nn.Module-класс -> nn.Module."""
    if isinstance(activation, str):
        return {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
        }[activation.lower()]()
    if isinstance(activation, type):
        return activation()
    return activation


def initialize_weights_reporternet(m):
    """
    Инициализация в стиле Keras-дефолтов (glorot_uniform для весов, нули для bias),
    чтобы поведение совпадало с оригинальной TF-реализацией.
    Использование: model.apply(initialize_weights_reporternet)
    """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class DilatedResidualBlock(nn.Module):
    """conv_x = act(Conv1d(x, dilation=d)); x = conv_x + x; [BN]; [Dropout]"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation="relu",
        batch_norm: bool = False,
        dropout_rate: float | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )
        self.act = _get_activation(activation)
        self.bn = nn.BatchNorm1d(channels) if batch_norm else nn.Identity()
        self.dropout = (
            nn.Dropout(dropout_rate)
            if dropout_rate is not None and dropout_rate > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_x = self.act(self.conv(x))
        x = conv_x + x                      # keras.layers.add([conv_x, x])
        x = self.bn(x)
        x = self.dropout(x)
        return x


class ReporterNet(nn.Module):
    """
    Parameters
    ----------
    in_ch : int
        Число входных каналов (4 для one-hot; больше, если добавлены
        feature/reverse-каналы через transforms).
    output_dim : int
        Размерность выхода (1 для single-cell-type, len(cell_types) для multi).
    filters : int
        Число фильтров во всех свёртках (в статье 512).
    num_dilation_layers : int
        Число residual-блоков (в статье 8).
    dilation_rates : Sequence[int]
        Коэффициенты дилатации, len >= num_dilation_layers.
    conv1_kernel_size : int
        Размер ядра стем-свёртки (7 при обучении с нуля, 21 при transfer
        learning из ChromBPNet).
    dilation_kernel_size : int
        Размер ядра в residual-блоках (3).
    dropout_rate : float | Sequence[float] | None
        Один общий dropout или список длины num_dilation_layers + 1
        (первый элемент — после стема, остальные — после каждого блока).
        В скриптах оригинала передавалось "0.1,0.1,...,0.1" (9 значений).
    activation : str
        'relu' (по умолчанию) или 'silu'.
    batch_norm : bool
        Добавлять BatchNorm после стема, после каждого блока и после пулинга.
    """

    def __init__(
        self,
        in_ch: int = 4,
        output_dim: int = 1,
        filters: int = 512,
        num_dilation_layers: int = 8,
        dilation_rates: Sequence[int] = (2, 4, 8, 16, 32, 32, 32, 32),
        conv1_kernel_size: int = 7,
        dilation_kernel_size: int = 3,
        dropout_rate=None,
        activation: str = "relu",
        batch_norm: bool = False,
    ):
        super().__init__()

        if len(dilation_rates) < num_dilation_layers:
            raise ValueError(
                f"Нужно как минимум {num_dilation_layers} значений dilation_rates, "
                f"получено {len(dilation_rates)}"
            )

        # dropout: скаляр -> список длины num_dilation_layers + 1
        if dropout_rate is None or isinstance(dropout_rate, (int, float)):
            dropouts = [dropout_rate] * (num_dilation_layers + 1)
        else:
            dropouts = list(dropout_rate)
            if len(dropouts) != num_dilation_layers + 1:
                raise ValueError(
                    f"dropout_rate должен быть скаляром или списком длины "
                    f"{num_dilation_layers + 1}, получено {len(dropouts)}"
                )

        self.in_ch = in_ch
        self.output_dim = output_dim
        self.filters = filters

        # --- stem: Conv1D(filters, conv1_kernel_size, padding='same', act) ---
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, filters, kernel_size=conv1_kernel_size, padding="same"),
            _get_activation(activation),
            nn.BatchNorm1d(filters) if batch_norm else nn.Identity(),
            nn.Dropout(dropouts[0])
            if dropouts[0] is not None and dropouts[0] > 0
            else nn.Identity(),
        )

        # --- residual-блоки с дилатацией ---
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    channels=filters,
                    kernel_size=dilation_kernel_size,
                    dilation=dilation_rates[i],
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout_rate=dropouts[i + 1],
                )
                for i in range(num_dilation_layers)
            ]
        )

        # --- голова: GAP -> [BN] -> Dense(output_dim), линейная активация ---
        self.post_pool_bn = nn.BatchNorm1d(filters) if batch_norm else nn.Identity()
        self.head = nn.Linear(filters, output_dim)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Выход bottleneck-слоя (B, filters, L) — удобно для интерпретации/TL."""
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.get_embedding(x)                 # (B, filters, L)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)   # GlobalAveragePooling1D
        x = self.post_pool_bn(x)
        x = self.head(x)                          # (B, output_dim)
        return x.squeeze(-1)                      # как в HumanLegNet/MPRAnn


# --------------------------------------------------------------------------- #
# Метрики / лоссы из reporterNet/utils/metrics_util.py
# --------------------------------------------------------------------------- #
def pearson_r(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-4):
    """Дифференцируемый Pearson r (порт pearson_r из metrics_util.py)."""
    x = y_true - y_true.mean()
    y = y_pred - y_pred.mean()
    num = (x * y).sum()
    den = torch.sqrt((x * x).sum() * (y * y).sum())
    return num / (den + eps)


class MSEMinusPearsonLoss(nn.Module):
    """combined_loss из оригинала: MSE - PearsonR. В train_mpra_model_k562.sh
    использовался чистый 'mse', но комбинированный лосс тоже доступен."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_pred, y_true) - pearson_r(y_true, y_pred)
