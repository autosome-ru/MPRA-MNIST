import warnings
from typing import Optional, Sequence, Union
 
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

 
Number = Union[int, float]


class Compose:
    """
    Composes several transforms together. This transform does not support torchscript.

    Parameters
    ----------
    transforms : List[Callable]
        List of transformations to apply sequentially.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        for transformation in self.transforms:
            target = transformation(target)

        return target

    def __repr__(self):
        transformations = "\n    ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}(\n    {transformations}\n)"


class Normalize(nn.Module):
    """
    A module for normalizing a target tensor with specified mean and standard deviation.

    This class applies normalization to input data using the formula:
    normalized_target = (target - mean) / std

    Attributes:
    ----------
    mean : float or torch.Tensor
        The mean value used for normalization.
    std : float or torch.Tensor
        The standard deviation used for normalization.
    """

    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, target):
        target = (target - self.mean) / self.std
        return target

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})" 
 
def _as_tensor(target, dtype=torch.float64) -> Tensor:
    if torch.is_tensor(target):
        return target.to(dtype)
    return torch.as_tensor(target, dtype=dtype)
 

class SoftBinTarget(nn.Module):
    """Scalar expression -> soft probability vector over FACS bins.
 
    Reproduces `SeqDatasetProb.__getitem__`'s `y_probs` exactly:
 
        norm  = N(loc=target + shift, scale=scale)
        probs = norm.cdf(POINTS)[1:] - norm.cdf(POINTS)[:-1]
 
    Parameters
    ----------
    n_bins : int
        Number of output classes (18 in the challenge). Ignored when `points`
        is given explicitly.
    shift : float
        Offset added to the target before it is used as the mean. The default
        0.5 centres the normal in the middle of the bin, because bin `k` covers
        the interval (k, k+1].
    scale : float
        Standard deviation of the assumed measurement distribution. Larger
        values spread mass over neighbouring bins (more label smoothing);
        `scale -> 0` approaches a one-hot label.
    points : sequence of float, optional
        Explicit bin edges, length `n_bins + 1`. Defaults to
        `[-inf, 1, 2, ..., n_bins - 1, +inf]`, i.e. the challenge's POINTS.
    normalize : bool
        Renormalize the vector to sum to 1. With the default infinite outer
        edges the sum is already 1 up to float error; it matters only if you
        pass finite outer edges.
    dtype : torch.dtype
        Output dtype (float32 by default, matching the reference dataset).
    strict_range : bool
        Warn once if a target falls outside `[points[1] - 1, points[-2] + 1]`,
        which usually means the labels are not on the bin scale (e.g. the
        val/test MAUDE values, or an already-normalized target).
 
    Shape
    -----
    Input: scalar or tensor of shape (...); Output: (..., n_bins).
 
    Examples
    --------
    >>> tf = SoftBinTarget()
    >>> tf(torch.tensor(11.0)).shape
    torch.Size([18])
    >>> tf(torch.tensor(11.0)).argmax().item()
    11
    """
 
    def __init__(
        self,
        n_bins: int = 18,
        shift: float = 0.5,
        scale: float = 0.5,
        points: Optional[Sequence[Number]] = None,
        normalize: bool = False,
        dtype: torch.dtype = torch.float32,
        strict_range: bool = False,
    ):
        super().__init__()
        if points is None:
            points = np.array([-np.inf, *range(1, n_bins), np.inf], dtype=np.float64)
        else:
            points = np.asarray(points, dtype=np.float64)
            n_bins = len(points) - 1
        if len(points) != n_bins + 1:
            raise ValueError(
                f"points must have n_bins + 1 = {n_bins + 1} edges, got {len(points)}"
            )
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
 
        self.n_bins = int(n_bins)
        self.shift = float(shift)
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.out_dtype = dtype
        self.strict_range = bool(strict_range)
        self._warned = False
 
        # buffers so the transform follows .to()/pickling like any nn.Module
        self.register_buffer("points", torch.from_numpy(points), persistent=False)
        finite = points[np.isfinite(points)]
        self._lo = float(finite.min() - 1.0) if finite.size else -np.inf
        self._hi = float(finite.max() + 1.0) if finite.size else np.inf
 
    # -- helpers ---------------------------------------------------------- #
    def _check_range(self, target: Tensor) -> None:
        if not self.strict_range or self._warned:
            return
        if bool(((target < self._lo) | (target > self._hi)).any()):
            self._warned = True
            warnings.warn(
                f"{type(self).__name__}: target(s) outside [{self._lo}, {self._hi}]. "
                "Rafi2024 train labels are FACS bins (0..17); val/test labels are "
                "MAUDE estimates on a different scale and should not be soft-binned.",
                stacklevel=2,
            )
 
    # -- forward ---------------------------------------------------------- #
    def forward(self, target) -> Tensor:
        target = _as_tensor(target, torch.float64)
        self._check_range(target)
 
        loc = target + self.shift
        edges = self.points.to(device=target.device, dtype=torch.float64)
 
        # (..., n_bins + 1); +-inf edges give cdf exactly 0 / 1
        z = (edges.expand(*target.shape, edges.shape[0]) - loc.unsqueeze(-1)) / self.scale
        cdf = torch.special.ndtr(z)
        probs = cdf[..., 1:] - cdf[..., :-1]
 
        if self.normalize:
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
 
        return {'value': target, 'probs': probs.to(self.out_dtype)}
 
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_bins={self.n_bins}, shift={self.shift}, "
            f"scale={self.scale}, normalize={self.normalize})"
        )
 
class ExpectedBin(nn.Module):
    """Inverse of `SoftBinTarget`: probability vector -> expected bin value.
 
    Mirrors `score = (softmax(logits) * arange(n_bins)).sum(1)` in
    `AutosomeFinalLayersBlock.forward`. Useful for sanity-checking the encoding
    and for converting model outputs back to a scalar for correlation metrics.
 
    Parameters
    ----------
    n_bins : int
        Number of classes.
    centers : sequence of float, optional
        Value assigned to each bin. Defaults to `arange(n_bins)`, matching the
        `bins` buffer of the reference final block. Note this pairs with
        `shift=0.5`, so a round-trip is offset by `shift` by construction.
    """
 
    def __init__(self, n_bins: int = 18, centers: Optional[Sequence[Number]] = None):
        super().__init__()
        if centers is None:
            centers = np.arange(n_bins, dtype=np.float32)
        else:
            centers = np.asarray(centers, dtype=np.float32)
            n_bins = len(centers)
        self.n_bins = int(n_bins)
        self.register_buffer("centers", torch.from_numpy(centers), persistent=False)
 
    def forward(self, probs: Tensor) -> Tensor:
        centers = self.centers.to(device=probs.device, dtype=probs.dtype)
        return (probs * centers).sum(-1)
 
    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_bins={self.n_bins})"
 
