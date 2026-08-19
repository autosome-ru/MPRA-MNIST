import torch.nn as nn
from mpramnist.dataclass import ScalarFeature, Categorial, seqobj

SingletonCategory = Categorial(levels={0:0, 1:1})
class IsSingleton(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Seq: seqobj, target) -> seqobj:
        is_integer = (target == target.round()).item()
        Seq.scalars['is_singleton'] = ScalarFeature(1 if is_integer else 0, tp=SingletonCategory) 
        return Seq

    def __repr__(self):
        return self.__class__.__name__ + "()"


class NotASingleton(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Seq: seqobj) -> seqobj:
        Seq.scalars['is_singleton'] = ScalarFeature(0, tp=SingletonCategory) 
        return Seq

    def __repr__(self):
        return self.__class__.__name__ + "()"