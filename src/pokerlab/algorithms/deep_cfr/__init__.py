"""Deep CFR for small imperfect-information poker games.

Importing this package does not require PyTorch. Constructing neural networks
requires the optional ``deep-cfr`` extra.
"""

from pokerlab.algorithms.deep_cfr.config import DeepCFRConfig
from pokerlab.algorithms.deep_cfr.encoding import KuhnCodec, LeducCodec
from pokerlab.algorithms.deep_cfr.inference import DeepCFRPolicy
from pokerlab.algorithms.deep_cfr.trainer import DeepCFRTrainer

__all__ = [
    "DeepCFRConfig",
    "DeepCFRPolicy",
    "DeepCFRTrainer",
    "KuhnCodec",
    "LeducCodec",
]
