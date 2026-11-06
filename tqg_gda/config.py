"""Typed configuration for the SEGA-style TQG pipeline (not legacy configs)."""
from dataclasses import asdict, dataclass, field
import json
import math

from .v1.query import ProbeConfig
from .v2.query import V2Config


def _positive(value, name, integer=False, zero=False):
    if (isinstance(value, bool) or not isinstance(value, int if integer else (int, float))
            or not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(f'{name} must be a finite {"nonnegative" if zero else "positive"} number'
                         + (' (integer)' if integer else ''))


@dataclass(frozen=True)
class EncoderConfig:
    conv: str = 'gin'
    hidden: int = 64
    num_layers: int = 3
    out_dim: int = 16
    dropout: float = 0.

    def __post_init__(self):
        if self.conv not in ('gin', 'gine'):
            raise ValueError('encoder.conv must be gin or gine')
        for key in ('hidden', 'num_layers', 'out_dim'):
            _positive(getattr(self, key), key, integer=True)
        if isinstance(self.dropout, bool) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')


@dataclass(frozen=True)
class PretrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = .001
    weight_decay: float = 0.

    def __post_init__(self):
        for key in ('epochs', 'batch_size'):
            _positive(getattr(self, key), key, integer=True)
        _positive(self.lr, 'pretrain.lr')
        _positive(self.weight_decay, 'pretrain.weight_decay', zero=True)


@dataclass(frozen=True)
class AdaptConfig:
    steps: int = 200
    fit_batch_size: int = 16
    query_batch_size: int = 196
    target_batch_size: int = 16
    lr: float = .0003
    feedback_weight: float = .1
    optimizer: str = 'sgd'

    def __post_init__(self):
        for key in ('steps', 'fit_batch_size', 'query_batch_size', 'target_batch_size'):
            _positive(getattr(self, key), key, integer=True)
        _positive(self.lr, 'adapt.lr')
        _positive(self.feedback_weight, 'feedback_weight', zero=True)
        if self.optimizer not in ('sgd', 'adam'):
            raise ValueError('optimizer must be sgd or adam')


@dataclass(frozen=True)
class Config:
    cache_path: str
    source_split: str
    target_index: str
    dataset: str
    version: str = 'v2'
    seed: int = 3405
    fit_fraction: float = .8
    split_seed: int = 431
    use_node_attributes: bool = False
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    v1: ProbeConfig = field(default_factory=ProbeConfig)
    v2: V2Config = field(default_factory=V2Config)
    dtype: str = 'float32'
    device: str = 'cpu'
    num_threads: int = 1
    deterministic: bool = True
    out_dir: str = './runs/tqg'

    def __post_init__(self):
        for key in ('cache_path', 'source_split', 'target_index', 'dataset', 'out_dir', 'device'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(key + ' must be a nonempty string')
        for key in ('seed', 'split_seed'):
            _positive(getattr(self, key), key, integer=True, zero=True)
        _positive(self.num_threads, 'num_threads', integer=True)
        if self.version not in ('v1', 'v2', 'source_only'):
            raise ValueError('version must be v1, v2 or source_only')
        if self.dtype not in ('float32', 'float64'):
            raise ValueError('dtype must be float32 or float64')
        if isinstance(self.fit_fraction, bool) or not 0 < self.fit_fraction < 1:
            raise ValueError('fit_fraction must be in (0, 1)')
        if not isinstance(self.deterministic, bool) or not isinstance(self.use_node_attributes, bool):
            raise ValueError('deterministic and use_node_attributes must be booleans')
        if self.v1.direction != 'target' or self.v1.responses != 'odd_even':
            raise ValueError('Canonical v1 uses target directions and odd+even; use legacy workflows for ablations')
        if self.v2.signature != 'representation_response' or self.v2.calibration.get('mode', 'joint') != 'joint':
            raise ValueError('Canonical v2 uses H+A and joint calibration; use diagnostic workflows for ablations')
        if self.version == 'v2' and (self.adapt.optimizer != 'sgd' or self.adapt.query_batch_size < 2):
            raise ValueError('v2 requires momentum-free SGD and at least two query graphs')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, kind in (('encoder', EncoderConfig), ('pretrain', PretrainConfig),
                          ('adapt', AdaptConfig), ('v1', ProbeConfig), ('v2', V2Config)):
            if key in values:
                values[key] = kind(**values[key])
        return cls(**values)

    @classmethod
    def from_json(cls, path):
        with open(path) as stream:
            return cls.from_dict(json.load(stream))
