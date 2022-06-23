from collections import Sequence
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import torch

from skorch.cli import parse_args  # pylint: disable=unused-import
from skorch.utils import _make_split
from skorch.utils import is_torch_data_type
from skorch.utils import to_tensor


class SliceDataset(Sequence):
    # pylint: disable=anomalous-backslash-in-string

    def __init__(self, dataset, idx=0, indices=None):
        self.dataset = dataset
        self.idx = idx
        self.indices = indices

        self.indices_ = (self.indices if self.indices is not None
                         else np.arange(len(self.dataset)))
        self.ndim = 1

    def __len__(self):
        return len(self.indices_)

  