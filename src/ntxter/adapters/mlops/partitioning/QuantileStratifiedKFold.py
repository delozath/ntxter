from typing import override
from dataclasses import dataclass


import numpy as np


from ntxter.core import utils

from ntxter.core.mlops.partitioning import BaseQuantileStratifiedKFold


@dataclass
class QuantileStratifiedKFoldConfig:
    n_splits: int = 5
    n_bins: int = 5
    prop_test: float = 0.2
    shuffle: bool = True
    random_state: int | None = None
    clip_outliers: str = 'skip'
    k_outlier: float = 1.5
    
    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("Number of splits must be at least 2.")
        if self.n_bins < 2:
            raise ValueError("Number of bins must be at least 2.")
        if self.n_bins > self.n_splits:
            raise ValueError("Number of bins must be lower than number of splits")


class QuantileStratifiedKFold(BaseQuantileStratifiedKFold):
    def __init__(self, **kwargs):
        self.skf_params, self._params = utils.split_dataclass_kwargs(QuantileStratifiedKFoldConfig, **kwargs)
        super().__init__(self.skf_params.n_splits)

    @override
    def _iter_test_masks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups = None,
        **kwargs
    ):
        X, y = self._validations(X, y)
        shape = X.shape[0]
        index = np.arange(shape).astype(int)

        groups, mask_outliers = self.quantiles(
            y,
            self.skf_params.n_bins,
            self.skf_params.clip_outliers,
            self.skf_params.k_outlier
         )
        
        for _ in range(self.n_splits):
            mask = np.zeros_like(y).astype(bool)
            tt_idx = np.array([], dtype=int)
            for grp in np.unique(groups):
                grp_idx = index[grp == groups]
                #
                n_grp_idx =  grp_idx.shape[0] 
                n_tt = np.ceil(n_grp_idx * self.skf_params.prop_test).astype(int)
                if n_tt < 2:
                    raise ValueError("Too many splits. Balance the trade-off between them and the number of bins")
                
                np.random.shuffle(grp_idx)
                tt_idx = np.concatenate((tt_idx, grp_idx[-n_tt:]))
            
            mask[tt_idx] = True

            yield mask