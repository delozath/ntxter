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
    clip_outliers: str = 'tukey'
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

    def split_outliers_clipping(self, X, y):
       index = np.arange(y.shape[0])
       for mask_idx, outlier_idx in self._quantile_groups(X, y):
            tn_idx = np.setdiff1d(index[~mask_idx], outlier_idx)
            tt_idx = np.setdiff1d(index[ mask_idx], outlier_idx)

            yield tn_idx, tt_idx, outlier_idx
    
    def split(self, X, y=None, groups=None):
        for tn_idx, tt_idx in super().split(X, y, groups):
            yield tn_idx, tt_idx

    @override
    def _iter_test_masks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups = None,
        **kwargs
    ):
        for mask_idx, _ in self._quantile_groups(X, y):
            yield mask_idx
    
    def _quantile_groups(self, X, y):
        X, y = self._validations(X, y)
        shape = X.shape[0]
        index = np.arange(shape).astype(int)

        groups = self.quantiles(y, self.skf_params.n_bins)
        outlier_mask = self._outliers_clipping(y)
        outlier_idx = index[~outlier_mask]

        for _ in range(self.n_splits):
            mask_idx = np.zeros_like(y).astype(bool)
            tt_idx = np.array([], dtype=int)
            for grp in np.unique(groups):
                grp_idx = index[grp == groups]
                n_grp_idx =  grp_idx.shape[0]

                n_tt = np.ceil(n_grp_idx * self.skf_params.prop_test).astype(int)
                if n_tt < 2:
                    raise ValueError("Too many splits. Balance the trade-off between them and the number of bins")
                
                np.random.shuffle(grp_idx)
                tt_idx_part = grp_idx[:n_tt]
                tt_idx = np.concatenate((tt_idx, tt_idx_part), axis=0)
                
            mask_idx[tt_idx] = True

            yield mask_idx, outlier_idx

    def _outliers_clipping(self, y):
        if self.skf_params.clip_outliers == 'tukey':
            qn = np.array([np.quantile(y, q) for q in [0.25, 0.75]])
            irq = np.diff(qn)

            outliers = (self.skf_params.k_outlier * irq) * [-1, 1] + qn #Tukey rule
            mask_outliers = (y>=outliers[0]) & (y<=outliers[1])
            return mask_outliers
        else:
            raise NotImplementedError(f"clip method `{self.skf_params.clip_outliers}` is not supported.")