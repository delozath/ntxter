import numpy as np

from sklearn.model_selection import StratifiedKFold
from ntxter.validation import ArrayIndexSlice, UnpackDataAndCols


class StratifiedKFoldWrapper(StratifiedKFold):
    X_train   = ArrayIndexSlice()
    y_train   = ArrayIndexSlice()
    X_test    = ArrayIndexSlice()
    y_test    = ArrayIndexSlice()
    #X_unseen  = ArrayIndexSlice()
    #y_unseen  = ArrayIndexSlice()
    #
    def __init__(self, X, y, n_splits=5, shuffle=True, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.X = X
        self.y = y
    #
    def split(self, groups=None):
        for tn_idx, tt_idx in super().split(self.X, self.y, groups):
            self.X_train = self.X, tn_idx
            self.y_train = self.y, tn_idx
            self.X_test  = self.X, tt_idx
            self.y_test  = self.y, tt_idx
            yield