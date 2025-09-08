from ntxter.validation import UnpackDataAndCols

class BundleTrainTestSplit:
    _X = UnpackDataAndCols()
    _y = UnpackDataAndCols()
    def __init__(self, X, y):
        self._X = X
        self._y = y
        self._n_train = None
        self._n_test = None
        self._n_validation = None
    #
    @property
    def X(self):
        return self._X[UnpackDataAndCols.VALS]
    #
    @property
    def y(self):
        return self._y[UnpackDataAndCols.VALS]
    #
    @property
    def X_names(self):
        return self._X[UnpackDataAndCols.COLS]
    #
    @property
    def y_name(self):
        return self._y[UnpackDataAndCols.COLS]
    #
    @property
    def X_train(self):
        if self._n_train is not None:
            return self._X[UnpackDataAndCols.VALS][self._n_train]
        else:
            raise AttributeError("Train/test indexes must be set before calling X_train property")
    @property
    def y_train(self):
        if self._n_train is not None:
            return self._y[UnpackDataAndCols.VALS][self._n_train]
        else:
            raise AttributeError("Train/test indexes must be set before calling y_train property")
    #
    @property
    def X_test(self):
        if self._n_test is not None:
            return self._X[UnpackDataAndCols.VALS][self._n_test]
        else:
            raise AttributeError("Train/test indexes must be set before calling X_test property")
    @property
    def y_test(self):
        if self._n_test is not None:
            return self._y[UnpackDataAndCols.VALS][self._n_test]
        else:
            raise AttributeError("Train/test indexes must be set before calling y_test property")
    #
    @property
    def X_validation(self):
        if self._n_validation is not None:
            return self._X[UnpackDataAndCols.VALS][self._n_validation]
        else:
            raise AttributeError("Train/test indexes must be set before calling X_validation property")
    @property
    def y_validation(self):
        if self._n_validation is not None:
            return self._y[UnpackDataAndCols.VALS][self._n_validation]
        else:
            raise AttributeError("Train/test indexes must be set before calling y_validation property")        
    #
    def split(self, n_train, n_test, n_validation=None):
        if n_validation is not None:
            self._n_validation = n_validation
        #
        self._n_train = n_train
        self._n_test  = n_test
