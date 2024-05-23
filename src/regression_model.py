import threading

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import _accumulate_prediction, RandomForestRegressor
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn_quantile import RandomForestMaximumRegressor
from sklearn_quantile.ensemble.quantile import _accumulate_prediction as _accumulate_prediction_sklearn_quantile


def predict_std(m, X):
    check_is_fitted(m)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(m.n_estimators, m.n_jobs)

    # apply method requires X to be of dtype np.float32
    X = check_array(X, dtype=np.float32, accept_sparse="csc")

    predictions = np.empty((len(X), m.n_estimators))

    lock = threading.Lock()
    Parallel(n_jobs=n_jobs, verbose=m.verbose, require="sharedmem")(
        delayed(_accumulate_prediction_sklearn_quantile)(est.predict, X, i, predictions, lock)
        for i, est in enumerate(m.estimators_)
    )

    return np.std(predictions, axis=-1)


class ExtendedRandomForestMaximumRegressor(RandomForestMaximumRegressor):
    def predict_mean(self, X):
        """
        Direct copy from ForestRegressor in sklearn
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.estimators_
        )

        y_hat /= len(self.estimators_)

        return y_hat

    def predict_std(self, X):
        return predict_std(self, X)

    def predict_max(self, X):
        return self.predict(X)


class ExtendedRandomForestRegressor(RandomForestRegressor):
    def predict_std(self, X):
        return predict_std(self, X)

    def predict_mean(self, X):
        return self.predict(X)
