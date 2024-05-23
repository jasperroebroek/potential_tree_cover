from typing import Optional, List, Union

import numpy as np
import xarray as xr
from sklearn.metrics import zero_one_loss, mean_pinball_loss, r2_score, mean_squared_error

from src.data import load_data
from src.data_preprocessing import load_parse_landcover, define_clip_mask


class Scorer:
    def __init__(self):
        print("Init scorer")
        treecover, landcover = load_parse_landcover()
        intact_landscapes = load_data("intact_landscapes").values
        leave_out_regions = define_clip_mask()

        # protected areas that are not falling in the ocean
        self.mask = np.logical_and.reduce(
            [intact_landscapes == 1, landcover != 0, landcover != 200, ~np.isnan(treecover),
             leave_out_regions == 1]
        )
        self.y_true = treecover[self.mask]
        self.y_true_digits = self.digitize(self.y_true)

    @staticmethod
    def digitize(v: np.ndarray, bins: Optional[List[int]] = None) -> np.ndarray:
        if bins is None:
            bins = [0, 1000, 10001]
        return np.digitize(v, bins=bins, right=True)

    def score(self, da: Union[np.ndarray, xr.DataArray], bins: Optional[List[int]] = None) -> float:
        if isinstance(da, xr.DataArray):
            da = da.values
        vals = da[self.mask]
        nan_filter = ~np.isnan(vals)
        y_pred_digitized = self.digitize(vals[nan_filter], bins=bins)
        y_true_digitized = self.digitize(self.y_true[nan_filter], bins=bins)
        return zero_one_loss(y_true_digitized, y_pred_digitized)

    def score_pinball(self, da: Union[np.ndarray, xr.DataArray], alpha: float = 1) -> float:
        if isinstance(da, xr.DataArray):
            da = da.values
        y_pred = da[self.mask]
        nan_filter = ~np.isnan(y_pred)
        return mean_pinball_loss(self.y_true[nan_filter], y_pred[nan_filter], alpha=alpha)

    def score_r2(self, da: Union[np.ndarray, xr.DataArray]) -> float:
        if isinstance(da, xr.DataArray):
            da = da.values
        y_pred = da[self.mask]
        nan_filter = ~np.isnan(y_pred)
        return r2_score(self.y_true[nan_filter], y_pred[nan_filter])

    def score_rmse(self, da: Union[np.ndarray, xr.DataArray]) -> float:
        if isinstance(da, xr.DataArray):
            da = da.values
        y_pred = da[self.mask]
        nan_filter = ~np.isnan(y_pred)
        return mean_squared_error(self.y_true[nan_filter], y_pred[nan_filter]) ** 0.5
