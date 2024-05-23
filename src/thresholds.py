from pathlib import Path

import rioxarray as rxr
import numpy as np

from src.data import load_data, ClimateLoader
from src.misc import template_array


class PotentialTreecoverClipper:
    def __init__(self):
        self.treecover = load_data('treecover').values
        self.potential_treecover = load_data('potential_treecover').values

        self.treecover_mask = self.treecover > 0

        self.path = Path("data_processed/thresholds")
        self.path.mkdir(parents=True, exist_ok=True)

    def clip_climate(self, feature: int) -> None:
        if feature < 1 or feature > 19:
            raise ValueError("Value out of range")

        data = load_data(ClimateLoader(feature)).values[0]
        data_with_trees = data[self.treecover_mask]

        threshold_low = np.nanmin(data_with_trees)
        threshold_high = np.nanmax(data_with_trees)

        threshold_low_mask = data < threshold_low
        threshold_high_mask = data > threshold_high

        da_below = template_array()
        da_below.data[threshold_low_mask] = self.potential_treecover[threshold_low_mask]
        da_below.rio.to_raster(self.path / f"bio_{feature}_below.tif", compress="LZW")

        da_above = template_array()
        da_above.data[threshold_high_mask] = self.potential_treecover[threshold_high_mask]
        da_above.rio.to_raster(self.path / f"bio_{feature}_above.tif", compress="LZW")


clipper = PotentialTreecoverClipper()
for i in range(1, 20):
    print(f"Clipping climate: {i}")
    clipper.clip_climate(i)
