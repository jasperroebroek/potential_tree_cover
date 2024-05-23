from pathlib import Path
from typing import Union

import numpy as np
import rioxarray as rxr
import xarray as xr


def template_array(m: str = "data/treecover/treecover_reprojected.tif") -> xr.DataArray:
    template = rxr.open_rasterio(m, cache=False)
    arr = (
        xr.DataArray(
            data=np.nan,
            coords={'y': template.y.values,
                    'x': template.x.values},
            dims=['y', 'x']
        ).rio.write_crs("ESRI:54012", inplace=True)
    )
    return arr


def store_data(data: Union[np.ndarray, xr.DataArray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, xr.DataArray):
        t = data
    elif isinstance(data, np.ndarray):
        t = template_array()
        t.data[:] = data
    else:
        raise TypeError

    t.rio.to_raster(path, compress='DEFLATE', zlevel=9)
