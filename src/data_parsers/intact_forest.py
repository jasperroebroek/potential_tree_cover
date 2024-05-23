from pathlib import Path

import geopandas as gpd
from rasterio import features

from src.misc import template_array
from src.paths import path_data_processed, path_data


def define_intact_forests() -> None:
    path = path_data_processed / "intact_forest.tif"
    if path.exists():
        return

    print("- Define intact forest")
    path.parent.mkdir(parents=True, exist_ok=True)

    template = template_array()
    IFL = gpd.read_file(path_data / "intact_forest" / "ifl_2016.shp").to_crs("ESRI:54012")

    shapes = ((geom, 1) for geom in IFL.geometry)
    template.data[:] = features.rasterize(shapes=shapes, fill=0, out_shape=template.shape,
                                          transform=template.rio.transform(recalc=True))

    template.rio.to_raster(path, compress='LZW')
