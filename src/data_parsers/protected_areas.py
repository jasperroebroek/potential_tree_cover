from pathlib import Path

import geopandas as gpd
import pandas as pd
from rasterio import features

from src.misc import template_array


def define_protected_areas() -> None:
    print("- Define protected areas")

    paths = Path("data", "protected_areas").rglob("WDPA_*polygons.shp")
    dfs = []
    for f in paths:
        print(f)
        cdf = gpd.read_file(f)
        cdf = cdf.loc[cdf.IUCN_CAT.isin(('Ib', 'Ia'))]
        dfs.append(cdf)

    df = pd.concat(dfs).to_crs("ESRI:54012")

    shapes = ((geom, 1) for geom in df.geometry)

    output_path = Path("data_processed", "protected_areas.tif")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    template = template_array()
    template.data[:] = features.rasterize(shapes=shapes, fill=0, out_shape=template.shape,
                                          transform=template.rio.transform(recalc=True))
    template.rio.to_raster(output_path, compress='LZW')
