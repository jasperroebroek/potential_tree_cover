from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Optional, List, Dict

import numpy as np
import xarray as xr
from src.sampling.sampling import define_sampling_mask
from focal_stats import rolling_window

from src.climate_params import FutureClimateParams, CurrentClimateParams
from src.data import load_data
from src.misc import template_array, store_data
from src.paths import path_data_processed
from src.settings import Settings, path_from_settings


@dataclass
class LandCoverClass:
    name: str
    idx: int
    # tc (treecover):
    #   -> None: nodata
    #   ->    0: no treecover
    #   ->   -1: treecover present in data
    tc: Optional[int] = None
    # weight:
    #   ->    1: weight 1
    #   ->    2: weight middle
    #   ->    3: weight high
    weight: Optional[int] = None


def define_landcover_classes() -> List[LandCoverClass]:
    return [
        LandCoverClass('nodata', 0),
        LandCoverClass('shrubs', 20, tc=-1, weight=2),
        LandCoverClass('herbaceous', 30, tc=-1, weight=2),
        LandCoverClass('cultivated', 40),
        LandCoverClass('urban', 50),
        LandCoverClass('bare', 60, tc=0, weight=3),
        LandCoverClass('snow_ice', 70, tc=0, weight=3),
        LandCoverClass('water', 80),
        LandCoverClass('wetland', 90, tc=-1, weight=1),
        LandCoverClass('moss_lichen', 100, tc=0, weight=3),
        LandCoverClass('ocean', 200),
    ] + [
        # All forest classes are created equal
        LandCoverClass('forest', i, tc=-1, weight=1)
        for i in [111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126]
    ]


@dataclass
class ClipRegion:
    minx: float
    maxx: float
    miny: float
    maxy: float


def define_clip_regions() -> List[ClipRegion]:
    # Defined in ESRI:54012
    return [
        ClipRegion(5_000_000, 6_500_000, 6_500_000, 8_000_000),  # Siberia
        ClipRegion(400_000, 3_100_000, -3_000_000, -300_000),  # Sub Sahara Africa
        ClipRegion(-6_500_000, -4_500_000, -6_000_000, -4_000_000),  # South America
        ClipRegion(-10_000_000, -8_500_000, 4_500_000, 6_000_000),  # North America
        ClipRegion(12_000_000, 14_000_000, -2_500_000, -500_000),  # Oceania
        ClipRegion(-6_500_000, -5_000_000, -500_000, 1_000_000),  # Amazon
    ]


def define_clip_mask() -> xr.DataArray:
    t = template_array()
    t.data[:] = 0

    for region in define_clip_regions():
        t.loc[
            dict(x=slice(region.minx, region.maxx), y=slice(region.maxy, region.miny))
        ] = 1

    path = path_data_processed / 'clip_mask.tif'
    if not path.exists():
        store_data(t, path=path)

    return t


def parse_landcover_classes(
    weight_map: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, int], Dict[str, int]]:
    if weight_map is None:
        weight_map = {1: 1, 2: 2, 3: 3}

    l = define_landcover_classes()

    class_division = {}
    class_tc = {}
    class_weight = {}

    for lcc in l:
        if lcc.tc is None:
            continue

        if lcc.name in class_division:
            class_division[lcc.name].append(lcc.idx)
        else:
            class_division[lcc.name] = [lcc.idx]

        if lcc.name in class_tc:
            if not class_tc[lcc.name] == lcc.tc:
                raise ValueError
        class_tc[lcc.name] = lcc.tc

        if lcc.name in class_weight:
            if not class_weight[lcc.name] == weight_map[lcc.weight]:
                raise ValueError
        class_weight[lcc.name] = weight_map[lcc.weight]

    return class_division, class_tc, class_weight


def parse_landcover(
    lc: np.ndarray, tc: np.ndarray, climate: np.ndarray, artifical_land: np.ndarray
) -> np.ndarray:
    # IF landcover is shrubs or grass in wet tropical areas, they are removed as being non-natural
    lc[np.logical_and(climate == 1, lc == 20)] = 220
    lc[np.logical_and(climate == 1, lc == 30)] = 230

    # IF grassland in artificial land (Martin Jung above 10%) they are removed
    lc[np.logical_and(artifical_land > 100, lc == 30)] = 230

    # Grassland and Shrubs directly adjacent to agriculture are removed as they are considered non-natural
    agriculture_rolling = (rolling_window(lc, window_size=3) == 40).sum(axis=(2, 3))
    lc[1:-1, 1:-1][np.logical_and(lc[1:-1, 1:-1] == 20, agriculture_rolling)] = 320
    lc[1:-1, 1:-1][np.logical_and(lc[1:-1, 1:-1] == 30, agriculture_rolling)] = 330

    # IF not enough wetland pixels are available, remove them as the algorithm is not detailed enough to pick this up
    wetland_rolling = (rolling_window(lc, window_size=5) == 90).sum(axis=(2, 3)) < 4
    lc[2:-2, 2:-2][np.logical_and(wetland_rolling, lc[2:-2, 2:-2] == 90)] = 290

    # IF forests have lower than 15% treecover they are removed
    lc[np.logical_and.reduce([tc < 1500, lc > 100, lc < 200])] = 250

    # Setting nodata where no treecover or no landcover data is available
    lc[np.isnan(tc)] = 0
    lc[np.isnan(lc)] = 0

    path = path_data_processed / 'landcover_augmented.tif'
    if not path.exists():
        store_data(lc, path)

    return lc.astype(np.int32)


@lru_cache(maxsize=1)
def load_parse_landcover() -> Tuple[np.ndarray, np.ndarray]:
    """
    (treecover, landcover_parsed)
    """
    print('Loading forests')
    treecover_raster = load_data('treecover')
    treecover = treecover_raster.values
    treecover_raster.close()

    landcover_raster = load_data('landcover')
    landcover = landcover_raster.values
    landcover_raster.close()

    climate_classes_raster = load_data('climate_classes')
    climate_classes = climate_classes_raster.values
    climate_classes_raster.close()

    artificial_land_raster = load_data('artificial_land')
    artificial_land = artificial_land_raster.values
    artificial_land_raster.close()

    landcover_parsed = parse_landcover(
        landcover, treecover, climate_classes, artificial_land
    )
    return treecover, landcover_parsed


def define_sampling(
    settings: Settings, landcover: np.ndarray, target: np.ndarray, name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: sample_mask, sampled_target, sample_weights
    """
    print('- Creating sampling mask')

    weight_map = {1: 1, 2: settings.weight_mid, 3: settings.weight_high}

    class_division, class_tc, class_weight = parse_landcover_classes(weight_map)

    sample_mask = define_sampling_mask(
        target,
        landcover,
        settings.input_reduction_factor,
        {c: class_division[c] for c in class_division if class_weight[c] is not None},
        class_tc,
    )

    # Assign zero treecover where needed (bare/snow and ice/lichen and moss)
    assign_zero_treecover = [
        lcc.idx for lcc in define_landcover_classes() if lcc.tc == 0
    ]
    target[np.isin(landcover, assign_zero_treecover)] = 0

    # Create weight map
    weights = np.full_like(landcover, np.nan, dtype=np.float64)
    for lcc in define_landcover_classes():
        if lcc.weight is not None:
            weights[landcover == lcc.idx] = weight_map[lcc.weight]

    print('- Storing sampling mask')
    base_path = path_from_settings(settings, 'model')

    path = base_path / f'sample_mask_{name}.tif'
    if not path.exists():
        store_data(sample_mask, path=path)

    path = base_path / f'sampled_{name}.tif'
    if not path.exists():
        store_data(target, path=path)

    path = base_path / f'sample_weights_{name}.tif'
    if not path.exists():
        store_data(weights, path=path)

    return sample_mask, target, weights


def load_future_feature_data(settings: Settings) -> List[np.ndarray]:
    """:returns
    X_test, test_mask
    """
    params = settings.climate_params
    if not isinstance(params, FutureClimateParams):
        raise TypeError('Loading future data with CurrentClimateParams')

    print('Loading future data')
    features_raster = load_data('X', settings).sel(
        model=params.model, scenario=params.scenario, period=params.period
    )

    features = features_raster.values

    if not isinstance(settings.climate_params, CurrentClimateParams):
        features_raster.close()

    mask = np.isnan(features.sum(axis=0)) == 0

    return [features[:, mask].T, mask]
