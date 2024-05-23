import copy
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union, List, Sequence, Optional, Tuple, Type, Dict, Callable

import affine  # type: ignore
import cartopy.crs as ccrs  # type: ignore
import numpy as np
import rioxarray as rxr  # type: ignore
import xarray as xr
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning, CRSError
from rasterio.features import geometry_mask  # type: ignore

from src.climate_params import ClimateParams, FutureClimateParams, CurrentClimateParams
from src.paths import path_data, path_data_reprojected, path_data_processed
from src.settings import Settings, path_from_settings

template_path = path_data / "treecover" / "treecover_reprojected.tif"
template = rxr.open_rasterio(template_path, cache=False)
template_crs = "ESRI:54012"


def load_file(
    f: Union[str, Path], cache: bool = True, **kwargs
) -> Union[xr.DataArray, xr.Dataset]:
    try:
        da = rxr.open_rasterio(
            f, masked=True, parse_coordinates=True, cache=cache, **kwargs
        )
    except NotGeoreferencedWarning:
        print(f)
        raise CRSError

    return da


def load_flat_file(
    f: Union[str, Path], cache: bool = True, name: str = "data"
) -> xr.DataArray:
    da = load_file(f, cache=cache).sel(band=1).drop_vars("band")

    if isinstance(da, xr.Dataset):
        raise TypeError

    return da.rename(name)


def load_flat_file_current(path: Path, name: str, settings: Settings) -> xr.DataArray:
    if isinstance(settings.climate_params, FutureClimateParams):
        raise TypeError("This file does not exist for future conditions")
    return load_flat_file(path, name=name)


def convert_path(path: Path) -> Path:
    """Function that converts a path of the data to the reprojected data.
    NOTE: This is not always guaranteed to work"""
    return Path(path_data_reprojected, *path.parts[1:]).with_suffix(".tif")


def define_mask_range(a: np.ndarray) -> Tuple[int, int]:
    if len(a.shape) > 1:
        raise IndexError
    s = np.argmax(a).item()
    t = a.size - np.argmax(a[::-1]).item()
    return s, t


def define_mask(crs: str, transform: affine.Affine, shape: Tuple[int, int]):
    if crs == "ESRI:54012":
        projection = ccrs.EckertIV()
    else:
        projection = ccrs.Projection(crs)
    outline_mask = geometry_mask(
        [projection.boundary], out_shape=shape, transform=transform, invert=True
    )

    s, t = define_mask_range(outline_mask[:, 0])
    outline_mask[s:t, 0] = True
    s, t = define_mask_range(outline_mask[:, -1])
    outline_mask[s:t, -1] = True

    mask = np.full_like(outline_mask, fill_value=False)

    # horizontal
    for i in range(mask.shape[0]):
        s, t = define_mask_range(outline_mask[i])
        mask[i, s:t] = True

    return mask


def reproject(
    da: xr.DataArray, path: Path, resampling: Resampling = Resampling.nearest
) -> None:
    """da is reprojected and stored in path"""
    print(f"Reprojecting: da -> {path}")

    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    da_reprojected = da.rio.reproject_match(template, resampling=resampling).astype(
        np.float32
    )

    transform = da_reprojected.rio.transform(recalc=True)
    shape = (len(da_reprojected.y), len(da_reprojected.x))
    mask = define_mask(template_crs, transform, shape)
    da_reprojected.data[..., ~mask] = np.nan

    da_reprojected.attrs.pop("long_name", None)
    da_reprojected.rio.to_raster(path, compress="LZW")

    da_reprojected.close()


def verify_mergeable_settings(settings: Union[List[Settings], Settings]):
    """Check validity of params"""
    if isinstance(settings, Settings):
        settings = [settings]

    for s in settings:
        if any(
            (
                s.leave_out_regions != settings[0].leave_out_regions,
                s.weight_mid != settings[0].weight_mid,
                s.weight_high != settings[0].weight_high,
                s.weight_per_leaf != settings[0].weight_per_leaf,
            )
        ):
            raise ValueError(f"Conflicting settings: {s}, {settings[0]}")

    params = [s.climate_params for s in settings]
    for p in params:
        if type(p) == ClimateParams:
            raise TypeError("DataParams base class not allowed")

        if not isinstance(p, type(params[0])):
            raise TypeError(
                "All parameter sets need to point to either the present or the future"
            )

        if p.check_validity(file_integrity=False) is False:
            raise ValueError(f"Parameters do not exist: \n{p}")

    for i, p in enumerate(params):
        for j, p_other in enumerate(params):
            if j == i:
                continue

            if p.equals(p_other):
                raise ValueError(f"Parameters duplicated: \n{p}\n{p_other}")


@dataclass
class DataEntry:
    name: str = "data"
    subset: Union[str, int, dict, None] = None
    path: Optional[Path] = None
    path_processed: Optional[Path] = None
    resampling: Resampling = Resampling.nearest
    flat: bool = True
    process_func: Optional[Callable[["DataEntry"], None]] = None

    def get_path(self) -> Optional[Path]:
        return self.path

    def get_processed_path(self) -> Path:
        if self.path_processed is None:
            path = self.get_path()
            if path is None:
                raise ValueError("Neither path nor processed path specified")
            return convert_path(path)

        return self.path_processed

    def is_processed(self) -> bool:
        return self.get_processed_path().exists()

    def process(self) -> None:
        if self.process_func is not None:
            print(f"processing da > {self.name}")
            r = self.process_func(self)
        else:
            r = reproject_data_entry(self)

        gc.collect()
        return r


def reproject_data_entry(de: DataEntry) -> None:
    path = de.get_path()
    if path is None:
        raise ValueError(f"Can't process without a path being specified: {de.name}")

    output_path = de.get_processed_path()

    da = load_file(path, cache=False)

    if isinstance(de.subset, (str, int)):
        da = da.sel(band=de.subset)
    elif isinstance(de.subset, dict):
        da = da.sel(**de.subset)

    reproject(da, output_path, resampling=de.resampling)
    da.close()


def define_intact_forests_data_entry(de: DataEntry) -> None:
    from src.data_parsers.intact_forest import define_intact_forests

    return define_intact_forests()


def define_protected_areas_data_entry(de: DataEntry) -> None:
    from src.data_parsers.protected_areas import define_protected_areas

    return define_protected_areas()


def combine_tcc(de: DataEntry, settings: Settings) -> None:
    dr = TreecoverDisturbanceRegimeLoader().load(settings)
    pt = PotentialTreeCoverLoader().load(settings)
    tcc = (pt * (1 - dr)).rename("tcc")
    path = path_from_settings(settings, "model") / "tcc.tif"
    tcc.rio.to_raster(path, compress="LZW")


def parse_luh2(de: DataEntry):
    da = load_file(de.path, decode_times=False)[0].isel(time=-1)
    da.rio.write_crs("EPSG:4326", inplace=True)
    dap = da["primf"] + da["secdf"]
    reproject(dap, convert_path(de.path), resampling=Resampling.bilinear)


def define_future_climate_files() -> List[Path]:
    if Path("/eos").exists():
        path_sub = Path(
            "/eos",
            "jeodpp",
            "data",
            "base",
            "Meteo",
            "GLOBAL",
            "WorldClim",
            "VER2-1",
            "future",
            "Data",
            "GeoTIFF",
        )
    else:
        path_sub = path_data / "climate" / "future"

    return [Path(path) for path in path_sub.glob("wc2.1_30s_bioc_*.tif")]


def define_future_climate_data() -> List[DataEntry]:
    return [
        DataEntry(
            name=f"bio_{feature}",
            subset=feature,
            path=Path(path),
            path_processed=path_data_reprojected
            / "climate"
            / "future"
            / (Path(path).stem + f"_bio_{feature}.tif"),
        )
        for path in define_future_climate_files()
        for feature in range(1, 20)
    ]


class LoaderStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def load(self, settings: Settings) -> xr.DataArray:
        pass

    @abstractmethod
    def process(self, settings: Settings) -> None:
        pass

    @abstractmethod
    def is_processed(self, settings: Settings) -> bool:
        pass


class LoaderStrategySingle(LoaderStrategy, ABC):
    data: DataEntry
    model: bool = False
    scenario: bool = False
    period: bool = False

    def get_data_entry(self, settings: Settings) -> DataEntry:
        return self.data

    def get_path(self, settings: Settings) -> Path:
        path = self.get_data_entry(settings).get_path()
        if path is None:
            raise OSError("File does not exist")
        return path

    def get_processed_path(self, settings: Settings) -> Path:
        return self.get_data_entry(settings).get_processed_path()

    def process(self, settings: Settings) -> None:
        self.get_data_entry(settings).process()

    def is_processed(self, settings: Settings) -> bool:
        return self.get_data_entry(settings).is_processed()

    def load(self, settings: Settings) -> xr.DataArray:
        de = self.get_data_entry(settings)
        if de.flat:
            return load_flat_file(de.get_processed_path(), name=self.name)
        else:
            return load_file(de.get_processed_path()).rename(self.name)


class LoaderStrategyMultiple(LoaderStrategy, ABC):
    data: List[DataEntry]
    model: bool = False
    scenario: bool = False
    period: bool = False

    def get_data_entries(self, settings: Settings) -> List[DataEntry]:
        return self.data

    def get_paths(self, settings: Settings) -> List[Path]:
        paths = [
            data_entry.get_path() for data_entry in self.get_data_entries(settings)
        ]
        if None in paths:
            raise OSError("Not all paths can be found")
        return paths

    def get_processed_paths(self, settings: Settings) -> List[Path]:
        return [
            data_entry.get_processed_path()
            for data_entry in self.get_data_entries(settings)
        ]

    def process(self, settings: Settings) -> None:
        for data_entry in self.get_data_entries(settings):
            if not data_entry.is_processed():
                data_entry.process()

    def is_processed(self, settings: Settings) -> bool:
        for data_entry in self.get_data_entries(settings):
            if not data_entry.is_processed():
                return False

        return True

    def load(self, settings: Settings) -> xr.DataArray:
        de = self.get_data_entries(settings)[0]
        fun = load_flat_file if de.flat else load_file

        rasters = [
            fun(de.get_processed_path()).astype(np.float32).rename(de.name)
            for de in self.get_data_entries(settings)
        ]

        return xr.merge(rasters).to_array(dim="variable").rename(self.name)


class ClimateLoader(LoaderStrategyMultiple):
    name: str = "climate"

    def __init__(self, features: Optional[Union[int, Sequence[int]]] = None):
        if features is None:
            self.features = list(range(1, 20))
        elif isinstance(features, int):
            self.features = [features]
        else:
            self.features = list(features)


class CurrentClimateLoader(ClimateLoader):
    name: str = "climate"
    data: List[DataEntry] = [
        DataEntry(
            name=f"bio_{i}",
            path=path_data / "climate" / "current" / f"wc2.1_30s_bio_{i}.tif",
        )
        for i in range(1, 20)
    ]

    def get_data_entries(self, settings: Settings) -> List[DataEntry]:
        return [self.data[i - 1] for i in self.features]


class FutureClimateLoader(ClimateLoader):
    name: str = "climate_future"
    data: List[DataEntry] = define_future_climate_data()
    model: bool = True
    scenario: bool = True
    period: bool = True

    def get_data_entries(self, settings: Settings) -> List[DataEntry]:
        params = settings.climate_params
        return [
            de
            for de in self.data
            for feature in self.features
            if de.path_processed.stem
            == f"wc2.1_30s_bioc_{params.to_str()}_bio_{feature}"
        ]


class CombinedClimateLoader(ClimateLoader):
    name: str = "climate_combined"

    def get_loader(self, settings: Settings) -> ClimateLoader:
        params = settings.climate_params
        if isinstance(params, CurrentClimateParams):
            return CurrentClimateLoader(self.features)
        elif isinstance(params, FutureClimateParams):
            return FutureClimateLoader(self.features)
        else:
            raise TypeError("Needs to be CurrentDataParams or FutureDataParams")

    def get_data_entries(self, settings: Settings) -> List[DataEntry]:
        return self.get_loader(settings).get_data_entries(settings)

    def get_paths(self, settings: Settings) -> List[Path]:
        return self.get_loader(settings).get_paths(settings)

    def get_processed_paths(self, settings: Settings) -> List[Path]:
        return self.get_loader(settings).get_processed_paths(settings)

    def is_processed(self, settings: Settings) -> bool:
        return self.get_loader(settings).is_processed(settings)

    def process(self, settings: Settings) -> None:
        self.get_loader(settings).process(settings)

    def load(self, settings: Settings) -> xr.DataArray:
        return self.get_loader(settings).load(settings)


class CHELSAClimateLoader(ClimateLoader):
    name: str = "chelsa_climate"
    data: List[DataEntry] = [
        DataEntry(
            name=f"bio_{i}",
            path=path_data
            / "chelsa"
            / "v2"
            / "bioclimatic_variables"
            / f"CHELSA_bio{i}_1981-2010_V.2.1.tif",
        )
        for i in range(1, 20)
    ]

    def get_data_entries(self, settings: Settings) -> List[DataEntry]:
        if isinstance(settings.climate_params, FutureClimateParams):
            raise NotImplementedError("FutureClimateParams")
        return [self.data[i - 1] for i in self.features]


class SoilLoader(LoaderStrategyMultiple):
    data: List[DataEntry] = [
        DataEntry(name=soil_param, path=path_data / "soil" / f"{soil_param}.vrt")
        for soil_param in ["bdod", "cfvo", "clay", "sand", "silt"]
    ]
    name: str = "soil"
    model: bool = True
    scenario: bool = True
    period: bool = True


class WISESoilLoader(LoaderStrategyMultiple):
    data: List[DataEntry] = [
        DataEntry(name=soil_param, path=path_data / "WISE30sec" / f"{soil_param}.tif")
        for soil_param in ["bdod", "cfvo", "clay", "sand", "silt"]
    ]
    name: str = "soil"
    model: bool = True
    scenario: bool = True
    period: bool = True


class TopographyLoader(LoaderStrategyMultiple):
    data: List[DataEntry] = [
        DataEntry(name=topo_param, path=path_data / "topography" / f"{topo_param}.vrt")
        for topo_param in ["elevation", "aspect", "slope"]
    ]
    name: str = "topography"
    model: bool = True
    scenario: bool = True
    period: bool = True


class XLoader(LoaderStrategyMultiple):
    data = (
        CurrentClimateLoader.data
        + FutureClimateLoader.data
        + SoilLoader.data
        + TopographyLoader.data
    )
    name: str = "X"
    model: bool = True
    scenario: bool = True
    period: bool = True
    current_climate_r: Optional[xr.DataArray] = None

    @staticmethod
    def get_clim_loader(settings: Settings) -> LoaderStrategyMultiple:
        if settings.climate_loader == "climate":
            return CombinedClimateLoader()
        if settings.climate_loader == "climate_chelsa":
            return CHELSAClimateLoader()
        raise KeyError(f"Unknown climate loader: {settings.climate_loader=}")

    @staticmethod
    def get_soil_loader(settings: Settings) -> LoaderStrategyMultiple:
        SoilLoader() if settings.soil_loader == "soil" else WISESoilLoader()
        if settings.soil_loader == "soil":
            return SoilLoader()
        if settings.soil_loader == "soil_wise":
            return WISESoilLoader()
        raise KeyError(f"unknown soil loader: {settings.soil_loader}")

    def get_data_entries(self, settings: Settings) -> List[DataEntry]:
        clim_loader = self.get_clim_loader(settings)
        soil_loader = self.get_soil_loader(settings)

        return (
            clim_loader.get_data_entries(settings)
            + soil_loader.get_data_entries(settings)
            + TopographyLoader().get_data_entries(settings)
        )

    def load(self, settings: Settings) -> xr.DataArray:
        if (
            isinstance(settings.climate_params, CurrentClimateParams)
            and self.current_climate_r is not None
            and settings.climate_loader == "climate"
            and settings.soil_loader == "soil"
        ):
            print("= Loading stored features")
            return self.current_climate_r

        r = super().load(settings)

        if (
            isinstance(settings.climate_params, CurrentClimateParams)
            and settings.climate_loader == "climate"
            and settings.soil_loader == "soil"
        ):
            XLoader.current_climate_r = r

        return r


class XLoaderSub(XLoader):
    name: str = "X_sub"

    def get_data_entries(self, settings: Settings) -> List[DataEntry]:
        clim_loader = self.get_clim_loader(settings)

        return [
            clim_loader.get_data_entries(settings)[0],
            TopographyLoader().get_data_entries(settings)[0],
        ]


class TreeCoverLoader(LoaderStrategySingle):
    data: DataEntry = DataEntry(
        name="treecover", path=path_data / "treecover" / "treecover.tif"
    )
    name: str = "treecover"


class TreeCoverAfricaLoader(LoaderStrategySingle):
    data: DataEntry = DataEntry(
        name="treecover_africa", path=path_data / "treecover_africa" / "treecover_africa.tif"
    )
    name: str = "treecover_africa"


class PotentialTreeCoverBastinLoader(LoaderStrategySingle):
    name: str = "potential_treecover_bastin"
    data: DataEntry = DataEntry(
        name="potential_treecover_bastin",
        path=Path("data") / "bastin_potential_treecover" / "Total_potential.tif",
    )


class LandCoverLoader(LoaderStrategySingle):
    name: str = "landcover"
    data: DataEntry = DataEntry(
        name="landcover",
        path=Path("data") / "landcover" / "landcover.vrt",
        resampling=Resampling.mode,
    )


class CCILandCoverLoader(LoaderStrategySingle):
    name: str = "landcover_cci"
    data: DataEntry = DataEntry(
        name="landcover_cci",
        path=Path("data")
        / "landcover_cci"
        / "ESACCI-LC-L4-LCCS-Map-300m-P1Y-2000-v2.0.7.tif",
        resampling=Resampling.mode,
    )


class ClimateClassLoader(LoaderStrategySingle):
    name: str = "climate_classes"
    data: DataEntry = DataEntry(
        name="climate_classes",
        path=Path("data") / "climate_classes" / "Beck_KG_V1_present_0p0083.tif",
    )


class IntactForestLoader(LoaderStrategySingle):
    name: str = "intact_forest"
    data: DataEntry = DataEntry(
        name="intact_forest",
        path_processed=path_data_processed / "intact_forest.tif",
        process_func=define_intact_forests_data_entry,
    )


class ProtectedAreasLoader(LoaderStrategySingle):
    name: str = "protected_areas"
    data: DataEntry = DataEntry(
        name="protected_areas",
        path_processed=path_data_processed / "protected_areas.tif",
        process_func=define_protected_areas_data_entry,
    )


class ArtificialLandLoader(LoaderStrategySingle):
    name: str = "artificial_land"
    data: DataEntry = DataEntry(
        name="artificial_land",
        path=path_data
        / "artificial_land"
        / "iucn_habitatclassification_fraction_lvl1__1400_Artificial - Terrestrial__ver004.tif",
    )


class IntactLandscapeLoader(LoaderStrategyMultiple):
    name: str = "intact_landscapes"
    data: List[DataEntry] = [
        DataEntry(
            path_processed=path_data_processed / "intact_forest.tif",
            process_func=define_intact_forests_data_entry,
        ),
        DataEntry(
            path_processed=path_data_processed / "protected_areas.tif",
            process_func=define_protected_areas_data_entry,
        ),
    ]

    def load(self, settings: Settings) -> xr.DataArray:
        return ((load_data("intact_forest") + load_data("protected_areas")) > 0).rename(
            self.name
        )


class PotentialTreeCoverLoader(LoaderStrategySingle):
    name: str = "potential_treecover"
    data: List[DataEntry] = [
        DataEntry(path_processed=Path(path))
        for path in path_data_processed.rglob("potential_treecover_mean.tif")
    ]
    model: bool = True
    scenario: bool = True
    period: bool = True

    def get_data_entry(self, settings: Settings) -> DataEntry:
        return DataEntry(
            name=self.name,
            path_processed=path_from_settings(settings, "model")
            / "potential_treecover_mean.tif",
        )


class PotentialTreeCoverStdLoader(LoaderStrategySingle):
    name: str = "potential_treecover_std"
    data: List[DataEntry] = [
        DataEntry(path_processed=Path(path))
        for path in path_data_processed.rglob("potential_treecover_std.tif")
    ]
    model: bool = True
    scenario: bool = True
    period: bool = True

    def get_data_entry(self, settings: Settings) -> DataEntry:
        return DataEntry(
            name=self.name,
            path_processed=path_from_settings(settings, "model")
            / "potential_treecover_std.tif",
        )


class TreecoverDisturbanceRegimeLoader(LoaderStrategySingle):
    name: str = "treecover_disturbance_regime"
    data: List[DataEntry] = [
        DataEntry(path_processed=Path(path))
        for path in path_data_processed.rglob("treecover_disturbance_regime_mean.tif")
    ]
    model: bool = True
    scenario: bool = True
    period: bool = True

    def get_data_entry(self, settings: Settings) -> DataEntry:
        return DataEntry(
            name=self.name,
            path_processed=path_from_settings(settings, "model")
            / "treecover_disturbance_regime_mean.tif",
        )


class TreecoverDisturbanceRegimeStdLoader(LoaderStrategySingle):
    name: str = "treecover_disturbance_regime_std"
    data: List[DataEntry] = [
        DataEntry(path_processed=Path(path))
        for path in path_data_processed.rglob("treecover_disturbance_regime_std.tif")
    ]
    model: bool = True
    scenario: bool = True
    period: bool = True

    def get_data_entry(self, settings: Settings) -> DataEntry:
        return DataEntry(
            name=self.name,
            path_processed=path_from_settings(settings, "model")
            / "treecover_disturbance_regime_std.tif",
        )


class TreecoverCarryingCapacityLoader(LoaderStrategySingle):
    name: str = "tcc"
    data: List[DataEntry] = [
        DataEntry(path_processed=Path(path))
        for path in path_data_processed.rglob("tcc.tif")
    ]
    model: bool = True
    scenario: bool = True
    period: bool = True

    def get_data_entry(self, settings: Settings) -> DataEntry:
        return DataEntry(
            name=self.name,
            path_processed=path_from_settings(settings, "model") / "tcc.tif",
            process_func=partial(combine_tcc, settings=settings),
        )


class LandscapeClassLoader(LoaderStrategySingle):
    name: str = "landscape_classes"
    data: DataEntry = DataEntry(
        name="landscape_classes", path=path_data / "landscape" / "landscape.tif"
    )


class LUH2Loader(LoaderStrategySingle):
    name: str = "luh2"
    scenario: bool = True
    period: bool = True

    data: List[DataEntry] = [
        DataEntry(
            name="luh2_ssp245",
            path=(
                path_data
                / "luh2"
                / "multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc"
            ),
            process_func=parse_luh2,
        ),
        DataEntry(
            name="luh2_ssp370",
            path=(
                path_data
                / "luh2"
                / "multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_2015-2100.nc"
            ),
            process_func=parse_luh2,
        ),
    ]

    def get_data_entry(self, settings: Settings) -> DataEntry:
        if settings.climate_params is None:
            raise ValueError("Climate scenario missing in params")
        elif isinstance(settings.climate_params, CurrentClimateParams):
            raise TypeError("LUH2 is only defined for the future")
        elif (
            isinstance(settings.climate_params, FutureClimateParams)
            and settings.climate_params.period != "2081-2100"
        ):
            raise ValueError("Only valid for end of the century")

        if (
            isinstance(settings.climate_params, FutureClimateParams)
            and settings.climate_params.scenario == "ssp245"
        ):
            return self.data[0]
        elif (
            isinstance(settings.climate_params, FutureClimateParams)
            and settings.climate_params.scenario == "ssp370"
        ):
            return self.data[1]
        else:
            raise ValueError("Not covered climate scenario")


def get_loader_strategy(var: str) -> Type[LoaderStrategy]:
    # noinspection PyPep8Naming
    STRATEGIES: Dict[str, Type[LoaderStrategy]] = {
        "climate": CombinedClimateLoader,
        "climate_chelsa": CHELSAClimateLoader,
        "soil": SoilLoader,
        "soil_wise": WISESoilLoader,
        "topography": TopographyLoader,
        "X": XLoader,
        "X_sub": XLoaderSub,
        "treecover": TreeCoverLoader,
        "treecover_africa": TreeCoverAfricaLoader,
        "potential_treecover_bastin": PotentialTreeCoverBastinLoader,
        "landcover": LandCoverLoader,
        "landcover_cci": CCILandCoverLoader,
        "climate_classes": ClimateClassLoader,
        "intact_forest": IntactForestLoader,
        "protected_areas": ProtectedAreasLoader,
        "artificial_land": ArtificialLandLoader,
        "intact_landscapes": IntactLandscapeLoader,
        "potential_treecover": PotentialTreeCoverLoader,
        "potential_treecover_std": PotentialTreeCoverStdLoader,
        "treecover_disturbance_regime": TreecoverDisturbanceRegimeLoader,
        "disturbance_regime_std": TreecoverDisturbanceRegimeStdLoader,
        "tcc": TreecoverCarryingCapacityLoader,
        "landscape_classes": LandscapeClassLoader,
        "luh2": LUH2Loader,
    }

    if var not in STRATEGIES:
        raise ValueError("Loading strategy does not exist")
    else:
        return STRATEGIES[var]


def expand_dims(
    da: xr.DataArray, settings: Settings, strategy: LoaderStrategy
) -> xr.DataArray:
    params = copy.deepcopy(settings.climate_params)

    if params is None:
        return da

    dp = params.__dict__
    dims = {k: [dp[k]] for k in dp if strategy.__getattribute__(k)}

    return da.expand_dims(dims)


def _load_single(strategy, settings: Settings) -> xr.DataArray:
    if not strategy.is_processed(settings):
        strategy.process(settings)

    return expand_dims(
        strategy.load(settings), settings=settings, strategy=strategy
    ).rename(strategy.name)


def _load(
    strategy: LoaderStrategy, settings: Union[Settings, List[Settings]]
) -> xr.DataArray:
    verify_mergeable_settings(settings)

    if isinstance(settings, Settings):
        da = _load_single(strategy, settings)
    else:  # if list of dataparams
        dal = [_load_single(strategy, s) for s in settings]
        da_combined = xr.combine_by_coords(dal, combine_attrs="drop_conflicts")
        if isinstance(da_combined, xr.Dataset) and len(da_combined.data_vars):
            da = da_combined[list(da_combined.keys())[0]]
        elif isinstance(da_combined, xr.Dataset):
            raise IndexError("Too many data variables")
        else:
            da = da_combined
    return da


def load_data(
    strategy: Union[str, LoaderStrategy],
    settings: Optional[Union[Settings, List[Settings]]] = None,
) -> xr.DataArray:
    if settings is None:
        settings = Settings(climate_params=CurrentClimateParams())

    if isinstance(strategy, str):
        strategy = get_loader_strategy(strategy)()
    return _load(strategy, settings=settings)
