import copy
from typing import List

import cartopy.crs as ccrs
import focal_stats as fs
import geomappy as mp
import geomappy.colors
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, hex2color
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from skimage.measure import label

from src.climate_params import define_future_params, FutureClimateParams
from src.data import load_data
from src.data_preprocessing import define_clip_mask, define_clip_regions
from src.misc import template_array
from src.parameter_optimisation import parameter_optimisation
from src.settings import Settings

km2 = r"km$^{2}$"
delta = r"$\it{\Delta}$"

climate_params_245 = define_future_params(
    scenarios="ssp245", periods="2081-2100", verify_file_integrity=True
)

settings_current = parameter_optimisation()
settings_current_leave_out_regions = parameter_optimisation()
settings_current_leave_out_regions.leave_out_regions = True
settings_245 = []
for c in climate_params_245:
    s = parameter_optimisation()
    s.climate_params = c
    settings_245.append(s)

print("Loading treecover")
treecover = load_data("treecover") / 10000

print("Loading weather stations")
# weather_stations = gpd.read_file("data/weather_stations/data/commonData\Data0\stations1.shp")

print("Loading tcc and Bastin")
tcc_current = load_data("tcc", settings_current) / 10000
tcc_bastin = load_data("potential_treecover_bastin") / 100

print("Loading tcc future")
tcc_ssp245_full = []
for s in settings_245:
    print(f"- {s}")
    tcc_ssp245_full.append(
        load_data("tcc", s).isel(model=0, scenario=0, period=0) / 10000
    )

print("- processing mean")
tcc_ssp245_sum = None
for t in tcc_ssp245_full:
    print(t.model.item())
    if tcc_ssp245_sum is None:
        tcc_ssp245_sum = t.copy()
        continue
    tcc_ssp245_sum += t
tcc_ssp245 = tcc_ssp245_sum / len(tcc_ssp245_full)

print("- processing loss")
model_loss_sum = None
for t in tcc_ssp245_full:
    print(t.model.item())
    if model_loss_sum is None:
        model_loss_sum = ((t - treecover) < 0).astype(np.int32)
        continue
    model_loss_sum += ((t - treecover) < 0).astype(np.int32)
model_loss = model_loss_sum / len(tcc_ssp245_full)

print("Loading leave out regions")
tcc_leave_out_regions = load_data("tcc", settings_current_leave_out_regions) / 10000
leave_out_regions = define_clip_mask()

print("Loading landscape")
landcover = load_data("landcover")
topography = load_data("topography")
elevation = topography.sel(variable="elevation")
slope = topography.sel(variable="slope")
intact_areas = load_data("intact_landscapes")

treecover.data[np.isnan(tcc_current.data)] = np.nan
elevation.data[np.isnan(tcc_current.data)] = np.nan
slope.data[np.isnan(tcc_current.data)] = np.nan

print("Treecover std")
treecover_std = template_array()
treecover_std.data[:] = fs.focal_std(treecover.data, window_size=5)

print("TCC std")
tcc_std = template_array()
tcc_std.data[:] = fs.focal_std(tcc_current.data, window_size=5)

print("TCC Bastin std")
tcc_bastin_std = template_array()
tcc_bastin_std.data[:] = fs.focal_std(tcc_bastin.data, window_size=5)

print("Elevation std")
elevation_std = template_array()
elevation_std.data[:] = fs.focal_std(elevation.data, window_size=5)

print("Load LUH2 ssp245")
luh2_treecover = load_data(
    "luh2",
    Settings(
        climate_params=FutureClimateParams(
            model="suppress_validation", scenario="ssp245", period="2081-2100"
        )
    ),
).sel(period="2081-2100", scenario="ssp245")

intact_tcc_mask = np.logical_and.reduce(
    [~np.isnan(treecover.data), ~np.isnan(tcc_current.data), intact_areas]
)
intact_tcc_bastin_mask = np.logical_and.reduce(
    [~np.isnan(treecover.data), ~np.isnan(tcc_bastin.data), intact_areas]
)
leave_out_regions_intact_mask = np.logical_and(
    leave_out_regions.data == 1, intact_tcc_mask
)
leave_out_regions_intact_bastin_mask = np.logical_and(
    leave_out_regions.data == 1, intact_tcc_bastin_mask
)


def plot_left_out_areas():
    with mpl.rc_context({"hatch.linewidth": 0.1}):
        f, ax = plt.subplots(
            figsize=(9, 9), subplot_kw=dict(projection=ccrs.EckertIV())
        )
        ax.set_global()
        ax.coastlines(linewidth=0.2)
        mp.add_gridlines(ax, 30, alpha=0.2)

        for region in define_clip_regions():
            width = region.maxx - region.minx
            height = region.maxy - region.miny
            ax.add_patch(
                Rectangle(
                    (region.minx, region.miny),
                    width,
                    height,
                    linewidth=0.1,
                    edgecolor="r",
                    facecolor="lightgrey",
                    alpha=0.5,
                )
            )
            ax.add_patch(
                Rectangle(
                    (region.minx, region.miny),
                    width,
                    height,
                    linewidth=0,
                    edgecolor="black",
                    facecolor="none",
                    hatch="///",
                )
            )

        plt.savefig("figures/leave_out_areas.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_intact_landscapes():
    ia = intact_areas.copy(deep=True)
    ia.data[np.isnan(tcc_current.data)] = False
    ia_low_res = ia.astype(np.float32).rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=0, resampling=Resampling.max
    )

    common_kwargs = dict(
        interpolation="nearest",
        transform=ccrs.EckertIV(),
    )

    f, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection=ccrs.EckertIV()))
    ax.set_global()
    ax.coastlines(linewidth=0.2)
    mp.add_gridlines(ax, 30, alpha=0.2)

    mp.plot_classified_raster(
        ia_low_res.data,
        extent=ia_low_res.get_extent(),
        ax=ax,
        legend=None,
        levels=[0, 1],
        colors=["none", "sienna"],
        **common_kwargs,
    )
    plt.savefig("figures/intact_areas.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_raw_landcover():
    landcover_low_res = landcover.rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=0
    )

    classes = [
        0,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        111,
        112,
        113,
        114,
        115,
        116,
        121,
        122,
        123,
        124,
        125,
        126,
        200,
    ]
    colors = [
        "#282828",
        "#ffbb22",
        "#ffff4c",
        "#f096ff",
        "#fa0000",
        "#b4b4b4",
        "#f0f0f0",
        "#0032c8",
        "#0096a0",
        "#fae6a0",
        "#58481f",
        "#009900",
        "#70663e",
        "#00cc00",
        "#4e751f",
        "#007800",
        "#666000",
        "#8db400",
        "#8d7400",
        "#a0dc00",
        "#929900",
        "#648c00",
        "#000080",
    ]
    labels = [
        "No data",
        "Shrubs",
        "Herbaceous vegetation",
        "Cultivated and managed vegetation",
        "Urban / built up",
        "Bare / sparse vegetation",
        "Snow and ice",
        "Permanent water bodies",
        "Herbaceous wetland",
        "Moss and lichen",
        "Closed forest, evergreen needle leaf",
        "Closed forest, evergreen broad leaf",
        "Closed forest, deciduous needle leaf",
        "Closed forest, deciduous broad leaf",
        "Closed forest, mixed",
        "Closed forest, not matching any of the other definitions",
        "Open forest, evergreen needle leaf",
        "Open forest, evergreen broad leaf",
        "Open forest, deciduous needle leaf",
        "Open forest, deciduous broad leaf",
        "Open forest, mixed",
        "Open forest, not matching any of the other definitions",
        "Oceans/ seas",
    ]

    common_kwargs = dict(
        interpolation="nearest",
        transform=ccrs.EckertIV(),
    )

    f, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection=ccrs.EckertIV()))
    ax.set_global()
    ax.coastlines(linewidth=0.2)
    mp.add_gridlines(ax, 30, alpha=0.2)

    ax, cbar = mp.plot_classified_raster(
        landcover_low_res.data,
        extent=landcover_low_res.get_extent(),
        ax=ax,
        levels=classes,
        colors=colors,
        labels=labels,
        suppress_warnings=True,
        legend_kw=dict(position="right", aspect=30),
        **common_kwargs,
    )
    cbar.ax.invert_yaxis()
    plt.savefig("figures/landcover.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_raw_treecover():
    treecover_low_res = treecover.rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=np.nan
    )

    colors: List = mp.colors.colors_discrete("Greens", 9).tolist()
    colors[0] = "#fdf0ff"
    cmap = ListedColormap(colors)
    bins = [0, 0.15, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]

    common_kwargs = dict(
        interpolation="nearest",
        transform=ccrs.EckertIV(),
    )

    f, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=ccrs.EckertIV()))
    ax.set_global()
    ax.coastlines(linewidth=0.2)
    mp.add_gridlines(ax, 30, alpha=0.2)

    ax, cbar = mp.plot_raster(
        treecover_low_res.data,
        extent=treecover_low_res.get_extent(),
        ax=ax,
        bins=bins,
        cmap=cmap,
        legend_kw=dict(position="right", aspect=40, shrink=0.7),
        **common_kwargs,
    )
    cbar.set_label("Tree cover fraction [-]", labelpad=8)
    plt.savefig("figures/treecover.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_bastin_comparison_histogram():
    bins = np.linspace(-0.65, 0.65, 25)
    f, ax = plt.subplots()
    ax.hist(
        tcc_current.data[intact_tcc_mask] - treecover.data[intact_tcc_mask],
        bins=bins,
        alpha=0.7,
        density=True,
        label="This study",
        color="lightgrey",
        hatch="//",
    )
    ax.hist(
        tcc_bastin.data[intact_tcc_bastin_mask]
        - treecover.data[intact_tcc_bastin_mask],
        bins=bins,
        alpha=1,
        label="Bastin et al. 2019",
        density=True,
        color="#726F97",
        zorder=-1,
    )
    ax.set_xlabel("Deviation to tree cover in intact landscapes [-]")
    ax.set_ylabel("Density [-]")
    ax.legend()
    plt.savefig("figures/bastin_comparison_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_bastin_comparison_panel(clip, n, h_pad, w_pad):
    common_kwargs = dict(
        interpolation="nearest", transform=ccrs.EckertIV(), decorate_basemap=False
    )
    colors: List = mp.colors.colors_discrete("Greens", 9).tolist()
    colors[0] = "#fdf0ff"
    treecover_kwargs = dict(
        **common_kwargs,
        bins=[0, 0.15, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1],
        cmap=ListedColormap(colors),
    )
    std_kwargs = dict(
        **common_kwargs, cmap="Greys", bins=[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]
    )

    f, ax = plt.subplots(
        ncols=2, nrows=4, figsize=(10, 20), subplot_kw=dict(projection=ccrs.EckertIV())
    )
    ax = ax.flatten()

    for cax in ax:
        cax.set_extent((clip[0], clip[2], clip[1], clip[3]), crs=ccrs.EckertIV())
        cax.coastlines("50m", linewidth=0.5)
        mp.add_ticks(cax, 10)
        mp.add_gridlines(cax, 10)

    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)

    treecover.rio.clip_box(*clip).plot_raster(ax=ax[0], **treecover_kwargs)
    treecover_std.rio.clip_box(*clip).plot_raster(ax=ax[1], **std_kwargs)
    tcc_current.rio.clip_box(*clip).plot_raster(ax=ax[2], **treecover_kwargs)
    tcc_std.rio.clip_box(*clip).plot_raster(ax=ax[3], **std_kwargs)
    tcc_bastin.rio.clip_box(*clip).plot_raster(ax=ax[4], **treecover_kwargs)
    tcc_bastin_std.rio.clip_box(*clip).plot_raster(ax=ax[5], **std_kwargs)
    (
        elevation.rio.clip_box(*clip).plot_raster(
            ax=ax[6],
            bins=[0, 2, 5, 10, 25, 100, 250, 500, 1000, 1500, 2500],
            cmap="terrain",
            **common_kwargs,
        )
    )
    (
        elevation_std.rio.clip_box(*clip).plot_raster(
            ax=ax[7], bins=[0, 5, 10, 25, 50, 100, 250, 500, 1000], **common_kwargs
        )
    )

    ax[0].set_title("Tree cover")
    ax[1].set_title("Tree cover std")
    ax[2].set_title("Tree cover carrying capacity")
    ax[3].set_title("Tree cover carrying capacity std")
    ax[4].set_title("Bastin potential tree cover")
    ax[5].set_title("Bastin potential tree cover std")
    ax[6].set_title("Elevation")
    ax[7].set_title("Elevation std")

    for i in range(8):
        ax[i].text(
            -0.03,
            0.98,
            chr(65 + i),
            va="top",
            ha="right",
            weight="bold",
            fontsize=14,
            transform=ax[i].transAxes,
        )

    plt.savefig(f"figures/bastin_comparison_{n}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_result_maps():
    tcc_current_low_res = tcc_current.rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=np.nan
    )
    tcc_dif_low_res = (tcc_ssp245 - tcc_current).rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=np.nan
    )

    common_kwargs = dict(
        interpolation="nearest",
        transform=ccrs.EckertIV(),
        legend_kw=dict(position="right", shrink=0.8, aspect=40),
    )
    colors: List = mp.colors.colors_discrete("Greens", 9).tolist()
    colors[0] = "#fdf0ff"

    treecover_kwargs = dict(
        **common_kwargs,
        bins=[0, 0.15, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1],
        cmap=ListedColormap(colors),
    )
    treecover_change_kwargs = dict(
        **common_kwargs,
        cmap="BrBG",
        bins=[-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5],
    )

    f, ax = plt.subplots(
        nrows=2, figsize=(8, 8), subplot_kw=dict(projection=ccrs.EckertIV())
    )
    for cax in ax:
        cax.set_global()
        cax.coastlines(linewidth=0.2)
        mp.add_gridlines(cax, 30, alpha=0.2)
    plt.tight_layout()

    ax1, cbar1 = mp.plot_raster(
        tcc_current_low_res.data,
        extent=tcc_current_low_res.get_extent(),
        **treecover_kwargs,
        ax=ax[0],
    )
    ax2, cbar2 = mp.plot_raster(
        tcc_dif_low_res.data,
        extent=tcc_current_low_res.get_extent(),
        **treecover_change_kwargs,
        ax=ax[1],
    )

    cbar1.set_label("Potential tree cover [-]")
    cbar2.set_label("Potential tree cover change [-]")

    ax1.text(
        0,
        0.99,
        "A",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        weight="bold",
        fontsize=12,
    )
    ax2.text(
        0,
        0.99,
        "B",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        weight="bold",
        fontsize=12,
    )

    plt.savefig("figures/results.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_future_forest_loss():
    model_loss_low_res = model_loss.astype(np.float64).rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=np.nan
    )
    treecover_loss = tcc_ssp245 - treecover
    treecover_loss_low_res = -treecover_loss.rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=np.nan
    )
    treecover_loss_low_res.data[treecover_loss_low_res.data < 0] = 0

    treecover_change_kwargs = dict(
        interpolation="nearest",
        transform=ccrs.EckertIV(),
        legend_kw=dict(position="right", shrink=0.8, aspect=40),
        cmap="Oranges",
        bins=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    )

    f, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=ccrs.EckertIV()))
    ax.set_global()
    ax.coastlines(linewidth=0.2)
    mp.add_gridlines(ax, 30, alpha=0.2)
    plt.tight_layout()

    ax, cbar = mp.plot_raster(
        treecover_loss_low_res.data,
        extent=model_loss_low_res.get_extent(),
        **treecover_change_kwargs,
        ax=ax,
    )

    cbar.set_label("Potential tree cover loss [-]")

    plt.savefig("figures/potential_treecover_loss.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_reforestation_potential():
    # reforestation potential
    treecover_loss = tcc_current - treecover
    treecover_loss_low_res = treecover_loss.rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=np.nan
    )
    treecover_loss_bastin = tcc_bastin - treecover
    treecover_loss_bastin_low_res = treecover_loss_bastin.rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=np.nan
    )

    common_kwargs = dict(
        interpolation="nearest",
        transform=ccrs.EckertIV(),
        legend_kw=dict(position="right", shrink=0.8, aspect=40),
    )
    treecover_change_kwargs = dict(
        **common_kwargs,
        cmap="BrBG",
        bins=[-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5],
    )

    f, ax = plt.subplots(
        nrows=2, figsize=(8, 8), subplot_kw=dict(projection=ccrs.EckertIV())
    )
    for cax in ax:
        cax.set_global()
        cax.coastlines(linewidth=0.2)
        mp.add_gridlines(cax, 30, alpha=0.2)
    plt.tight_layout()

    ax1, cbar1 = mp.plot_raster(
        treecover_loss_low_res.data,
        extent=treecover_loss_low_res.get_extent(),
        **treecover_change_kwargs,
        ax=ax[0],
    )
    ax2, cbar2 = mp.plot_raster(
        treecover_loss_bastin_low_res.data,
        extent=treecover_loss_low_res.get_extent(),
        **treecover_change_kwargs,
        ax=ax[1],
    )

    cbar1.set_label("Reforestation potential [-]")
    cbar2.set_label("Reforestation potential [-]")

    plt.savefig(
        "figures/reforestation_potential_compared_to_bastin.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    f, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=ccrs.EckertIV()))
    ax.set_global()
    ax.coastlines(linewidth=0.2)
    mp.add_gridlines(ax, 30, alpha=0.2)
    plt.tight_layout()

    ax, cbar = mp.plot_raster(
        treecover_loss_low_res.data,
        extent=treecover_loss_low_res.get_extent(),
        **treecover_change_kwargs,
        ax=ax,
    )

    cbar.set_label("Reforestation potential [-]")

    plt.savefig("figures/reforestation_potential.png", dpi=300, bbox_inches="tight")
    plt.show()

    d = {}
    for lc_class in np.unique(landcover):
        if np.isnan(lc_class):
            continue
        mask = landcover.data == lc_class
        d[int(lc_class)] = (
            treecover.where(mask).sum().item(),
            treecover_loss.where(mask).sum().item(),
            treecover_loss_bastin.where(mask).sum().item(),
        )
        print(int(lc_class))
        print(f"treecover {d[int(lc_class)][0]:.2f}")
        print(f"treecover delta {d[int(lc_class)][1]:.2f}")
        print(f"treecover delta Bastin {d[int(lc_class)][2]:.2f}")

    descriptions = {
        "20": "shrubland",
        "30": "herbaceous vegetation",
        "40": "cultivated and managed vegetation",
        "50": "urban/ built up",
        "60": "bare/ sparse vegetation",
        "70": "snow and ice",
        "90": "wetlands",
        "100": "moss and lichen",
        ("111", "112", "113", "114", "115", "116"): "closed forest",
        ("121", "122", "123", "124", "125", "126"): "open forest",
    }
    ys = list(np.arange(len(descriptions)) * 2)

    f, ax = plt.subplots()
    for i, k in enumerate(list(descriptions.keys())):
        if isinstance(k, tuple):
            keys = k
        else:
            keys = (k,)

        reforestation = sum([d[int(l)][1] for l in keys]) / 1_000_000
        reforestation_bastin = sum([d[int(l)][2] for l in keys]) / 1_000_000
        ax.barh(
            ys[i] - 0.36,
            reforestation,
            height=0.65,
            color="lightgrey",
            label="This study" if not i else None,
        )
        ax.barh(
            ys[i] + 0.36,
            reforestation_bastin,
            height=0.65,
            color="#726F97",
            label="Bastin et al. (2019)" if not i else None,
        )

    ax.set_yticks(ys, descriptions.values())
    # ax.set_xlim(None, 5.2)
    ax.set_ylim(-2.5, ys[-1] + 2)
    ax.axvline(0, linestyle="--", color="darkblue", alpha=0.6, linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel(f"Reforestation potential [million {km2}]")
    ax.tick_params(length=0, pad=15, axis="y")
    ax.legend(loc=1)

    plt.tight_layout()
    plt.savefig(
        "figures/reforestation_potential_bars_2.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_luh2_mismatch():
    treecover_mismatch = luh2_treecover - (tcc_ssp245 > 0.30)
    treecover_mismatch.data[treecover_mismatch.data < 0] = 0
    treecover_mismatch_low_res = treecover_mismatch.rio.reproject(
        ccrs.EckertIV(), resolution=(10000, 10000), nodata=np.nan
    )

    treecover_mismatch_kwargs = dict(
        interpolation="nearest",
        transform=ccrs.EckertIV(),
        legend_kw=dict(position="right", shrink=0.8, aspect=40),
        cmap="YlOrBr",
        bins=[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.5, 0.75, 1],
    )

    f, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=ccrs.EckertIV()))
    ax.set_global()
    ax.coastlines(linewidth=0.2)
    mp.add_gridlines(ax, 30, alpha=0.2)
    plt.tight_layout()

    _, cbar = mp.plot_raster(
        treecover_mismatch_low_res.data,
        extent=treecover_mismatch_low_res.get_extent(),
        **treecover_mismatch_kwargs,
        ax=ax,
    )

    cbar.set_label("Tree cover overshoot [-]")

    plt.savefig("figures/luh2_treecover_overshoot.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_uncertainty():
    settings = parameter_optimisation(0)
    settings_model_params = parameter_optimisation(1)

    settings_chelsa = copy.deepcopy(settings)
    settings_chelsa.climate_loader = "climate_chelsa"

    settings_wise = copy.deepcopy(settings)
    settings_wise.soil_loader = "soil_wise"

    tcc = tcc_current

    prt = load_data("potential_treecover", settings) / 10000
    prt_std = load_data("potential_treecover_std", settings) / 10000

    treecover_dr = load_data("treecover_disturbance_regime", settings)
    treecover_dr_std = load_data("disturbance_regime_std", settings)

    prt_rel_uncertainty = (prt_std / prt) ** 2
    dr_rel_uncertainty = (treecover_dr_std / (1 - treecover_dr)) ** 2
    tcc_std = (prt_rel_uncertainty + dr_rel_uncertainty) ** (1 / 2) * tcc

    tcc_model_params = load_data("tcc", settings_model_params) / 10000
    tcc_chelsa = load_data("tcc", settings_chelsa) / 10000
    tcc_wise = load_data("tcc", settings_wise) / 10000

    tcc_dif_model_params = tcc_model_params - tcc
    tcc_dif_model_uncertainty = tcc_std
    tcc_dif_chelsa = tcc_chelsa - tcc
    tcc_dif_wise = tcc_wise - tcc

    reprojection_kw = dict(dst_crs=ccrs.EckertIV(), resolution=(10000, 10000), njobs=10)
    tcc_dif_model_params_reprojected = tcc_dif_model_params.rio.reproject(
        **reprojection_kw
    )
    tcc_dif_model_uncertainty_reprojected = tcc_dif_model_uncertainty.rio.reproject(
        **reprojection_kw
    )
    tcc_dif_chelsa_reprojected = tcc_dif_chelsa.rio.reproject(**reprojection_kw)
    tcc_dif_wise_reprojected = tcc_dif_wise.rio.reproject(**reprojection_kw)

    # Uncertainty maps - panel

    bins = [-0.25, -0.2, -0.15, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
    plot_kw = dict(
        bins=bins,
        cmap="PRGn",
        legend=None,
        interpolation="nearest",
        extent=tcc_dif_wise_reprojected.get_extent(),
    )

    f, ax = plt.subplots(
        nrows=4, figsize=(10, 15), subplot_kw=dict(projection=ccrs.EckertIV())
    )
    ax = ax.flatten()
    for cax in ax:
        cax.set_global()
        cax.coastlines(linewidth=0.2)
        mp.add_gridlines(cax, 30, alpha=0.2)

    plt.tight_layout(h_pad=3)
    legend_ax = f.add_axes([0.88, 0.36, 0.02, 0.28])

    mp.plot_raster(tcc_dif_model_params_reprojected.values, ax=ax[0], **plot_kw)
    mp.plot_raster(tcc_dif_model_uncertainty_reprojected.values, ax=ax[1], **plot_kw)
    mp.plot_raster(tcc_dif_chelsa_reprojected.values, ax=ax[2], **plot_kw)
    mp.plot_raster(tcc_dif_wise_reprojected.values, ax=ax[3], **plot_kw)

    ax[0].set_title("Model parameter uncertainty")
    ax[1].set_title("Epistemic uncertainty")
    ax[2].set_title("Climate state uncertainty")
    ax[3].set_title("Soil characteristics uncertainty")

    ax[0].text(0, 0.9, "A", weight="bold", transform=ax[0].transAxes, fontsize=15)
    ax[1].text(0, 0.9, "B", weight="bold", transform=ax[1].transAxes, fontsize=15)
    ax[2].text(0, 0.9, "C", weight="bold", transform=ax[2].transAxes, fontsize=15)
    ax[3].text(0, 0.9, "D", weight="bold", transform=ax[3].transAxes, fontsize=15)

    cbar = plt.colorbar(mappable=ax[-1].images[-1], cax=legend_ax, extend="both")
    cbar.set_label(rf"{delta}TCC [-]", fontsize=16, labelpad=20)
    cbar.set_ticks(bins)

    plt.savefig("figures/uncertainty_maps_panel.png", bbox_inches="tight", dpi=300)
    plt.show()

    # Plot main contributor to uncertainty
    u_merged = np.abs(
        np.stack(
            (
                np.zeros_like(tcc_dif_model_params_reprojected.values),
                tcc_dif_model_params_reprojected.values,
                tcc_dif_model_uncertainty_reprojected.values,
                tcc_dif_chelsa_reprojected.values,
                tcc_dif_wise_reprojected.values,
            ),
            axis=2,
        )
    )
    u_merged[u_merged > 10e10] = np.nan

    max_val = u_merged.max(axis=2)
    u_merged[(u_merged == max_val[:, :, None]).sum(axis=2) > 1] = np.nan

    idxmax = u_merged.argmax(axis=2).astype(float)
    idxmax[np.isnan(u_merged[..., 1:]).sum(axis=2) > 0] = 0
    idxmax[idxmax == 0] = np.nan

    f, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.EckertIV()})
    mp.add_gridlines(ax=ax, lines=30)
    ax.coastlines(linewidth=0.2)

    colors = (
        hex2color("#D1DDDC"),
        hex2color("#C6AF99"),
        hex2color("#436354"),
        hex2color("#FAD061"),
    )

    mp.plot_classified_raster(
        idxmax,
        ax=ax,
        levels=[1, 2, 3, 4],
        colors=colors,
        labels=(
            "model parameters",
            "epistemic uncertainty",
            "climate data",
            "soil data",
        ),
        interpolation="nearest",
        extent=tcc_dif_model_params_reprojected.get_extent(),
        legend="legend",
        legend_kw={"bbox_to_anchor": (1.3, 1), "frameon": False},
    )

    inset_ax = f.add_axes([0.20, 0.43, 0.12, 0.08], transform=ax.transAxes)

    m = (idxmax >= 1).sum()
    for i in range(1, 5):
        inset_ax.bar(i, (idxmax == i).sum() / m, color=colors[i - 1])
    inset_ax.set_ylim(0, 0.7)
    inset_ax.set_xticks([])
    inset_ax.set_ylabel("Contribution [-]")

    for p in ax.get_legend().get_patches():
        p._linewidth = 0

    plt.savefig(
        "figures/main_contributor_uncertainty.png", bbox_inches="tight", dpi=300
    )
    plt.show()


def plot_africa_comparison():
    bounds = (
        -1947813.624531312,
        -4717931.944108543,
        5198186.375468688,
        5009068.055891457,
    )
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])

    climate_classes = load_data("climate_classes")

    treecover_africa = load_data("treecover_africa").rio.clip_box(*bounds) / 100
    treecover_clipped = treecover.rio.clip_box(*bounds)
    intact_areas_clipped = intact_areas.rio.clip_box(*bounds)
    tcc_clipped = tcc_current.rio.clip_box(*bounds)
    climate_classes_clipped = climate_classes.rio.clip_box(*bounds)

    intact_idx = label(
        fs.focal_max(intact_areas_clipped.values, window_size=15), connectivity=2
    )
    intact_idx[intact_areas_clipped.values == 0] = 1

    output = np.full_like(intact_idx, dtype=float, fill_value=np.nan)
    for clim_class in np.unique(climate_classes_clipped):
        print(f"{clim_class=}")
        if clim_class == 0:
            continue

        mask = np.logical_and(
            intact_areas_clipped.values, climate_classes_clipped.values == clim_class
        )

        clim_treecover_africa = np.nanmean(treecover_africa.values[mask])
        clim_treecover = np.nanmean(treecover_clipped.values[mask])

        output[climate_classes_clipped.values == clim_class] = (
            clim_treecover_africa - clim_treecover
        )

    output[np.isnan(treecover_africa)] = np.nan
    output[np.isnan(treecover_clipped)] = np.nan

    bins = [
        -0.5,
        -0.25,
        -0.2,
        -0.15,
        -0.1,
        -0.05,
        -0.02,
        0.02,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.5,
    ]

    ax = mp.basemap(projection=ccrs.EckertIV())
    ax.set_extent(extent, crs=ccrs.EckertIV())
    ax.coastlines(linewidth=0.2)
    mp.add_gridlines(ax=ax, lines=30)
    _, cbar = mp.plot_raster(
        output,
        bins=bins,
        interpolation="nearest",
        cmap="RdYlGn",
        ax=ax,
        extent=extent,
        legend_kw={"shrink": 0.8, "aspect": 18, "pad_fraction": 1.4},
    )
    cbar.set_label(f"{delta}TCC [-]", labelpad=10)
    cbar.set_ticks(bins)
    plt.savefig("figures/uncertainty_africa.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    plot_result_maps()
    plot_future_forest_loss()
    plot_luh2_mismatch()
    plot_bastin_comparison_histogram()
    plot_reforestation_potential()
    plot_bastin_comparison_panel(
        (-650_000, 5_000_000, 1_850_000, 7_500_000), "europe", h_pad=-12, w_pad=6
    )
    plot_bastin_comparison_panel(
        (0, -3_500_000, 4_800_000, 1_000_000), "africa", h_pad=-18, w_pad=4
    )
    plot_bastin_comparison_panel(
        (9_000_000, -5_500_000, 15_500_000, 1_000_000), "australia", h_pad=-12, w_pad=6
    )
    plot_bastin_comparison_panel(
        (-7_800_000, -2_800_000, -4_000_000, 1_000_000),
        "south_america",
        h_pad=-12,
        w_pad=6,
    )


if __name__ == "__main__":
    main()
