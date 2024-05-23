import numpy as np
from focal_stats import focal_std
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.climate_params import define_future_params, FutureClimateParams
from src.data import load_data
from src.data_preprocessing import define_clip_mask
from src.misc import template_array
from src.parameter_optimisation import parameter_optimisation
from src.settings import Settings


climate_params_245 = define_future_params(scenarios='ssp245', periods='2081-2100', verify_file_integrity=False)

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

print("Loading tcc")
tcc_current = load_data("tcc", settings_current) / 10000

tcc_bastin = load_data("potential_treecover_bastin") / 100
tcc_ssp245_full = load_data("tcc", settings_245).sel(scenario='ssp245', period='2081-2100') / 10000
tcc_ssp245 = tcc_ssp245_full.mean(dim=['model'])

tcc_leave_out_regions = load_data("tcc", settings_current_leave_out_regions) / 10000
leave_out_regions = define_clip_mask()

print("Loading landscape")
landcover = load_data("landcover")
topography = load_data("topography")
elevation = topography.sel(variable='elevation')
slope = topography.sel(variable='slope')
intact_areas = load_data("intact_landscapes")

treecover.data[np.isnan(tcc_current.data)] = np.nan
elevation.data[np.isnan(tcc_current.data)] = np.nan
slope.data[np.isnan(tcc_current.data)] = np.nan

print("Treecover std")
treecover_std = template_array()
treecover_std.data[:] = focal_std(treecover.data, window_size=5)

print("TCC std")
tcc_std = template_array()
tcc_std.data[:] = focal_std(tcc_current.data, window_size=5)

print("TCC Bastin std")
tcc_bastin_std = template_array()
tcc_bastin_std.data[:] = focal_std(tcc_bastin.data, window_size=5)

print("Elevation std")
elevation_std = template_array()
elevation_std.data[:] = focal_std(elevation.data, window_size=5)

print("Load LUH2 ssp245")
luh2_treecover = load_data("luh2", Settings(
    climate_params=FutureClimateParams(model='suppress_validation', scenario='ssp245', period='2081-2100')
)).sel(period='2081-2100', scenario='ssp245')

climate_classes = load_data("climate_classes")
climate_expected_treecover = template_array()
for m in range(1, 31):
    mask = np.logical_and.reduce([climate_classes == m, intact_areas, ~np.isnan(treecover)])
    v = treecover.data[mask].mean()
    climate_expected_treecover.data[climate_classes == m] = v
    print(m, v)
(climate_expected_treecover * 10000).rio.to_raster("data_processed/climate_expected_treecover.tif", compress="LZW")

intact_tcc_mask = np.logical_and.reduce([~np.isnan(treecover.data), ~np.isnan(tcc_current.data), intact_areas])
intact_tcc_bastin_mask = np.logical_and.reduce([~np.isnan(treecover.data), ~np.isnan(tcc_bastin.data), intact_areas])
leave_out_regions_mask = np.logical_and(leave_out_regions.data == 1, ~np.isnan(tcc_current.data))
leave_out_regions_intact_mask = np.logical_and(leave_out_regions.data == 1, intact_tcc_mask)
leave_out_regions_intact_bastin_mask = np.logical_and(leave_out_regions.data == 1, intact_tcc_bastin_mask)

# stats
print("Compared to Bastin")
print(f"r2: {r2_score(treecover.data[intact_tcc_mask], tcc_current.data[intact_tcc_mask]):.3f} "
      f"{r2_score(treecover.data[intact_tcc_bastin_mask], tcc_bastin.data[intact_tcc_bastin_mask]):.3f}")
print(f"MAE: {mean_absolute_error(treecover.data[intact_tcc_mask], tcc_current.data[intact_tcc_mask]):.3f} "
      f"{mean_absolute_error(treecover.data[intact_tcc_bastin_mask], tcc_bastin.data[intact_tcc_bastin_mask]):.3f}")
print("r2 in intact areas in the left out regions")
print(
    f"r2: {r2_score(treecover.data[leave_out_regions_intact_mask], tcc_current.data[leave_out_regions_intact_mask]):.3f}")
print(
    f"MAE: {mean_absolute_error(treecover.data[leave_out_regions_intact_mask], tcc_current.data[leave_out_regions_intact_mask]):.3f}")


r2_score(tcc_current.data[leave_out_regions_mask], tcc_leave_out_regions.data[leave_out_regions_mask])

mask_tcc = np.logical_and.reduce((~np.isnan(tcc_current), ~np.isnan(climate_expected_treecover), ~np.isnan(tcc_bastin)))
r2_score(climate_expected_treecover.data[mask_tcc], tcc_current.data[mask_tcc])
r2_score(climate_expected_treecover.data[mask_tcc], tcc_bastin.data[mask_tcc])

print("Reforestation potential")
mask = np.logical_and(landcover == 20, treecover == 0)
treecover.where(mask).sum()
tcc_current.where(mask).sum()
