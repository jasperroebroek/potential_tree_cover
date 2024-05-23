import copy

from src.model_treecover import model_potential_treecover
from src.model_treecover_disturbance import model_treecover_disturbance_regime
from src.parameter_optimisation import parameter_optimisation

# MODELLING
# parameter optimisation
print("Parameter optimisation")
settings = parameter_optimisation()

# CHELSA climate
print("Chelsa")
settings_chelsa = copy.deepcopy(settings)
settings_chelsa.climate_loader = 'climate_chelsa'
model_potential_treecover(settings_chelsa, climate_params=None)
model_treecover_disturbance_regime(settings_chelsa, climate_params=None)

# WISE30SEC soil
print("Wise")
settings_wise = copy.deepcopy(settings)
settings_wise.soil_loader = 'soil_wise'
model_potential_treecover(settings_wise, climate_params=None)
model_treecover_disturbance_regime(settings_wise, climate_params=None)

# Parameter uncertainty
print("Parameter uncertainty")
settings_parameter_uncertainty = parameter_optimisation(1)
model_potential_treecover(settings_parameter_uncertainty, climate_params=None)
model_treecover_disturbance_regime(settings_parameter_uncertainty, climate_params=None)

# current climate models
print("Current climate models")
model_potential_treecover(settings, climate_params=None)
model_treecover_disturbance_regime(settings, climate_params=None)

# future climate models
print("Future")
model_potential_treecover(settings)
model_treecover_disturbance_regime(settings)
