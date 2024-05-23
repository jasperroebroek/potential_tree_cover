from dataclasses import dataclass, field
from pathlib import Path

from src.climate_params import ClimateParams, CurrentClimateParams


@dataclass
class Settings:
    weight_mid: int = None
    weight_high: int = None
    weight_per_leaf: int = None
    input_reduction_factor: int = 5
    leave_out_regions: bool = False
    climate_params: ClimateParams = field(default_factory=CurrentClimateParams)
    climate_loader: str = 'climate'
    soil_loader: str = 'soil'

    def __post_init__(self):
        if self.climate_loader not in ("climate", "climate_chelsa"):
            raise NotImplementedError
        if self.soil_loader not in ("soil", "soil_wise"):
            raise NotImplementedError

    def model_to_str(self, include_leave_out_regions: bool = False):
        parts = [str(self.weight_mid), str(self.weight_high), str(self.weight_per_leaf)]
        if self.input_reduction_factor != 5:
            parts.append(str(self.input_reduction_factor))
        if self.leave_out_regions and include_leave_out_regions:
            parts.append('leave-out-regions')
        return '_'.join(parts)

    def climate_to_str(self) -> str:
        if self.climate_loader == 'climate_chelsa':
            return f'chelsa_{self.climate_params.to_str()}'

        return self.climate_params.to_str()


def construct_path(settings: Settings, layer: str = 'root') -> Path:
    from src.paths import path_data_processed

    layer_mapping = [
        ('root', path_data_processed),
        (
            'clipping',
            'leave_out_regions' if settings.leave_out_regions is True else 'global',
        ),
        ('climate', settings.climate_to_str())
    ]

    if settings.soil_loader != 'soil':
        layer_mapping.append(('soil', settings.soil_loader))

    layer_mapping.append(('model', f'model_{settings.model_to_str()}'))

    layers = [layer_mapping[i][0] for i in range(len(layer_mapping))]
    full_path = [layer_mapping[i][1] for i in range(len(layer_mapping))]

    if layer not in layers:
        raise NotImplementedError(
            f'variable {layer} not in possible variables: {layers}'
        )

    return Path(*full_path[: layers.index(layer) + 1])


def path_from_settings(settings: Settings, variable: str = 'root') -> Path:
    path = construct_path(settings, variable)
    path.mkdir(parents=True, exist_ok=True)
    return path
