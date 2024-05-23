from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional

import rioxarray as rxr
from rasterio import RasterioIOError

ClimateModelDefinition = namedtuple('Definition', 'version resolution var model scenario period')


def parse_files() -> set[ClimateModelDefinition]:
    from src.data import define_future_climate_files
    definitions = set()
    for f in define_future_climate_files():
        fname = f.stem
        version, resolution, var, model, scenario, period = fname.split("_")
        definitions.add(ClimateModelDefinition(version, resolution, var, model, scenario, period))

    return definitions


def define_models() -> set[str]:
    definitions = parse_files()
    return set(d.model for d in definitions)


def define_scenarios() -> set[str]:
    definitions = parse_files()
    return set(d.scenario for d in definitions)


def define_periods() -> set[str]:
    definitions = parse_files()
    return set(d.period for d in definitions)


class ClimateParams:
    @abstractmethod
    def check_validity(self, file_integrity: bool = True) -> bool:
        pass

    @abstractmethod
    def equals(self, other) -> bool:
        pass

    @abstractmethod
    def to_str(self) -> str:
        pass


@dataclass
class FutureClimateParams(ClimateParams):
    model: str
    scenario: str
    period: str = "2081-2100"

    def check_validity(self, file_integrity: bool = True) -> bool:
        if self.model == 'suppress_validation':
            return True
        from src.data import define_future_climate_files
        filename = f"wc2.1_30s_bioc_{self.to_str()}.tif"
        file_exists = False
        file_path: Optional[Path] = None

        for path in define_future_climate_files():
            if filename in str(path):
                file_exists = True
                file_path = path

        if not file_exists:
            return False

        if file_integrity:
            try:
                rxr.open_rasterio(file_path, cache=False)
            except RasterioIOError:
                print(f"Could not open file: {file_path}")
                return False

        return True

    def equals(self, other: ClimateParams) -> bool:
        if not isinstance(other, FutureClimateParams):
            return False

        if self.model != other.model:
            return False

        if self.scenario != other.scenario:
            return False

        if self.period != other.period:
            return False

        return True

    def to_str(self) -> str:
        return f"{self.model}_{self.scenario}_{self.period}"


@dataclass
class CurrentClimateParams(ClimateParams):
    def check_validity(self, file_integrity: bool = True) -> bool:
        return True

    def equals(self, other: ClimateParams) -> bool:
        if not isinstance(other, CurrentClimateParams):
            return False

        return True

    def to_str(self) -> str:
        return "current_climate"


def define_current_params() -> list[CurrentClimateParams]:
    return [CurrentClimateParams()]


def define_future_params(models: Union[str, List[str]] = "*",
                         scenarios: Union[str, List[str]] = "*",
                         periods: Union[str, List[str]] = "*",
                         verify_file_integrity: bool = True) -> List[FutureClimateParams]:
    """input of models, scenarios and periods in str or list form. * indicates all options"""
    if models == "*":
        selected_models = define_models()
    elif isinstance(models, str):
        selected_models = [models],
    elif isinstance(models, list):
        selected_models = models
    else:
        raise ValueError

    if scenarios == "*":
        selected_scenarios = define_scenarios()
    elif isinstance(scenarios, str):
        selected_scenarios = [scenarios]
    elif isinstance(scenarios, list):
        selected_scenarios = scenarios
    else:
        raise ValueError

    if periods == "*":
        # selected_periods = define_periods()
        selected_periods = ['2041-2060', '2081-2100']
    elif isinstance(periods, str):
        selected_periods = [periods]
    elif isinstance(periods, list):
        selected_periods = periods
    else:
        raise ValueError

    params_list = [FutureClimateParams(model=model, scenario=scenario, period=period)
                   for model in selected_models
                   for scenario in selected_scenarios
                   for period in selected_periods]

    return list(filter(lambda x: x.check_validity(verify_file_integrity), params_list))


def define_all_params() -> List[ClimateParams]:
    return define_current_params() + define_future_params()


def missing_future_params() -> None:
    models = define_models()
    scenarios = define_scenarios()
    periods = define_periods()

    for model in models:
        for scenario in scenarios:
            for period in periods:
                params = FutureClimateParams(model=model, scenario=scenario, period=period)
                if not params.check_validity():
                    print(params)
