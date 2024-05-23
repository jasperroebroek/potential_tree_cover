import copy
import gc
from typing import Union, Callable, NewType, Optional, List, Protocol

import numpy as np
from joblib import Memory

from src.climate_params import (
    FutureClimateParams,
    CurrentClimateParams,
    define_future_params,
)
from src.data import LoaderStrategySingle
from src.data_preprocessing import load_future_feature_data
from src.misc import template_array, store_data
from src.regression_model import ExtendedRandomForestMaximumRegressor
from src.settings import Settings

location = './cachedir'
memory = Memory(location, verbose=True)


class ExtendedRegressor(Protocol):
    def predict_mean(self, X):
        ...

    def predict_std(self, X):
        ...


TestDataLoadFun = NewType('TestDataLoader', Callable[[Settings], List[np.ndarray]])
TrainFun = NewType('TrainFun', Callable[[Settings], ExtendedRegressor])


def predict_to_output(
    rf: ExtendedRegressor,
    X_test: np.ndarray,
    mask: np.ndarray,
    fun: str,
    strategy: LoaderStrategySingle,
    settings: Settings,
) -> None:
    print('= Predicting')
    predict_function = f'predict_{fun}'

    result = template_array()
    result.data[mask] = getattr(rf, predict_function)(X_test)

    path = strategy.get_processed_path(settings)
    print('= Storing data')
    store_data(result, path=path)


def predict_current(
    rf: ExtendedRegressor,
    settings: Settings,
    load_current_train_test_data_fun: TestDataLoadFun,
    mean_strategy: LoaderStrategySingle,
    std_strategy: LoaderStrategySingle,
) -> None:
    if mean_strategy.is_processed(settings) and std_strategy.is_processed(settings):
        return

    print('Predicting current climate')

    (
        X_train,
        y_train,
        train_weights,
        train_mask,
        X_test,
        y_test,
        test_mask,
    ) = load_current_train_test_data_fun(settings)

    if not mean_strategy.is_processed(settings):
        predict_to_output(
            rf,
            X_test=X_test,
            mask=test_mask,
            fun='mean',
            strategy=mean_strategy,
            settings=settings,
        )

    if not std_strategy.is_processed(settings):
        predict_to_output(
            rf,
            X_test=X_test,
            mask=test_mask,
            fun='std',
            strategy=std_strategy,
            settings=settings,
        )


def predict_future(
    rf: ExtendedRandomForestMaximumRegressor,
    settings: Settings,
    mean_strategy: LoaderStrategySingle,
    std_strategy: LoaderStrategySingle,
) -> None:
    if mean_strategy.is_processed(settings) and std_strategy.is_processed(settings):
        return

    gc.collect()
    print(settings.climate_params)
    print('Predict future')

    X_test, updated_mask = load_future_feature_data(settings)
    print('- Prediction')

    if not mean_strategy.is_processed(settings):
        predict_to_output(
            rf,
            X_test=X_test,
            mask=updated_mask,
            fun='mean',
            strategy=mean_strategy,
            settings=settings,
        )

    if not std_strategy.is_processed(settings):
        predict_to_output(
            rf,
            X_test=X_test,
            mask=updated_mask,
            fun='std',
            strategy=std_strategy,
            settings=settings,
        )


def model_template(
    train_fun: TrainFun,
    load_current_train_test_data_fun: TestDataLoadFun,
    settings: Settings,
    mean_strategy: LoaderStrategySingle,
    std_strategy: LoaderStrategySingle,
    climate_params: Optional[
        Union[FutureClimateParams, List[FutureClimateParams], str]
    ] = '*',
) -> None:
    """
    Params
    ------
    train_fun:
        Function that returns a trained machine learning model
    load_current_train_test_data_fun:
        Function that loads the training and testing data for training the model and test the data un current climate
    settings:
        model settings object.
    mean_strategy, std_strategy:
        Data Loader Strategy to retrieve paths
    climate_params:
        future climate parameters for specific runs. Default is "*", which runs all scenarios. None runs none
    """
    if climate_params is None:
        selected_params = []
    elif climate_params == '*':
        selected_params = define_future_params()
    elif isinstance(climate_params, FutureClimateParams):
        selected_params = [climate_params]
    else:
        for p in climate_params:
            if isinstance(p, CurrentClimateParams):
                raise TypeError
        selected_params = climate_params

    print(f"Modelling {mean_strategy.name.replace('_', ' ')}")
    print('-------------------------------------------------------------------')

    rf = train_fun(settings)
    predict_current(
        rf,
        settings=settings,
        load_current_train_test_data_fun=load_current_train_test_data_fun,
        mean_strategy=mean_strategy,
        std_strategy=std_strategy,
    )

    if len(selected_params) == 0:
        return

    filtered_params = list(filter(lambda x: x.check_validity(), selected_params))
    if len(filtered_params) == 0:
        print('- No matching scenarios')
        return

    print(f'modelling future {len(filtered_params)} parameter sets')
    for p in filtered_params:
        future_settings = copy.deepcopy(settings)
        future_settings.climate_params = p
        predict_future(
            rf,
            settings=future_settings,
            mean_strategy=mean_strategy,
            std_strategy=std_strategy,
        )
