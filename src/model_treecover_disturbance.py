from typing import List, Optional, Union

import numpy as np
from joblib import Memory

from src.climate_params import FutureClimateParams
from src.data import (
    TreecoverDisturbanceRegimeLoader,
    TreecoverDisturbanceRegimeStdLoader,
    load_data,
)
from src.data_preprocessing import (
    load_parse_landcover,
    define_sampling,
    define_clip_mask,
)
from src.model_template import model_template
from src.regression_model import ExtendedRandomForestRegressor
from src.settings import Settings

location = './cachedir'
memory = Memory(location, verbose=True)


def load_current_data_treecover_disturbance_model(
    settings: Settings,
) -> List[np.ndarray]:
    """:returns
    X_train, y_train, train_weights, train_mask, X_test, y_test, test_mask
    """
    if isinstance(settings.climate_params, FutureClimateParams):
        raise TypeError('Training can only be done on CurrentClimateParams')

    print('Loading current data')
    potential_treecover = load_data('potential_treecover', settings)
    treecover, landcover = load_parse_landcover()
    intact_landscapes = load_data('intact_landscapes')

    dr_values = 1 - (treecover / potential_treecover)
    dr_values.data[dr_values.values > 1] = np.nan
    dr_values.data[dr_values.values < 0] = np.nan

    sample_mask, sampled_treecover, sample_weights = define_sampling(
        settings=settings, landcover=landcover, target=treecover, name='treecover_dr'
    )

    print('- Loading features')
    features_raster = load_data(
        'X',
        Settings(
            climate_loader=settings.climate_loader, soil_loader=settings.soil_loader
        ),
    )
    features = features_raster.values

    feature_mask = np.isnan(features.sum(axis=0)) == 0

    train_mask_parts = [
        feature_mask,
        intact_landscapes.values,
        ~np.isnan(dr_values.values),
        ~np.isnan(sample_weights),
        treecover > 1500,
        landcover > 100,
        landcover < 200,
    ]

    if settings.leave_out_regions:
        clip_mask = define_clip_mask().values == 0
        train_mask_parts.append(clip_mask)

    train_mask = np.logical_and.reduce(train_mask_parts)
    test_mask = feature_mask

    return [
        features[:, train_mask].T,
        dr_values.values[train_mask],
        sample_weights[train_mask],
        train_mask,
        features[:, test_mask].T,
        dr_values.values[test_mask],
        test_mask,
    ]


@memory.cache
def train_current_model_treecover_dr(
    settings: Settings,
) -> ExtendedRandomForestRegressor:
    (
        X_train,
        y_train,
        train_weights,
        train_mask,
        X_test,
        y_test,
        test_mask,
    ) = load_current_data_treecover_disturbance_model(settings)

    min_samples_leaf = settings.weight_per_leaf * (settings.input_reduction_factor**2)

    print('Training model: treecover disturbance regime')
    rf = ExtendedRandomForestRegressor(
        n_estimators=25,
        n_jobs=25,
        random_state=0,
        verbose=True,
        min_samples_leaf=min_samples_leaf,
    )
    rf.fit(X_train, y_train)

    return rf


def model_treecover_disturbance_regime(
    settings: Settings,
    climate_params: Optional[
        Union[FutureClimateParams, List[FutureClimateParams], str]
    ] = '*',
) -> None:
    return model_template(
        train_fun=train_current_model_treecover_dr,
        load_current_train_test_data_fun=load_current_data_treecover_disturbance_model,
        settings=settings,
        mean_strategy=TreecoverDisturbanceRegimeLoader(),
        std_strategy=TreecoverDisturbanceRegimeStdLoader(),
        climate_params=climate_params,
    )
