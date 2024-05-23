from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from joblib import Memory

from src.climate_params import FutureClimateParams
from src.data import PotentialTreeCoverLoader, PotentialTreeCoverStdLoader, load_data
from src.data_preprocessing import (
    load_parse_landcover,
    define_sampling,
    define_clip_mask,
)
from src.model_template import model_template
from src.regression_model import ExtendedRandomForestMaximumRegressor
from src.settings import Settings

location = './cachedir'
memory = Memory(location, verbose=True)
score_table_path = Path('data_processed', 'score.csv')


def load_current_data_treecover_model(settings: Settings) -> List[np.ndarray]:
    """:returns
    X_train, y_train, train_weights, train_mask, X_test, y_test, test_mask
    """
    print('Loading current data')
    treecover, landcover = load_parse_landcover()
    sample_mask, sampled_treecover, sample_weights = define_sampling(
        settings=settings, landcover=landcover, target=treecover, name='treecover'
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

    train_mask_parts = [feature_mask, sample_mask]

    if settings.leave_out_regions:
        clip_mask = define_clip_mask().values == 0
        train_mask_parts.append(clip_mask)

    train_mask = np.logical_and.reduce(train_mask_parts)
    test_mask = feature_mask

    return [
        features[:, train_mask].T,
        sampled_treecover[train_mask],
        sample_weights[train_mask],
        train_mask,
        features[:, test_mask].T,
        sampled_treecover[test_mask],
        test_mask,
    ]


@memory.cache
def train_current_model(settings: Settings) -> ExtendedRandomForestMaximumRegressor:
    (
        X_train,
        y_train,
        train_weights,
        train_mask,
        X_test,
        y_test,
        test_mask,
    ) = load_current_data_treecover_model(settings)

    print('Training model: potential treecover')

    min_weight_fraction_leaf = settings.weight_per_leaf / np.nansum(train_weights)
    rf = ExtendedRandomForestMaximumRegressor(
        n_estimators=25,
        n_jobs=25,
        random_state=0,
        verbose=True,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
    )
    rf.fit(X_train, y_train, sample_weight=train_weights)

    return rf


def model_potential_treecover(
    settings: Settings,
    climate_params: Optional[
        Union[FutureClimateParams, List[FutureClimateParams], str]
    ] = '*',
) -> None:
    return model_template(
        train_fun=train_current_model,
        load_current_train_test_data_fun=load_current_data_treecover_model,
        settings=settings,
        mean_strategy=PotentialTreeCoverLoader(),
        std_strategy=PotentialTreeCoverStdLoader(),
        climate_params=climate_params,
    )
