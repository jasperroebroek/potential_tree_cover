import gc
from typing import List

import pandas as pd

from src.data import PotentialTreeCoverLoader, PotentialTreeCoverStdLoader, load_data, TreecoverDisturbanceRegimeLoader, \
    TreecoverDisturbanceRegimeStdLoader
from src.model_treecover import model_potential_treecover, score_table_path
from src.model_treecover_disturbance import model_treecover_disturbance_regime
from src.scorer import Scorer
from src.settings import Settings
from src.climate_params import CurrentClimateParams


def read_score_table() -> List[Settings]:
    processed_settings = []

    if score_table_path.exists():
        for i, row in pd.read_csv(score_table_path).iterrows():
            processed_settings.append(convert_df_row_to_settings(row))

    return processed_settings


def update_score_table(settings: Settings, scorer: Scorer):
    gc.collect()

    processed_settings = read_score_table()

    if settings in processed_settings:
        return

    pt = load_data("potential_treecover", settings)
    tcc = load_data("tcc", settings)

    pd.DataFrame(index=[0], data={
        'weight_mid': settings.weight_mid,
        'weight_high': settings.weight_high,
        'weight_per_leaf': settings.weight_per_leaf,
        'score_classes': scorer.score(pt, bins=[0, 1500, 10001]),
        'score_classes_2': scorer.score(pt, bins=[0, 1500, 7000, 10001]),
        'score_pinball_1': scorer.score_pinball(pt, alpha=1),
        'score_pinball_099': scorer.score_pinball(pt, alpha=0.99),
        'score_r2': scorer.score_r2(tcc),
        'score_rmse': scorer.score_rmse(tcc),
        'score_tcc_classes': scorer.score(tcc, bins=[0, 1500, 7000, 10001])
    }).to_csv(
        score_table_path, mode='a', header=not score_table_path.exists(), index=False
    )


def convert_df_row_to_settings(r: pd.Series) -> Settings:
    return Settings(
        weight_mid=int(r.weight_mid),
        weight_high=int(r.weight_high),
        weight_per_leaf=int(r.weight_per_leaf)
    )


def parameter_optimisation(n: int = 0, verbose: bool = False) -> Settings:
    if verbose:
        print("Finding optimal parameters potential treecover model")

    processed_settings = read_score_table()
    for settings in processed_settings:
        settings.leave_out_regions = True

    scorer = None

    c = 0
    for weight_mid in [10, 20, 40, 100, 200]:
        for weight_high in [50, 100, 200, 500, 1000]:
            for weight_per_leaf in [500, 1000, 2500, 5000][::-1]:
                if weight_per_leaf / weight_high < 5:
                    continue
                if weight_per_leaf / weight_mid < 12.5:
                    continue
                if weight_high <= weight_mid:
                    continue
                c = c + 1

                current_settings = Settings(weight_mid=weight_mid,
                                            weight_high=weight_high,
                                            weight_per_leaf=weight_per_leaf,
                                            leave_out_regions=True,
                                            climate_params=CurrentClimateParams())

                if verbose:
                    print(f"Model {c} -> {current_settings}")

                if current_settings in processed_settings:
                    continue

                if scorer is None:
                    scorer = Scorer()

                if not (
                    PotentialTreeCoverLoader().is_processed(current_settings) and
                    PotentialTreeCoverStdLoader().is_processed(current_settings)
                ):
                    gc.collect()
                    model_potential_treecover(current_settings, climate_params=None)

                if not (
                    TreecoverDisturbanceRegimeLoader().is_processed(current_settings) and
                    TreecoverDisturbanceRegimeStdLoader().is_processed(current_settings)
                ):
                    gc.collect()
                    model_treecover_disturbance_regime(current_settings, climate_params=None)

                update_score_table(current_settings, scorer)

    df = pd.read_csv(score_table_path)
    row = df.sort_values('score_rmse').iloc[n]
    return convert_df_row_to_settings(row)
