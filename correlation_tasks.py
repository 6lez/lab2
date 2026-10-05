"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

MRI_COLUMN = "MRI_Count"
FEATURES = ("FSIQ", "VIQ", "PIQ", "Weight", "Height")
MISSING_MARKERS = ["NA", "N/A", "nan", "null", "?", "."]

def _select_group(df: pd.DataFrame, first_letter: str) -> pd.DataFrame:
    gender = df["Gender"].astype("string").str.strip().str.lower()
    return df[gender.str.startswith(first_letter)]

def _correlation_with_mri(df: pd.DataFrame, feature: str) -> float:
    pairs = df[[feature, MRI_COLUMN]].dropna()
    if len(pairs) < 2:
        return float("nan")
    return float(pairs[feature].corr(pairs[MRI_COLUMN], method="pearson"))

def _correlations_for_group(df: pd.DataFrame) -> dict[str, float]:
    return {feature: _correlation_with_mri(df, feature) for feature in FEATURES}

def _abs_or_neg_inf(value: float) -> float:
    return abs(value) if pd.notna(value) else float("-inf")

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=MISSING_MARKERS)

    men = _select_group(df, "m")
    women = _select_group(df, "f")

    men_correlation = _correlations_for_group(men)
    women_correlation = _correlations_for_group(women)

    strongest_mri_feature = max(
        FEATURES,
        key=lambda feature: max(
            _abs_or_neg_inf(women_correlation[feature]),
            _abs_or_neg_inf(men_correlation[feature]),
        ),
    )

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_correlation,
        men_mri_correlation=men_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )
input_data = BrainDataInput(csv_path="brainsize.txt")
result = analyze_brain_correlations(input_data)
print(result)
