"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary

ADULT_AGE_THRESHOLD = 30

TOP_FARES_COUNT = 5

def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)
    missing_by_column = {str(column): int(count) for column, count in df.isna().sum().items()}
    adults_over_30_count = int((df["Age"] > ADULT_AGE_THRESHOLD).sum())
    mean_age_by_pclass = {
        int(pclass): float(value)
        for pclass, value in df.groupby("Pclass")["Age"].mean().items()
    }
    survival_rate_by_pclass = {
        int(pclass): float(value)
        for pclass, value in df.groupby("Pclass")["Survived"].mean().items()
    }
    highest_fares = [float(fare) for fare in df["Fare"].nlargest(TOP_FARES_COUNT)]
    return TitanicSummary(
        row_count=int(len(df)),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )

input_data = TitanicInput(csv_path="titanic.csv")
result = analyze_titanic(input_data)
print(result)
