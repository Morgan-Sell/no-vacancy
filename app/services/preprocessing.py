import re
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from app.services.config_services import (
    BOOKING_MAP,
    MONTH_ABBREVIATION_MAP,
    VARIABLE_RENAME_MAP,
)


class NoVacancyDataProcessing(BaseEstimator, TransformerMixin):
    """
    A custom transformer for processing NoVacancy dataset.

    This class implements the `TransformerMixin` to preprocess and transform the NoVacancy dataset for machine learning tasks.

    Attributes:
        variable_rename (Dict[str, str]): Mapping of column names to their new names.
        month_abbreviation (Dict[int, str]): Mapping of month numbers to their abbreviated names.
        booking_map (Dict[str, int]): Mapping for target variable transformation (e.g., cancellation encoding).
        vars_to_drop (List[str]): List of column names to be removed during transformation.

    Methods:
        fit(X, y): Fits the transformer. Returns self without modifications.
        transform(X, y): Transforms the input DataFrame `X` and target `y` by applying standardization, feature engineering, and column removal.
        _to_snake_case(name): Converts a string to snake_case.
        _convert_columns_to_snake_case(df): Converts all column names in a DataFrame to snake_case.

    """

    def __init__(
        self,
        variable_rename: Dict[str, str],
        month_abbreviation: Dict[int, str],
        vars_to_drop: List[str],
        booking_map: Dict[str, int],
    ):
        super().__init__()

        self.variable_rename = variable_rename
        self.month_abbreviation = month_abbreviation
        self.vars_to_drop = vars_to_drop
        self.booking_map = booking_map

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.Series = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        X_tr = X.copy()
        y_tr = y.copy() if y is not None else None

        # Standardize column name text
        X_tr = self._convert_columns_to_snake_case(X_tr)

        # Extract month and day of week features from "date of reservation"
        X_tr["date_of_reservation"] = pd.to_datetime(
            X_tr["date_of_reservation"], format="%m/%d/%Y", errors="coerce"
        ).dt.date
        X_tr["month_of_reservation"] = pd.to_datetime(
            X_tr["date_of_reservation"], errors="coerce"
        ).dt.month
        X_tr["month_of_reservation"] = X_tr["month_of_reservation"].map(
            self.month_abbreviation
        )
        X_tr["day_of_week"] = pd.to_datetime(X_tr["date_of_reservation"]).dt.day_name()

        # Remove selected features
        X_tr.drop(self.vars_to_drop, axis=1, inplace=True)

        # Make select column names more intuitive
        X_tr.rename(columns=self.variable_rename, inplace=True)

        # Transform the target variable
        if y_tr is not None and self.booking_map is not None:
            y_tr = y_tr.map(self.booking_map)

        return X_tr, y_tr

    def _to_snake_case(self, name: str) -> str:
        # Replace hyphens (-) with underscores (_)
        name = name.replace("-", "_")

        # Preserve existing underscores and replace spaces or special characters with underscores
        name = re.sub(
            r"[^\w\s]", "", name
        )  # Remove special characters except underscores
        name = re.sub(
            r"(?<=[a-z])(?=[A-Z])", "_", name
        )  # Handle camelCase to snake_case
        name = re.sub(r"[\s]+", "_", name)  # Replace spaces with underscores
        return name.lower()

    def _convert_columns_to_snake_case(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform column names
        new_columns = [
            self._to_snake_case(col).replace("__", "_") for col in df.columns
        ]
        df.columns = new_columns
        return df
