from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np


class DiabetesPreprocessor:
    """
    Builds a preprocessing pipeline for the Biofusion Hybrid
    Diabetes Classification project.
    """

    def __init__(
        self,
        numerical_features,
        skewed_numerical_features,
        categorical_features
    ):
        self.numerical_features = numerical_features
        self.skewed_numerical_features = skewed_numerical_features
        self.categorical_features = categorical_features

    @staticmethod
    def log_transform(x):
        return np.log1p(x)

    def build(self):
        """
        Create and return the preprocessing ColumnTransformer.
        """

        # Numerical pipeline (normal)
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Skewed numerical pipeline
        skewed_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log_transform", FunctionTransformer(self.log_transform)),
            ("scaler", StandardScaler())
        ])

        # Categorical pipeline
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Full preprocessor
        preprocessor = ColumnTransformer(transformers=[
            ("num", num_pipeline, self.numerical_features),
            ("skewed_num", skewed_pipeline, self.skewed_numerical_features),
            ("cat", cat_pipeline, self.categorical_features)
        ])

        return preprocessor
