import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from preprocessor import DiabetesPreprocessor


class DiabetesTrainer:
    """
    Trainer class for the Biofusion Hybrid Diabetes
    classification model.
    """

    def __init__(
        self,
        train_path,
        test_path,
        target_column,
        numerical_features,
        skewed_numerical_features,
        categorical_features,
        random_state=42
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.target_column = target_column
        self.numerical_features = numerical_features
        self.skewed_numerical_features = skewed_numerical_features
        self.categorical_features = categorical_features
        self.random_state = random_state

    def load_train_data(self):
        return pd.read_csv(self.train_path)

    def load_test_data(self):
        return pd.read_csv(self.test_path)

    def train(self, model_output_path):
        # Load train data
        data = self.load_train_data()

        # Split features / target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        # Build preprocessor
        preprocessor = DiabetesPreprocessor(
            numerical_features=self.numerical_features,
            skewed_numerical_features=self.skewed_numerical_features,
            categorical_features=self.categorical_features
        ).build()

        # Final tuned SVC
        svc = SVC(
            C=0.5,
            gamma="scale",
            probability=True,
            random_state=self.random_state
        )

        # Full pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", svc)
        ])

        # Train model
        pipeline.fit(X, y)

        # Save model
        joblib.dump(pipeline, model_output_path)

        return pipeline

    def evaluate(self, model_path):
        # Load model
        model = joblib.load(model_path)

        # Load test data
        test_data = self.load_test_data()

        X_test = test_data.drop(columns=[self.target_column])
        y_test = test_data[self.target_column]

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return {
            "f1_score_weighted": f1,
            "classification_report": report,
            "confusion_matrix": cm
        }
