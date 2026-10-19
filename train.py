
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def main():
    # Load the Iris dataset
    dataset = load_iris()
    X, y = dataset.data, dataset.target

    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Create the machine learning pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        )),
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = float(accuracy_score(y_test, predictions))

    report = {
        "dataset": "Iris (scikit-learn built-in dataset)",
        "samples": int(len(y)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "features": dataset.feature_names,
        "classes": dataset.target_names.tolist(),
        "model": (
            "StandardScaler + RandomForestClassifier"
            "(n_estimators=200, random_state=42)"
        ),
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
            output_dict=True,
        ),
        "scikit_learn_version": sklearn.__version__,
    }

    # Save the trained model
    model_path = ARTIFACT_DIR / "iris_model.joblib"
    joblib.dump(model, model_path)

    # Save evaluation results
    report_path = ARTIFACT_DIR / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Display results
    print(json.dumps({
        "artifact": str(model_path),
        "evaluation_report": str(report_path),
        "accuracy": accuracy,
        "confusion_matrix": report["confusion_matrix"],
    }, indent=2))


if __name__ == "__main__":
    main()
