from pathlib import Path
import joblib
import pandas as pd

from app import train_models, MODEL_ARTIFACT_PATH


def main() -> None:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    df = pd.concat([train_df, test_df], ignore_index=True)

    def log_status(message: str) -> None:
        print(message)

    print("Training models offline...")
    model_results = train_models(df, _status_callback=log_status)

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_results, MODEL_ARTIFACT_PATH)
    print(f"Saved model artifacts to: {MODEL_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
