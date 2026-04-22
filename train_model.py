import argparse
from pathlib import Path

from ml_pipeline import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated audio AI classifier.")
    parser.add_argument("--dataset", required=True, help="CSV file with columns: path,label")
    parser.add_argument("--output", default="model.joblib", help="Output path for trained model bundle")
    parser.add_argument("--audio-root", default=None, help="Optional base directory for relative audio paths")
    parser.add_argument("--feature-store", default="training_features.csv", help="CSV file for stored numeric features")
    args = parser.parse_args()

    bundle = train_model(
        args.dataset,
        model_output=args.output,
        audio_root=args.audio_root,
        feature_store_csv=args.feature_store,
    )
    print(f"Model saved to {Path(args.output).resolve()}")
    print(f"Rows declared: {bundle['dataset_rows']}")
    print(f"Rows used: {bundle['used_rows']}")
    print(f"Class counts: {bundle['class_counts']}")
    print(f"Rows skipped: {len(bundle['skipped_rows'])}")
    print(f"Trained at: {bundle['trained_at']}")
    print(f"Used feature store: {bundle.get('used_feature_store', False)}")


if __name__ == "__main__":
    main()
