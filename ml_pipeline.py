import csv
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import joblib
import librosa
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyzer_ml import build_feature_vector, compute_dsp_metrics, extract_metadata

MODEL_LABELS = ["human", "hybrid", "ai"]
DEFAULT_MODEL_PATH = Path("model.joblib")
DEFAULT_FEATURE_STORE_PATH = Path("training_features.csv")


def _resolve_audio_path(audio_path: str, dataset_csv: Path, audio_root: Optional[Path]) -> Path:
    candidate = Path(audio_path)
    if candidate.is_absolute():
        return candidate
    if audio_root is not None:
        return audio_root / candidate
    return dataset_csv.parent / candidate


def load_dataset_rows(dataset_csv: str, audio_root: Optional[str] = None) -> list[dict]:
    csv_path = Path(dataset_csv)
    root_path = Path(audio_root) if audio_root else None
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label"}
        if reader.fieldnames is None or not required.issubset({field.strip() for field in reader.fieldnames}):
            raise ValueError("Dataset CSV harus punya kolom 'path' dan 'label'.")

        for row in reader:
            label = row["label"].strip().lower()
            if label not in MODEL_LABELS:
                raise ValueError(f"Label '{label}' tidak valid. Gunakan salah satu dari: {', '.join(MODEL_LABELS)}.")

            resolved_path = _resolve_audio_path(row["path"].strip(), csv_path, root_path)
            rows.append({"path": resolved_path, "label": label})
    return rows


def extract_training_matrix(dataset_rows: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    feature_rows = []
    labels = []
    feature_columns = None
    skipped_rows = []

    for row in dataset_rows:
        audio_path = row["path"]
        try:
            metadata = extract_metadata(str(audio_path))
            y, sr = librosa.load(str(audio_path), sr=None)
            dsp, submithub = compute_dsp_metrics(y, sr)
            feature_vector = build_feature_vector(metadata, dsp, submithub)
        except Exception as exc:
            skipped_rows.append(
                {
                    "path": str(audio_path),
                    "label": row["label"],
                    "reason": str(exc),
                }
            )
            continue

        if feature_columns is None:
            feature_columns = sorted(feature_vector.keys())

        feature_rows.append([feature_vector[name] for name in feature_columns])
        labels.append(MODEL_LABELS.index(row["label"]))

    if not feature_rows or feature_columns is None:
        raise ValueError("Dataset kosong. Tambahkan minimal beberapa file audio berlabel.")

    return np.asarray(feature_rows, dtype=float), np.asarray(labels, dtype=int), feature_columns, skipped_rows


def append_feature_row(
    feature_store_csv: str,
    label: str,
    feature_vector: Dict[str, float],
    source_name: str = "",
    source_path: str = "",
) -> None:
    if label not in MODEL_LABELS:
        raise ValueError(f"Label '{label}' tidak valid.")

    csv_path = Path(feature_store_csv)
    feature_columns = sorted(feature_vector.keys())
    fieldnames = ["label", "source_name", "source_path", "added_at"] + feature_columns
    row = {
        "label": label,
        "source_name": source_name,
        "source_path": source_path,
        "added_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    row.update({key: float(feature_vector.get(key, 0.0)) for key in feature_columns})

    existing_rows = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
        existing_fieldnames = reader.fieldnames or []
        merged_fields = []
        for name in ["label", "source_name", "source_path", "added_at", *existing_fieldnames, *fieldnames]:
            if name not in merged_fields:
                merged_fields.append(name)
        fieldnames = merged_fields

    for existing in existing_rows:
        for field in fieldnames:
            existing.setdefault(field, "")
    for field in fieldnames:
        row.setdefault(field, "")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows + [row])


def load_feature_store_rows(feature_store_csv: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    csv_path = Path(feature_store_csv)
    if not csv_path.exists():
        raise ValueError("Feature store belum tersedia.")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise ValueError("Feature store kosong.")
        fieldnames = reader.fieldnames or []

    feature_columns = [name for name in fieldnames if name not in {"label", "source_name", "source_path", "added_at"}]
    x_rows = []
    y_rows = []
    for row in rows:
        label = row["label"].strip().lower()
        if label not in MODEL_LABELS:
            continue
        x_rows.append([float(row.get(name, 0.0) or 0.0) for name in feature_columns])
        y_rows.append(MODEL_LABELS.index(label))

    if not x_rows:
        raise ValueError("Feature store tidak memiliki baris valid.")

    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int), feature_columns


def build_feature_store_from_dataset(
    dataset_csv: str,
    feature_store_csv: str = str(DEFAULT_FEATURE_STORE_PATH),
    audio_root: Optional[str] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    dataset_rows = load_dataset_rows(dataset_csv, audio_root=audio_root)
    feature_store_path = Path(feature_store_csv)
    rows_written = 0
    skipped_rows = []
    existing_source_paths = set()
    processed_rows = 0

    if feature_store_path.exists():
        with feature_store_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_path = (row.get("source_path") or "").strip()
                if source_path:
                    existing_source_paths.add(source_path)

    total_rows = len(dataset_rows)
    if progress_callback is not None:
        progress_callback(
            {
                "processed": 0,
                "total": total_rows,
                "added": rows_written,
                "skipped": len(skipped_rows),
                "current_file": "",
                "status": "starting",
            }
        )

    for row in dataset_rows:
        audio_path = row["path"]
        source_path = str(audio_path.relative_to(Path(dataset_csv).parent).as_posix()) if Path(audio_path).is_absolute() else str(audio_path)
        if source_path in existing_source_paths:
            processed_rows += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "processed": processed_rows,
                        "total": total_rows,
                        "added": rows_written,
                        "skipped": len(skipped_rows),
                        "current_file": source_path,
                        "status": "already_exists",
                    }
                )
            continue
        try:
            metadata = extract_metadata(str(audio_path))
            y, sr = librosa.load(str(audio_path), sr=None)
            dsp, submithub = compute_dsp_metrics(y, sr)
            feature_vector = build_feature_vector(metadata, dsp, submithub)
            append_feature_row(
                feature_store_csv=feature_store_csv,
                label=row["label"],
                feature_vector=feature_vector,
                source_name=metadata.filename,
                source_path=source_path,
            )
            rows_written += 1
            existing_source_paths.add(source_path)
            processed_rows += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "processed": processed_rows,
                        "total": total_rows,
                        "added": rows_written,
                        "skipped": len(skipped_rows),
                        "current_file": source_path,
                        "status": "added",
                    }
                )
        except Exception as exc:
            skipped_rows.append({"path": str(audio_path), "label": row["label"], "reason": str(exc)})
            processed_rows += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "processed": processed_rows,
                        "total": total_rows,
                        "added": rows_written,
                        "skipped": len(skipped_rows),
                        "current_file": source_path,
                        "status": "skipped",
                        "reason": str(exc),
                    }
                )

    return {
        "rows_seen": len(dataset_rows),
        "rows_added": rows_written,
        "rows_skipped": skipped_rows,
        "feature_store": str(feature_store_path),
    }


def train_model(
    dataset_csv: str,
    model_output: str = str(DEFAULT_MODEL_PATH),
    audio_root: Optional[str] = None,
    feature_store_csv: Optional[str] = None,
) -> dict:
    skipped_rows = []
    dataset_rows = []
    used_feature_store = False

    if feature_store_csv:
        feature_store_path = Path(feature_store_csv)
        if feature_store_path.exists():
            x_train, y_train, feature_columns = load_feature_store_rows(str(feature_store_path))
            used_feature_store = True
        else:
            dataset_rows = load_dataset_rows(dataset_csv, audio_root=audio_root)
            x_train, y_train, feature_columns, skipped_rows = extract_training_matrix(dataset_rows)
    else:
        dataset_rows = load_dataset_rows(dataset_csv, audio_root=audio_root)
        x_train, y_train, feature_columns, skipped_rows = extract_training_matrix(dataset_rows)

    class_counts = np.bincount(y_train, minlength=len(MODEL_LABELS))
    non_zero_counts = [count for count in class_counts if count > 0]
    if len(non_zero_counts) < 2:
        raise ValueError("Training butuh minimal 2 kelas berbeda.")

    min_class_count = min(non_zero_counts)
    if min_class_count < 2:
        raise ValueError("Setiap kelas yang dipakai untuk training butuh minimal 2 sampel agar probabilitas bisa dikalibrasi.")

    cv_folds = min(3, min_class_count)
    base_estimator = LogisticRegression(max_iter=4000, class_weight="balanced")
    calibrated = CalibratedClassifierCV(base_estimator, cv=cv_folds, method="sigmoid")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", calibrated),
        ]
    )
    pipeline.fit(x_train, y_train)

    bundle = {
        "model": pipeline,
        "feature_columns": feature_columns,
        "labels": MODEL_LABELS,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_rows": len(dataset_rows) if dataset_rows else int(len(y_train)),
        "used_rows": int(len(y_train)),
        "skipped_rows": skipped_rows,
        "class_counts": {MODEL_LABELS[idx]: int(count) for idx, count in enumerate(class_counts)},
        "used_feature_store": used_feature_store,
        "feature_store_csv": feature_store_csv or "",
    }
    joblib.dump(bundle, model_output)
    return bundle


def load_model_bundle(model_path: str = str(DEFAULT_MODEL_PATH)) -> Optional[dict]:
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def predict_probabilities(model_bundle: dict, feature_vector: Dict[str, float]) -> Dict[str, float]:
    feature_columns = model_bundle["feature_columns"]
    labels = model_bundle["labels"]
    model = model_bundle["model"]
    row = np.asarray([[feature_vector.get(name, 0.0) for name in feature_columns]], dtype=float)
    probs = model.predict_proba(row)[0]
    return {label: float(prob) * 100.0 for label, prob in zip(labels, probs)}


def assess_model_reliability(model_bundle: Optional[dict]) -> tuple[bool, str]:
    if model_bundle is None:
        return False, "Model belum tersedia, jadi sistem memakai heuristik."

    dataset_rows = int(model_bundle.get("dataset_rows", 0))
    class_counts = model_bundle.get("class_counts", {})
    min_class_count = min(class_counts.values()) if class_counts else 0

    if dataset_rows < 30:
        return False, f"Dataset training masih kecil ({dataset_rows} sampel)."
    if min_class_count < 10:
        return False, "Jumlah sampel per kelas masih terlalu sedikit."
    return True, f"Model terlatih dengan {dataset_rows} sampel."


def assess_prediction_confidence(probabilities: Optional[Dict[str, float]]) -> tuple[str, str]:
    if probabilities is None:
        return "low", "Prediksi model tidak tersedia."

    ordered = sorted(probabilities.values(), reverse=True)
    top_1 = ordered[0]
    top_2 = ordered[1] if len(ordered) > 1 else 0.0
    margin = top_1 - top_2

    if top_1 >= 75 and margin >= 25:
        return "high", "Model cukup yakin karena kelas teratas unggul jauh."
    if top_1 >= 55 and margin >= 10:
        return "medium", "Model memberi sinyal yang lumayan jelas, tapi belum dominan."
    return "low", "Perbedaan skor antar kelas terlalu tipis."
