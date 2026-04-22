import csv
from pathlib import Path

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4"}
LABEL_MAP = {
    "ai": "ai",
    "human": "human",
    "hybrid": "hybrid",
}


def main() -> None:
    base_dir = Path("data")
    output_path = Path("dataset.csv")
    rows = []

    for label_dir in sorted(base_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        label = LABEL_MAP.get(label_dir.name.lower())
        if label is None:
            continue

        for file_path in sorted(label_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            relative_path = file_path.as_posix()
            rows.append({"path": relative_path, "label": label})

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"dataset.csv created with {len(rows)} rows")


if __name__ == "__main__":
    main()
