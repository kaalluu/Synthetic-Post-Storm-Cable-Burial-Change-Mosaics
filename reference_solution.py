from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SIZE = (72, 54)
LABELS = np.array(["A", "B", "C", "D"])


def _one_hot(value: str, choices: list[str]) -> np.ndarray:
    return np.asarray([1.0 if value == choice else 0.0 for choice in choices], dtype=np.float32)


def _load_image_features(public_dir: Path, rel_path: str) -> np.ndarray:
    image = Image.open(public_dir / rel_path).convert("L").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    h, w = arr.shape
    gx = np.abs(np.diff(arr, axis=1)).mean(axis=0)
    gy = np.abs(np.diff(arr, axis=0)).mean(axis=1)
    bands = [
        arr,
        arr[h // 3: 2 * h // 3, :],
        arr[:, : w // 3],
        arr[:, w // 3: 2 * w // 3],
        arr[:, 2 * w // 3:],
        arr[: h // 2, :],
        arr[h // 2:, :],
    ]
    stats = []
    for band in bands:
        stats.extend(
            [
                float(band.mean()),
                float(band.std()),
                float(np.percentile(band, 8)),
                float(np.percentile(band, 50)),
                float(np.percentile(band, 92)),
            ]
        )
    return np.concatenate(
        [
            arr.reshape(-1),
            gx.astype(np.float32),
            gy.astype(np.float32),
            np.asarray(stats, dtype=np.float32),
        ]
    )


def _build_features(public_dir: Path, rows: pd.DataFrame) -> np.ndarray:
    visibility_choices = ["clear", "speckled", "low_gain", "multipath", "dropout"]
    texture_choices = ["flat", "rippled", "rocky", "vegetated"]
    difficulty_choices = ["easy", "medium", "hard"]

    image_features = []
    cache: dict[str, np.ndarray] = {}
    for _, row in rows.iterrows():
        path = str(row["image_path"])
        if path not in cache:
            cache[path] = _load_image_features(public_dir, path)
        meta = np.concatenate(
            [
                _one_hot(str(row["visibility"]), visibility_choices),
                _one_hot(str(row["seabed_texture"]), texture_choices),
                _one_hot(str(row["difficulty"]), difficulty_choices),
                np.asarray(
                    [
                        float(row["meters_per_pixel"]),
                        float(row["range_setting_m"]) / 120.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        image_features.append(np.concatenate([cache[path], meta]))
    return np.vstack(image_features).astype(np.float32)


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, l2: float = 50.0) -> np.ndarray:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    y = np.zeros((len(train_y), len(LABELS)), dtype=np.float32)
    label_to_index = {label: idx for idx, label in enumerate(LABELS)}
    for row_idx, label in enumerate(train_y.astype(str)):
        y[row_idx, label_to_index[label]] = 1.0

    kernel = train_x @ train_x.T
    kernel += l2 * np.eye(len(train_x), dtype=np.float32)
    alpha = np.linalg.solve(kernel, y)
    weights = train_x.T @ alpha
    return LABELS[np.argmax(test_x @ weights, axis=1)]


def _paired_change_features(public_dir: Path, row: pd.Series, cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, np.ndarray]:
    path = str(row["image_path"])
    if path not in cache:
        arr = np.asarray(Image.open(public_dir / path).convert("L"), dtype=np.float32)
        _, width = arr.shape
        panel_width = min(512, (width - 14) // 2)
        cache[path] = (arr[:, :panel_width], arr[:, -panel_width:])

    baseline, post = cache[path]

    def panel_stats(panel: np.ndarray) -> tuple[float, np.ndarray]:
        height, width = panel.shape
        band = panel[int(height * 0.34): int(height * 0.66), :]
        bright_ratio = float((band > 165).mean())
        thirds = []
        for idx in range(3):
            part = band[:, int(idx * width / 3): int((idx + 1) * width / 3)]
            thirds.append(float((part > 165).mean()))
        return bright_ratio, np.asarray(thirds, dtype=np.float32)

    baseline_bright, baseline_thirds = panel_stats(baseline)
    post_bright, post_thirds = panel_stats(post)
    return baseline_bright, post_bright, post_thirds - baseline_thirds


def main() -> None:
    root = Path(__file__).resolve().parent
    public_dir = root / "public"
    working_dir = root / "working"
    working_dir.mkdir(exist_ok=True)

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")

    majority_by_type = train.groupby("question_type")["answer_label"].agg(lambda labels: labels.mode()[0]).to_dict()
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for _, row in test.iterrows():
        question_type = str(row["question_type"])
        baseline_bright, post_bright, sector_delta = _paired_change_features(public_dir, row, cache)

        if question_type == "exposure_delta_bin":
            delta_proxy_m = max(0.0, post_bright - baseline_bright) * 512.0 * 384.0 * float(row["meters_per_pixel"]) / 12.0
            if delta_proxy_m < 2.0:
                answer = "A"
            elif delta_proxy_m < 18.0:
                answer = "B"
            elif delta_proxy_m < 45.0:
                answer = "C"
            else:
                answer = "D"
        elif question_type == "new_risk_sector":
            if float(sector_delta.max()) <= 0.003:
                answer = "D"
            else:
                answer = ["A", "B", "C"][int(sector_delta.argmax())]
        else:
            answer = str(majority_by_type.get(question_type, "A"))

        rows.append({"question_id": row["question_id"], "answer_label": answer})

    submission = pd.DataFrame(rows)
    submission.to_csv(working_dir / "submission.csv", index=False)
    print(f"wrote {working_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
