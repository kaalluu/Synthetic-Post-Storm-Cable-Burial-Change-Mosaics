from pathlib import Path
import base64
from io import BytesIO
import hashlib

import numpy as np
import pandas as pd
from PIL import Image


PUBLIC_COLUMNS = [
    "question_id",
    "scene_id",
    "image_path",
    "question_type",
    "difficulty",
    "visibility",
    "seabed_texture",
    "range_setting_m",
    "meters_per_pixel",
    "question",
    "choice_a",
    "choice_b",
    "choice_c",
    "choice_d",
]
ANSWER_COLUMNS = [
    "question_id",
    "answer_label",
    "question_type",
    "difficulty",
    "visibility",
    "seabed_texture",
    "ood_axis",
    "length_scale",
]
RANDOM_SEED = 20260619
PUBLIC_SALT = "subsea-cable-burial-sonar-vqa-public-v2"


def _stable_token(prefix, value, length=14):
    digest = hashlib.sha256(f"{PUBLIC_SALT}|{value}".encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def _stable_seed(value):
    digest = hashlib.sha256(f"{PUBLIC_SALT}|seed|{value}".encode("utf-8")).hexdigest()[:16]
    return (int(digest, 16) + RANDOM_SEED) % (2**32)


def _find_questions_file(raw):
    direct = raw / "questions.csv"
    if direct.exists():
        return direct
    matches = sorted(raw.rglob("questions.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not find questions.csv in the uploaded raw dataset")


def _find_image(raw, source_root, rel_path):
    rel = Path(str(rel_path).replace("\\", "/"))
    candidates = [
        raw / rel,
        source_root / rel,
        raw / rel.name,
        source_root / rel.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(raw.rglob(rel.name))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find image '{rel_path}'. Expected it under an images/ folder "
        "in the raw upload, or with the same filename somewhere inside the upload."
    )


def _load_encoded_images(raw, source_root):
    candidates = [source_root / "images.csv", raw / "images.csv", *sorted(raw.rglob("images.csv"))]
    images_csv = next((path for path in candidates if path.exists()), None)
    if images_csv is None:
        return {}

    frame = pd.read_csv(images_csv)
    required = {"image_path", "image_png_base64"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{images_csv} missing columns: {sorted(missing)}")

    encoded = {}
    for _, row in frame.iterrows():
        rel_path = str(row["image_path"]).replace("\\", "/")
        data = base64.b64decode(str(row["image_png_base64"]))
        encoded[rel_path] = data
        encoded[Path(rel_path).name] = data
    return encoded


def _make_public_rows(questions):
    public_rows = questions.copy()
    public_rows["_raw_image_path"] = public_rows["image_path"].astype(str)
    raw_scene = public_rows["scene_id"].astype(str)
    raw_question = public_rows["question_id"].astype(str)

    scene_map = {scene_id: _stable_token("s", scene_id) for scene_id in sorted(raw_scene.unique())}
    question_map = {qid: _stable_token("q", qid) for qid in sorted(raw_question.unique())}

    public_rows["scene_id"] = raw_scene.map(scene_map)
    public_rows["question_id"] = raw_question.map(question_map)
    public_rows["image_path"] = public_rows["scene_id"].map(lambda scene_id: f"images/{scene_id}.png")

    scale_by_scene = {}
    for original_scene_id, group in questions.groupby(questions["scene_id"].astype(str), sort=True):
        seed = _stable_seed(original_scene_id)
        rng = np.random.default_rng(seed)
        factor = float(rng.uniform(0.9875, 1.0125))
        meters_per_pixel = round(float(group["meters_per_pixel"].iloc[0]) * factor, 4)
        range_setting_m = round(meters_per_pixel * 512.0, 1)
        scale_by_scene[scene_map[original_scene_id]] = (range_setting_m, meters_per_pixel)

    public_rows["range_setting_m"] = public_rows["scene_id"].map(lambda scene_id: scale_by_scene[scene_id][0])
    public_rows["meters_per_pixel"] = public_rows["scene_id"].map(lambda scene_id: scale_by_scene[scene_id][1])
    return public_rows


def _write_obfuscated_image(src, data, dst, seed_value):
    if src is not None:
        image = Image.open(src).convert("L")
    elif data is not None:
        image = Image.open(BytesIO(data)).convert("L")
    else:
        raise FileNotFoundError(f"No source image data available for {dst.name}")

    seed = _stable_seed(seed_value)
    rng = np.random.default_rng(seed)
    arr = np.asarray(image, dtype=np.float32)

    contrast = float(rng.uniform(0.985, 1.015))
    brightness = float(rng.uniform(-1.5, 1.5))
    noise = rng.normal(0.0, 0.65, size=arr.shape)
    arr = (arr - 127.5) * contrast + 127.5 + brightness + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(dst)


def _copy_images(rows, raw, source_root, public):
    (public / "images").mkdir(parents=True, exist_ok=True)
    encoded_images = _load_encoded_images(raw, source_root)
    image_rows = rows[["_raw_image_path", "image_path", "scene_id"]].drop_duplicates()
    for _, row in image_rows.sort_values("image_path").iterrows():
        clean_rel = str(row["_raw_image_path"]).replace("\\", "/")
        dst = public / str(row["image_path"])
        src = None
        data = None
        try:
            src = _find_image(raw, source_root, clean_rel)
        except FileNotFoundError:
            data = encoded_images.get(clean_rel) or encoded_images.get(Path(clean_rel).name)
            if data is None:
                raise
        _write_obfuscated_image(src, data, dst, str(row["scene_id"]))


def prepare(raw, public, private):
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    questions_file = _find_questions_file(raw)
    source_root = questions_file.parent
    raw_questions = pd.read_csv(questions_file)
    required = set(PUBLIC_COLUMNS + ["split", "answer_label", "ood_axis", "length_scale"])
    missing = sorted(required - set(raw_questions.columns))
    if missing:
        raise ValueError(f"raw/questions.csv missing columns: {missing}")

    questions = _make_public_rows(raw_questions)
    train = questions[questions["split"] == "train"].copy()
    test = questions[questions["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("raw/questions.csv must contain both train and test splits")
    if set(train["scene_id"]).intersection(set(test["scene_id"])):
        raise ValueError("scene_id leakage between train and test splits")

    train_public = train[PUBLIC_COLUMNS + ["answer_label"]].copy()
    test_public = test[PUBLIC_COLUMNS].copy()
    answers = test[ANSWER_COLUMNS].copy()

    train_public.to_csv(public / "train.csv", index=False)
    test_public.to_csv(public / "test.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)

    sample = pd.DataFrame({"question_id": test_public["question_id"], "answer_label": "A"})
    sample.to_csv(public / "sample_submission.csv", index=False)

    _copy_images(questions, raw, source_root, public)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    prepare(root / "raw", root / "public", root / "private")
