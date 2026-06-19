from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import hashlib
import math
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


WIDTH = 512
HEIGHT = 384
N_TRAIN_SCENES = 168
N_TEST_SCENES = 72
BASE_SEED = 20260619


@dataclass
class Segment:
    start: float
    end: float


def stable_id(prefix: str, *parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def _choice(rng: random.Random, values: list[str], weights: list[float] | None = None) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def _make_segments(rng: random.Random, target_fraction: float) -> list[Segment]:
    if target_fraction <= 0.02:
        return []
    count = 1 if target_fraction < 0.35 else rng.choice([1, 2, 2, 3])
    remaining = target_fraction
    segments: list[Segment] = []
    for idx in range(count):
        frac = remaining if idx == count - 1 else rng.uniform(0.12, max(0.14, remaining * 0.72))
        frac = min(frac, remaining)
        for _ in range(100):
            start = rng.uniform(0.04, 0.96 - frac)
            end = start + frac
            if all(end < seg.start - 0.03 or start > seg.end + 0.03 for seg in segments):
                segments.append(Segment(start, end))
                break
        remaining -= frac
        if remaining <= 0.03:
            break
    return sorted(segments, key=lambda seg: seg.start)


def _segment_fraction_in_third(segments: list[Segment], third: int) -> float:
    lo = third / 3.0
    hi = (third + 1) / 3.0
    total = 0.0
    for seg in segments:
        total += max(0.0, min(seg.end, hi) - max(seg.start, lo))
    return total


def _risk_sector(segments: list[Segment]) -> str:
    thirds = [_segment_fraction_in_third(segments, idx) for idx in range(3)]
    if max(thirds) < 0.04:
        return "none"
    return ["left", "middle", "right"][int(np.argmax(thirds))]


def _burial_state(cable_present: bool, exposed_fraction: float) -> str:
    if not cable_present:
        return "no_visible_cable"
    if exposed_fraction < 0.16:
        return "mostly_buried"
    if exposed_fraction < 0.62:
        return "partially_exposed"
    return "mostly_exposed"


def _length_scale(mpp: float) -> str:
    if mpp <= 0.095:
        return "short_range"
    if mpp <= 0.15:
        return "medium_range"
    return "long_range"


def _exposed_length_bin(length_m: float) -> str:
    if length_m < 2.0:
        return "none"
    if length_m < 18.0:
        return "short"
    if length_m < 45.0:
        return "medium"
    return "long"


def _total_exposed_length_m(segments: list[Segment], meters_per_pixel: float) -> float:
    return float(sum((seg.end - seg.start) * WIDTH * meters_per_pixel for seg in segments))


def _derive_pre_storm_trace(post: dict) -> dict:
    rng = random.Random(post["seed"] + 17791)
    pre = post.copy()
    pre["seed"] = post["seed"] + 31037
    pre["wiggle"] = post["wiggle"] + rng.uniform(-0.18, 0.18)

    if not post["cable_present"]:
        pre_fraction = 0.0
    else:
        reduction = rng.uniform(0.10, 0.34)
        if rng.random() < 0.22:
            reduction = rng.uniform(0.00, 0.08)
        pre_fraction = max(0.0, post["exposed_fraction"] - reduction)

    pre["exposed_fraction"] = pre_fraction
    pre["segments"] = _make_segments(rng, pre_fraction)
    pre["burial_state"] = _burial_state(pre["cable_present"], pre_fraction)
    pre["highest_risk_sector"] = _risk_sector(pre["segments"])
    pre["anchor_scar_count"] = max(0, post["anchor_scar_count"] - rng.choice([0, 0, 1, 1, 2]))
    pre["crossing_hazard"] = post["crossing_hazard"] if rng.random() < 0.28 else "none"
    pre["shadow_side"] = "none" if pre_fraction <= 0.08 else _choice(rng, ["above", "below", "both"], [0.38, 0.38, 0.24])
    pre_length = _total_exposed_length_m(pre["segments"], pre["meters_per_pixel"])
    pre["exposed_length_bin"] = _exposed_length_bin(pre_length)
    return pre


def _new_exposure_sector(pre: dict, post: dict) -> str:
    gains = []
    for idx in range(3):
        post_value = _segment_fraction_in_third(post["segments"], idx)
        pre_value = _segment_fraction_in_third(pre["segments"], idx)
        gains.append(max(0.0, post_value - pre_value))
    if max(gains) < 0.035:
        return "none"
    return ["left", "middle", "right"][int(np.argmax(gains))]


def _delta_bin(delta_m: float) -> str:
    if delta_m < 2.0:
        return "stable"
    if delta_m < 18.0:
        return "minor"
    if delta_m < 45.0:
        return "moderate"
    return "severe"


def _intervention_priority(post: dict, delta_m: float) -> str:
    if not post["cable_present"]:
        return "resurvey"
    if post["crossing_hazard"] != "none" or post["anchor_scar_count"] >= 2:
        return "inspect"
    if delta_m >= 45.0 or post["exposed_fraction"] >= 0.62:
        return "rebury"
    if delta_m >= 2.0 or post["exposed_fraction"] >= 0.16:
        return "monitor"
    return "no_action"


def _burial_compliance(post: dict, delta_m: float) -> str:
    if not post["cable_present"]:
        return "indeterminate"
    if post["exposed_fraction"] >= 0.62 or delta_m >= 45.0:
        return "noncompliant"
    if post["exposed_fraction"] >= 0.16 or delta_m >= 18.0 or post["anchor_scar_count"] >= 2:
        return "marginal"
    return "within_spec"


def _add_change_labels(post: dict) -> dict:
    pre = _derive_pre_storm_trace(post)
    pre_length = _total_exposed_length_m(pre["segments"], post["meters_per_pixel"])
    post_length = _total_exposed_length_m(post["segments"], post["meters_per_pixel"])
    delta_m = max(0.0, post_length - pre_length)

    post["pre_trace"] = pre
    post["exposure_delta_bin"] = _delta_bin(delta_m)
    post["new_risk_sector"] = _new_exposure_sector(pre, post)
    post["intervention_priority"] = _intervention_priority(post, delta_m)
    post["new_scar_count"] = min(3, max(0, post["anchor_scar_count"] - pre["anchor_scar_count"]))
    post["new_crossing_threat"] = post["crossing_hazard"] if pre["crossing_hazard"] == "none" else "none"
    post["burial_compliance"] = _burial_compliance(post, delta_m)
    post["exposed_length_bin"] = _exposed_length_bin(post_length)
    return post


def make_trace(index: int, split: str) -> dict:
    seed = BASE_SEED + index * 7919
    rng = random.Random(seed)

    if split == "train":
        ood_axis = "in_distribution"
        if rng.random() < 0.12:
            ood_axis = _choice(rng, ["mild_ripple", "mild_speckle", "mild_clutter"])
    else:
        ood_axis = _choice(
            rng,
            [
                "in_distribution",
                "low_gain",
                "diagonal_cable",
                "heavy_ripple",
                "clutter_decoy",
                "partial_dropout",
            ],
            [0.34, 0.14, 0.16, 0.14, 0.12, 0.10],
        )

    seabed_texture = _choice(rng, ["flat", "rippled", "rocky", "vegetated"], [0.34, 0.30, 0.20, 0.16])
    if ood_axis in {"heavy_ripple", "mild_ripple"}:
        seabed_texture = "rippled"
    if ood_axis == "clutter_decoy":
        seabed_texture = _choice(rng, ["rocky", "vegetated"], [0.55, 0.45])

    visibility = _choice(rng, ["clear", "speckled", "low_gain", "multipath"], [0.44, 0.26, 0.17, 0.13])
    if ood_axis == "low_gain":
        visibility = "low_gain"
    if ood_axis == "partial_dropout":
        visibility = "dropout"
    if ood_axis in {"mild_speckle", "heavy_ripple"} and visibility == "clear":
        visibility = "speckled"

    difficulty = _choice(rng, ["easy", "medium", "hard"], [0.45, 0.38, 0.17])
    if split == "test" and ood_axis != "in_distribution":
        difficulty = _choice(rng, ["medium", "hard"], [0.45, 0.55])

    cable_present = rng.random() > (0.08 if split == "train" else 0.11)
    if not cable_present:
        exposed_fraction = 0.0
    else:
        state = _choice(rng, ["mostly_buried", "partially_exposed", "mostly_exposed"], [0.34, 0.46, 0.20])
        if state == "mostly_buried":
            exposed_fraction = rng.uniform(0.0, 0.14)
        elif state == "partially_exposed":
            exposed_fraction = rng.uniform(0.20, 0.58)
        else:
            exposed_fraction = rng.uniform(0.66, 0.90)

    segments = _make_segments(rng, exposed_fraction)
    shadow_side = "none"
    if exposed_fraction > 0.08:
        shadow_side = _choice(rng, ["above", "below", "both"], [0.38, 0.38, 0.24])

    if cable_present and exposed_fraction > 0.05:
        anchor_count = rng.choices([0, 1, 2, 3], weights=[0.48, 0.30, 0.16, 0.06], k=1)[0]
    else:
        anchor_count = 0
    if ood_axis == "clutter_decoy" and anchor_count == 0 and rng.random() < 0.50:
        anchor_count = 1 if cable_present else 0

    hazard_type = _choice(rng, ["none", "pipeline", "rope_net", "debris"], [0.60, 0.14, 0.13, 0.13])
    if not cable_present and hazard_type != "none":
        hazard_type = "none"
    if ood_axis == "clutter_decoy" and cable_present:
        hazard_type = _choice(rng, ["pipeline", "rope_net", "debris", "none"], [0.25, 0.25, 0.25, 0.25])

    slope = rng.uniform(-0.16, 0.16)
    if ood_axis == "diagonal_cable":
        slope = rng.choice([-1, 1]) * rng.uniform(0.26, 0.38)

    if split == "train":
        mpp = _choice(rng, [0.075, 0.10, 0.125, 0.16], [0.24, 0.36, 0.28, 0.12])
    else:
        mpp = _choice(rng, [0.075, 0.10, 0.125, 0.16, 0.22], [0.18, 0.28, 0.24, 0.18, 0.12])
    exposed_length_m = sum((seg.end - seg.start) * WIDTH * mpp for seg in segments)

    trace = {
        "scene_id": f"scene_{index:04d}",
        "seed": seed,
        "split": split,
        "ood_axis": ood_axis,
        "seabed_texture": seabed_texture,
        "visibility": visibility,
        "difficulty": difficulty,
        "cable_present": cable_present,
        "exposed_fraction": exposed_fraction,
        "segments": segments,
        "burial_state": _burial_state(cable_present, exposed_fraction),
        "highest_risk_sector": _risk_sector(segments),
        "anchor_scar_count": min(anchor_count, 3),
        "crossing_hazard": hazard_type,
        "shadow_side": shadow_side,
        "meters_per_pixel": mpp,
        "range_setting_m": round(WIDTH * mpp, 1),
        "length_scale": _length_scale(mpp),
        "exposed_length_bin": _exposed_length_bin(exposed_length_m),
        "slope": slope,
        "wiggle": rng.uniform(0.0, 1.0),
    }
    return _add_change_labels(trace)


def _base_sonar_array(trace: dict) -> np.ndarray:
    rng = np.random.default_rng(trace["seed"])
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    base = rng.normal(74, 10, size=(HEIGHT, WIDTH)).astype(np.float32)
    base += 12 * (x / WIDTH) + 5 * np.sin(y / 27.0)

    texture = trace["seabed_texture"]
    if texture == "rippled":
        base += 13 * np.sin((y * 0.18) + (x * 0.045) + trace["wiggle"] * 6)
        base += 5 * np.sin((y * 0.08) - (x * 0.03))
    elif texture == "rocky":
        base += rng.normal(0, 17, size=(HEIGHT, WIDTH))
    elif texture == "vegetated":
        base += 8 * np.sin(x * 0.10 + trace["wiggle"] * 5)
        for _ in range(80):
            cx = rng.integers(0, WIDTH)
            cy = rng.integers(0, HEIGHT)
            radius = rng.integers(2, 7)
            base[max(0, cy - radius): min(HEIGHT, cy + radius), max(0, cx - 1): min(WIDTH, cx + 2)] -= rng.uniform(5, 18)

    visibility = trace["visibility"]
    if visibility == "speckled":
        base += rng.normal(0, 18, size=(HEIGHT, WIDTH))
    elif visibility == "low_gain":
        base = 64 + (base - base.mean()) * 0.55
    elif visibility == "multipath":
        base += 7 * np.sin(x * 0.12)
        base += 5 * np.sin((x + y) * 0.055)
    elif visibility == "dropout":
        base = 67 + (base - base.mean()) * 0.72

    # Nadir band and range tick marks give the image a side-scan character.
    nadir = np.exp(-((y - HEIGHT / 2) ** 2) / (2 * 14 ** 2))
    base -= 16 * nadir
    for tick in range(64, WIDTH, 64):
        base[HEIGHT - 18: HEIGHT - 12, tick - 1: tick + 1] += 44

    return np.clip(base, 0, 255)


def _cable_points(trace: dict) -> list[tuple[int, int]]:
    points = []
    center = HEIGHT * 0.50
    amp = 10 if trace["difficulty"] != "easy" else 6
    for px in range(0, WIDTH, 8):
        rel = (px - WIDTH / 2) / WIDTH
        py = center + trace["slope"] * HEIGHT * rel + amp * math.sin(px / 55.0 + trace["wiggle"] * 5)
        points.append((px, int(round(py))))
    return points


def _point_on_path(points: list[tuple[int, int]], x: int) -> tuple[int, int]:
    if x <= points[0][0]:
        return points[0]
    for left, right in zip(points[:-1], points[1:]):
        if left[0] <= x <= right[0]:
            span = max(1, right[0] - left[0])
            t = (x - left[0]) / span
            y = left[1] * (1 - t) + right[1] * t
            return x, int(round(y))
    return points[-1]


def _draw_segmented_line(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], segments: list[Segment], fill: int, width: int) -> None:
    if not segments:
        return
    for seg in segments:
        lo = int(seg.start * WIDTH)
        hi = int(seg.end * WIDTH)
        sub = [pt for pt in points if lo <= pt[0] <= hi]
        if len(sub) >= 2:
            draw.line(sub, fill=fill, width=width, joint="curve")


def _draw_anchor_scars(draw: ImageDraw.ImageDraw, rng: random.Random, trace: dict, points: list[tuple[int, int]]) -> None:
    count = trace["anchor_scar_count"]
    xs = []
    for _ in range(count):
        xs.append(rng.randint(70, WIDTH - 70))
    if trace["ood_axis"] == "clutter_decoy":
        xs.extend([rng.randint(60, WIDTH - 60) for _ in range(rng.randint(1, 3))])

    for idx, x in enumerate(xs):
        _, y = _point_on_path(points, x)
        touches = idx < count
        offset = 0 if touches else rng.choice([-85, 85])
        y0 = y + offset
        length = rng.randint(95, 150)
        tilt = rng.choice([-1, 1]) * rng.uniform(0.45, 0.85)
        x1 = int(x - length / 2)
        x2 = int(x + length / 2)
        y1 = int(y0 - tilt * length / 2)
        y2 = int(y0 + tilt * length / 2)
        draw.line([(x1, y1), (x2, y2)], fill=35, width=rng.randint(5, 8))
        draw.line([(x1, y1 - 4), (x2, y2 - 4)], fill=125, width=2)


def _draw_hazard(draw: ImageDraw.ImageDraw, rng: random.Random, trace: dict, points: list[tuple[int, int]]) -> None:
    hazard = trace["crossing_hazard"]
    if hazard == "none":
        return
    x = rng.randint(145, WIDTH - 145)
    _, y = _point_on_path(points, x)
    if hazard == "pipeline":
        draw.line([(x - 24, y - 130), (x + 28, y + 130)], fill=168, width=12)
        draw.line([(x - 18, y - 128), (x + 34, y + 132)], fill=48, width=3)
    elif hazard == "rope_net":
        bbox = [x - 42, y - 34, x + 42, y + 34]
        draw.ellipse(bbox, outline=156, width=5)
        draw.arc([x - 58, y - 50, x + 58, y + 50], 20, 210, fill=62, width=4)
    elif hazard == "debris":
        for _ in range(11):
            cx = x + rng.randint(-40, 42)
            cy = y + rng.randint(-34, 34)
            r = rng.randint(7, 18)
            shade = rng.choice([42, 55, 145, 162])
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=shade)


def render_scene_image(trace: dict) -> Image.Image:
    arr = _base_sonar_array(trace)
    image = Image.fromarray(arr.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=0.35))
    draw = ImageDraw.Draw(image)
    rng = random.Random(trace["seed"] + 404)
    points = _cable_points(trace)

    if trace["cable_present"]:
        # Buried cable is faint and broken.
        for left, right in zip(points[:-1], points[1:]):
            if rng.random() > 0.32:
                draw.line([left, right], fill=101, width=2)

        segments = trace["segments"]
        if trace["shadow_side"] in {"above", "both"}:
            shadow_points = [(x, y - 10 - rng.randint(0, 3)) for x, y in points]
            _draw_segmented_line(draw, shadow_points, segments, fill=28, width=8)
        if trace["shadow_side"] in {"below", "both"}:
            shadow_points = [(x, y + 10 + rng.randint(0, 3)) for x, y in points]
            _draw_segmented_line(draw, shadow_points, segments, fill=28, width=8)
        _draw_segmented_line(draw, points, segments, fill=184, width=5)
        _draw_segmented_line(draw, points, segments, fill=218, width=2)

    _draw_anchor_scars(draw, rng, trace, points)
    _draw_hazard(draw, rng, trace, points)

    if trace["ood_axis"] == "partial_dropout" or trace["visibility"] == "dropout":
        for _ in range(2):
            x0 = rng.randint(30, WIDTH - 120)
            y0 = rng.randint(20, HEIGHT - 100)
            draw.rectangle([x0, y0, x0 + rng.randint(45, 110), y0 + rng.randint(35, 90)], fill=rng.randint(46, 74))

    if trace["difficulty"] == "hard":
        image = image.filter(ImageFilter.GaussianBlur(radius=0.65))

    return image


def render_scene(trace: dict, output_path: Path) -> None:
    image = render_scene_image(trace)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_paired_scene(trace: dict, output_path: Path) -> None:
    pre_image = render_scene_image(trace["pre_trace"])
    post_image = render_scene_image(trace)
    gap = 14
    canvas = Image.new("L", (WIDTH * 2 + gap, HEIGHT), 52)
    canvas.paste(pre_image, (0, 0))
    canvas.paste(post_image, (WIDTH + gap, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([WIDTH, 0, WIDTH + gap - 1, HEIGHT], fill=38)
    draw.text((18, 12), "BASELINE", fill=215)
    draw.text((WIDTH + gap + 18, 12), "POST-STORM", fill=215)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


QUESTION_SPECS = {
    "exposure_delta_bin": {
        "question": "Comparing baseline and post-storm panels, how much new exposed cable is visible?",
        "choices": {
            "A": "Stable: no measurable new exposure",
            "B": "Minor: under 18 m of new exposure",
            "C": "Moderate: 18 to 45 m of new exposure",
            "D": "Severe: over 45 m of new exposure",
        },
        "label_map": {"stable": "A", "minor": "B", "moderate": "C", "severe": "D"},
    },
    "new_risk_sector": {
        "question": "Which third of the corridor has the largest post-storm increase in exposure risk?",
        "choices": {"A": "Left third", "B": "Middle third", "C": "Right third", "D": "No meaningful increase"},
        "label_map": {"left": "A", "middle": "B", "right": "C", "none": "D"},
    },
    "new_scar_count": {
        "question": "How many new anchor-drag scars touch the cable in the post-storm panel?",
        "choices": {"A": "0", "B": "1", "C": "2", "D": "3 or more"},
        "label_map": {0: "A", 1: "B", 2: "C", 3: "D"},
    },
    "new_crossing_threat": {
        "question": "What new crossing threat appears after the storm compared with the baseline panel?",
        "choices": {"A": "No new crossing threat", "B": "Rigid pipeline", "C": "Rope or net loop", "D": "Boulder or debris pile"},
        "label_map": {"none": "A", "pipeline": "B", "rope_net": "C", "debris": "D"},
    },
    "burial_compliance": {
        "question": "Based on the post-storm panel and the scale metadata, what is the burial-compliance status?",
        "choices": {"A": "Within specification", "B": "Marginal: monitor closely", "C": "Non-compliant exposure", "D": "Indeterminate: resurvey needed"},
        "label_map": {"within_spec": "A", "marginal": "B", "noncompliant": "C", "indeterminate": "D"},
    },
    "intervention_priority": {
        "question": "What is the most appropriate post-storm intervention priority?",
        "choices": {"A": "No action", "B": "Monitor on next survey", "C": "Targeted inspection", "D": "Immediate reburial work order"},
        "label_map": {"no_action": "A", "monitor": "B", "inspect": "C", "rebury": "D", "resurvey": "C"},
    },
}


def rows_for_trace(trace: dict) -> list[dict]:
    rows = []
    image_path = f"images/{trace['scene_id']}.png"
    for question_type, spec in QUESTION_SPECS.items():
        value = trace[question_type]
        answer = spec["label_map"][value]
        qid = stable_id("q", trace["scene_id"], question_type)
        row = {
            "question_id": qid,
            "scene_id": trace["scene_id"],
            "image_path": image_path,
            "split": trace["split"],
            "question_type": question_type,
            "difficulty": trace["difficulty"],
            "visibility": trace["visibility"],
            "seabed_texture": trace["seabed_texture"],
            "ood_axis": trace["ood_axis"],
            "length_scale": trace["length_scale"],
            "range_setting_m": trace["range_setting_m"],
            "meters_per_pixel": trace["meters_per_pixel"],
            "question": spec["question"],
            "choice_a": spec["choices"]["A"],
            "choice_b": spec["choices"]["B"],
            "choice_c": spec["choices"]["C"],
            "choice_d": spec["choices"]["D"],
            "answer_label": answer,
        }
        rows.append(row)
    return rows


def generate_raw(root: Path) -> None:
    raw = root / "raw"
    image_dir = raw / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = N_TRAIN_SCENES + N_TEST_SCENES
    for index in range(total):
        split = "train" if index < N_TRAIN_SCENES else "test"
        trace = make_trace(index, split)
        render_paired_scene(trace, image_dir / f"{trace['scene_id']}.png")
        rows.extend(rows_for_trace(trace))

    frame = pd.DataFrame(rows)
    frame.to_csv(raw / "questions.csv", index=False)

    image_rows = []
    for image_path in sorted(image_dir.glob("*.png")):
        rel_path = f"images/{image_path.name}"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image_rows.append({"image_path": rel_path, "image_png_base64": encoded})
    pd.DataFrame(image_rows).to_csv(raw / "images.csv", index=False)


if __name__ == "__main__":
    generate_raw(Path(__file__).resolve().parent)
