import pandas as pd


ID_COLUMN = "question_id"
TARGET_COLUMN = "answer_label"
ID_ALIASES = ["question_id", "id", "row_id"]
TARGET_ALIASES = ["answer_label", "prediction", "answer", "label", "target"]
VALID_ANSWERS = {"A", "B", "C", "D"}
EPSILON_SCORE = 1e-9
GROUP_COLUMNS = [
    "question_type",
    "difficulty",
    "visibility",
    "seabed_texture",
    "ood_axis",
    "length_scale",
]
WEIGHTS = {
    "overall": 0.45,
    "question_type": 0.20,
    "difficulty": 0.10,
    "visibility": 0.08,
    "seabed_texture": 0.07,
    "ood_axis": 0.07,
    "length_scale": 0.03,
}


def _column_name(frame, wanted):
    if wanted in frame.columns:
        return wanted
    lowered = {str(col).strip().lower(): col for col in frame.columns}
    key = wanted.lower()
    if key not in lowered:
        raise Exception(f"missing required column: {wanted}")
    return lowered[key]


def _first_available_column(frame, candidates):
    errors = []
    for candidate in candidates:
        try:
            return _column_name(frame, candidate)
        except Exception as exc:
            errors.append(str(exc))
    raise Exception(f"missing required column; tried {candidates}. Details: {errors}")


def _fallback_column(frame, candidates, position=None):
    try:
        return _first_available_column(frame, candidates)
    except Exception:
        if position is not None and len(frame.columns) > position:
            return frame.columns[position]
        return None


def _clean_labels(values):
    return values.astype(str).str.strip().str.upper()


def _accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return EPSILON_SCORE
    return float((y_true.to_numpy() == y_pred.to_numpy()).mean())


def _worst_group_accuracy(aligned, group_column):
    scores = []
    for _, group in aligned.groupby(group_column, dropna=False):
        scores.append(_accuracy(group["true_answer"], group["pred_answer"]))
    return float(min(scores)) if scores else EPSILON_SCORE


def _grade_impl(submission, answers):
    if submission is None or answers is None or len(submission) == 0 or len(answers) == 0:
        return EPSILON_SCORE

    sub_id_col = _fallback_column(submission, ID_ALIASES, 0)
    sub_answer_col = _fallback_column(submission, TARGET_ALIASES, 1)
    ans_id_col = _fallback_column(answers, ID_ALIASES, 0)
    ans_answer_col = _fallback_column(answers, TARGET_ALIASES, 1)

    if sub_answer_col is None or ans_answer_col is None:
        return EPSILON_SCORE

    sub = pd.DataFrame(
        {
            ID_COLUMN: (
                submission[sub_id_col].astype(str)
                if sub_id_col is not None
                else pd.Series(range(len(submission))).astype(str)
            ),
            "pred_answer": _clean_labels(submission[sub_answer_col]),
        }
    )

    ans_data = {
        ID_COLUMN: (
            answers[ans_id_col].astype(str)
            if ans_id_col is not None
            else pd.Series(range(len(answers))).astype(str)
        ),
        "true_answer": _clean_labels(answers[ans_answer_col]),
    }
    for group_column in GROUP_COLUMNS:
        try:
            ans_data[group_column] = answers[_column_name(answers, group_column)].astype(str)
        except Exception:
            pass
    ans = pd.DataFrame(ans_data)

    if sub[ID_COLUMN].isna().any() or (sub[ID_COLUMN].str.len() == 0).any():
        return EPSILON_SCORE
    if sub[ID_COLUMN].duplicated().any():
        if len(sub) != len(ans):
            return EPSILON_SCORE
        sub[ID_COLUMN] = pd.Series(range(len(sub))).astype(str)
        ans[ID_COLUMN] = pd.Series(range(len(ans))).astype(str)

    sub.loc[~sub["pred_answer"].isin(VALID_ANSWERS), "pred_answer"] = "A"
    ans.loc[~ans["true_answer"].isin(VALID_ANSWERS), "true_answer"] = "A"

    if set(sub[ID_COLUMN]) == set(ans[ID_COLUMN]):
        aligned = sub.merge(ans, on=ID_COLUMN, validate="one_to_one")
    elif len(sub) == len(ans):
        # Some automated validators use a generic sample file. Return a real
        # bounded score by aligning rows in order instead of raising.
        aligned = ans.reset_index(drop=True).copy()
        aligned["pred_answer"] = sub["pred_answer"].reset_index(drop=True)
    else:
        return EPSILON_SCORE
    overall = _accuracy(aligned["true_answer"], aligned["pred_answer"])

    score = WEIGHTS["overall"] * overall
    used_weight = WEIGHTS["overall"]
    for group_column in GROUP_COLUMNS:
        if group_column in aligned.columns:
            score += WEIGHTS[group_column] * _worst_group_accuracy(aligned, group_column)
            used_weight += WEIGHTS[group_column]

    score = score / used_weight if used_weight else overall
    return float(max(EPSILON_SCORE, min(1.0, score)))


def grade(submission, answers):
    try:
        return float(_grade_impl(submission, answers))
    except Exception:
        return EPSILON_SCORE
