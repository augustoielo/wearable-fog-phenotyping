#!/usr/bin/env python3
"""
Step 08 — Stage-1 activity-context sensitivity analysis.

Recomputes frozen outer-test metrics within walking, right-turning, and left-turning strata.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

STAGE = "Stage 1"

ACTIVITY_LABELS = {
    1: "walking",
    6: "right_turn",
    7: "left_turn",
}

EXPECTED_STAGE1 = {
    "n_chunks": 3542,
    "class0": 2556,
    "class1": 986,
}

EXPECTED_CNN_POOLED_CM = {
    "tn": 2163,
    "fp": 393,
    "fn": 86,
    "tp": 900,
}

EXPECTED_MODELS = [
    "logistic_regression",
    "rbf_svm",
    "random_forest",
    "raw_temporal_cnn",
]


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()

    if (cwd / "data").exists() and (cwd / "src").exists():
        return cwd

    if cwd.name.lower() == "src" and (cwd.parent / "data").exists():
        return cwd.parent

    try:
        p = Path(__file__).resolve().parent
        if p.name.lower() == "src":
            return p.parent
    except NameError:
        pass

    raise RuntimeError(
        "Could not determine project root. Run from the repository root or src/."
    )


ROOT = find_project_root()

STEP02_DIR = (
    ROOT / "data" / "processed" / "pipeline"
    / "step_02_biomechanical_features"
)
STEP03_DIR = (
    ROOT / "data" / "processed" / "pipeline"
    / "step_03_traditional_ml_benchmark"
)
STEP04_DIR = (
    ROOT / "data" / "processed" / "pipeline"
    / "step_04_raw_temporal_cnn_benchmark"
)

STAGE1_FEATURE_PATH = STEP02_DIR / "step02_stage1_modeling_features.csv"
TRAD_PRED_PATH = STEP03_DIR / "step03_outer_test_predictions.csv"
CNN_PRED_PATH = STEP04_DIR / "step04_seed_ensemble_outer_predictions.csv"

PROCESSED_DIR = (
    ROOT / "data" / "processed" / "pipeline"
    / "step_08_stage1_activity_context_sensitivity"
)
REPORT_DIR = (
    ROOT / "reports" / "pipeline"
    / "step_08_stage1_activity_context_sensitivity"
)

for p in [PROCESSED_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

for p in [STAGE1_FEATURE_PATH, TRAD_PRED_PATH, CNN_PRED_PATH]:
    if not p.exists():
        raise FileNotFoundError(
            f"Required frozen pipeline output not found:\n{p}"
        )


def header(title):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)


def metrics_if_valid(y_true, y_pred, score):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(score, dtype=float)

    n0 = int(np.sum(y_true == 0))
    n1 = int(np.sum(y_true == 1))
    both = (n0 > 0) and (n1 > 0)

    recall0 = (
        float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        if n0 > 0 else np.nan
    )
    recall1 = (
        float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        if n1 > 0 else np.nan
    )

    if both:
        ba = float(balanced_accuracy_score(y_true, y_pred))
        auc = float(roc_auc_score(y_true, score))
        macro_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
    else:
        ba = np.nan
        auc = np.nan
        macro_f1 = np.nan

    return {
        "n": int(len(y_true)),
        "class0_n": n0,
        "class1_n": n1,
        "both_classes_present": int(both),
        "balanced_accuracy": ba,
        "macro_f1": macro_f1,
        "class0_recall": recall0,
        "class1_recall": recall1,
        "roc_auc": auc,
    }


header("1. LOAD FROZEN STAGE-1 OUTER-TEST PREDICTIONS")

stage1_features = pd.read_csv(STAGE1_FEATURE_PATH)

required_lookup = {"chunk_id", "activity", "subjectID", "y"}
missing = required_lookup - set(stage1_features.columns)
if missing:
    raise RuntimeError(
        f"Step-02 Stage-1 file missing columns: {sorted(missing)}"
    )

activity_check = stage1_features.groupby("chunk_id")["activity"].nunique()
if (activity_check != 1).any():
    raise RuntimeError(
        "Activity is not invariant within chunk_id in Step-02 Stage-1 data."
    )

lookup = (
    stage1_features[
        ["chunk_id", "subjectID", "activity", "y"]
    ]
    .drop_duplicates("chunk_id")
    .copy()
)

lookup["activity"] = lookup["activity"].astype(int)
lookup["activity_name"] = lookup["activity"].map(ACTIVITY_LABELS)

if lookup["activity_name"].isna().any():
    bad = sorted(
        lookup.loc[lookup["activity_name"].isna(), "activity"]
        .unique().tolist()
    )
    raise RuntimeError(f"Unexpected Stage-1 activity codes: {bad}")

trad = pd.read_csv(TRAD_PRED_PATH)

required_trad = {
    "stage", "model", "outer_fold", "chunk_id", "subjectID",
    "y_true", "y_pred", "score_class1",
}
missing = required_trad - set(trad.columns)
if missing:
    raise RuntimeError(
        f"Step-03 prediction file missing columns: {sorted(missing)}"
    )

trad = trad[trad["stage"].astype(str) == STAGE].copy()

trad = trad.merge(
    lookup[["chunk_id", "activity", "activity_name"]],
    on="chunk_id",
    how="left",
    validate="many_to_one",
)

if trad["activity"].isna().any():
    raise RuntimeError(
        "Some Step-03 Stage-1 predictions could not be matched to activity."
    )

trad = trad.rename(columns={"score_class1": "score"})

cnn = pd.read_csv(CNN_PRED_PATH)

required_cnn = {
    "stage", "outer_fold", "chunk_id", "subjectID",
    "activity", "y", "y_pred", "y_prob",
}
missing = required_cnn - set(cnn.columns)
if missing:
    raise RuntimeError(
        f"Step-04 CNN prediction file missing columns: {sorted(missing)}"
    )

cnn = cnn[cnn["stage"].astype(str) == STAGE].copy()
cnn["activity"] = cnn["activity"].astype(int)
cnn["activity_name"] = cnn["activity"].map(ACTIVITY_LABELS)

if cnn["activity_name"].isna().any():
    raise RuntimeError(
        "Unexpected activity code in Step-04 Stage-1 predictions."
    )

cnn = cnn.rename(columns={"y": "y_true", "y_prob": "score"})
cnn["model"] = "raw_temporal_cnn"

cols = [
    "stage", "model", "outer_fold", "chunk_id", "subjectID",
    "activity", "activity_name", "y_true", "y_pred", "score",
]

cnn = cnn[cols].copy()
trad = trad[cols].copy()

pred = pd.concat([trad, cnn], ignore_index=True)

print(f"Traditional Stage-1 prediction rows: {len(trad)}")
print(f"CNN Stage-1 prediction rows        : {len(cnn)}")
print(f"Combined prediction rows           : {len(pred)}")


header("2. STAGE-1 REPRODUCTION AND INTEGRITY GATES")

cnn_unique = cnn["chunk_id"].nunique()
cnn_class_counts = cnn["y_true"].value_counts().to_dict()

print(
    f"CNN unique chunks={cnn_unique} "
    f"(expected {EXPECTED_STAGE1['n_chunks']})"
)
print(
    f"CNN class0={int(cnn_class_counts.get(0, 0))} "
    f"(expected {EXPECTED_STAGE1['class0']})"
)
print(
    f"CNN class1={int(cnn_class_counts.get(1, 0))} "
    f"(expected {EXPECTED_STAGE1['class1']})"
)

if cnn_unique != EXPECTED_STAGE1["n_chunks"]:
    raise RuntimeError("Stage-1 CNN chunk-count reproduction gate failed.")
if int(cnn_class_counts.get(0, 0)) != EXPECTED_STAGE1["class0"]:
    raise RuntimeError("Stage-1 CNN class-0 count reproduction gate failed.")
if int(cnn_class_counts.get(1, 0)) != EXPECTED_STAGE1["class1"]:
    raise RuntimeError("Stage-1 CNN class-1 count reproduction gate failed.")

for model, g in trad.groupby("model"):
    n_chunks = g["chunk_id"].nunique()
    n_rows = len(g)
    print(f"{model:<22s} rows={n_rows}, unique chunks={n_chunks}")

    if n_rows != EXPECTED_STAGE1["n_chunks"]:
        raise RuntimeError(
            f"{model}: unexpected Stage-1 outer-test row count."
        )
    if n_chunks != EXPECTED_STAGE1["n_chunks"]:
        raise RuntimeError(
            f"{model}: unexpected Stage-1 unique chunk count."
        )

y = cnn["y_true"].to_numpy(int)
p = cnn["y_pred"].to_numpy(int)

observed_cm = {
    "tn": int(np.sum((y == 0) & (p == 0))),
    "fp": int(np.sum((y == 0) & (p == 1))),
    "fn": int(np.sum((y == 1) & (p == 0))),
    "tp": int(np.sum((y == 1) & (p == 1))),
}

print()
print("CNN pooled confusion:", observed_cm)
print("Expected:", EXPECTED_CNN_POOLED_CM)

if observed_cm != EXPECTED_CNN_POOLED_CM:
    raise RuntimeError(
        "Stage-1 CNN pooled-confusion reproduction gate failed."
    )

print()
print("STAGE-1 REPRODUCTION GATES: PASS")


header("3. STAGE-1 ACTIVITY × CLASS COMPOSITION")

context = (
    cnn.groupby(["activity", "activity_name", "y_true"])
    .agg(
        n_chunks=("chunk_id", "size"),
        n_subjects=("subjectID", "nunique"),
    )
    .reset_index()
    .sort_values(["activity", "y_true"])
)

activity_totals = (
    cnn.groupby(["activity", "activity_name"])
    .size()
    .rename("activity_total_chunks")
    .reset_index()
)

context = context.merge(
    activity_totals,
    on=["activity", "activity_name"],
    how="left",
)

context["within_activity_percent"] = (
    100.0 * context["n_chunks"] / context["activity_total_chunks"]
)

context.to_csv(
    REPORT_DIR / "01_stage1_activity_class_composition.csv",
    index=False,
)

print(
    context.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


header("4. WITHIN-ACTIVITY OUT-OF-SUBJECT PERFORMANCE")

activity_rows = []

for (model, activity, activity_name), g in pred.groupby(
    ["model", "activity", "activity_name"],
    sort=False,
):
    m = metrics_if_valid(
        g["y_true"],
        g["y_pred"],
        g["score"],
    )

    activity_rows.append(
        {
            "model": model,
            "activity": int(activity),
            "activity_name": activity_name,
            "n_subjects": int(g["subjectID"].nunique()),
            **m,
        }
    )

activity_perf = (
    pd.DataFrame(activity_rows)
    .sort_values(["model", "activity"])
    .reset_index(drop=True)
)

activity_perf.to_csv(
    REPORT_DIR / "02_stage1_within_activity_pooled_metrics.csv",
    index=False,
)

print(
    activity_perf[
        [
            "model", "activity_name", "n", "class0_n", "class1_n",
            "n_subjects", "balanced_accuracy", "class0_recall",
            "class1_recall", "roc_auc",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


header("5. ACTIVITY-EQUAL DESCRIPTIVE SUMMARY")

summary_rows = []

for model, g in activity_perf.groupby("model", sort=False):
    valid = g[g["both_classes_present"] == 1].copy()

    if len(valid) == 0:
        activity_equal_ba = np.nan
        activity_equal_auc = np.nan
    else:
        activity_equal_ba = float(valid["balanced_accuracy"].mean())
        activity_equal_auc = float(valid["roc_auc"].mean())

    summary_rows.append(
        {
            "model": model,
            "n_valid_activity_strata": int(len(valid)),
            "activity_equal_balanced_accuracy": activity_equal_ba,
            "activity_equal_roc_auc": activity_equal_auc,
            "min_within_activity_BA": (
                float(valid["balanced_accuracy"].min())
                if len(valid) else np.nan
            ),
            "max_within_activity_BA": (
                float(valid["balanced_accuracy"].max())
                if len(valid) else np.nan
            ),
        }
    )

activity_equal = pd.DataFrame(summary_rows)

activity_equal.to_csv(
    REPORT_DIR / "03_stage1_activity_equal_summary.csv",
    index=False,
)

print(
    activity_equal.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


header("6. FOLD-SPECIFIC WITHIN-ACTIVITY METRICS")

fold_rows = []

for (model, fold, activity, activity_name), g in pred.groupby(
    ["model", "outer_fold", "activity", "activity_name"],
    sort=False,
):
    m = metrics_if_valid(
        g["y_true"],
        g["y_pred"],
        g["score"],
    )

    fold_rows.append(
        {
            "model": model,
            "outer_fold": int(fold),
            "activity": int(activity),
            "activity_name": activity_name,
            "n_subjects": int(g["subjectID"].nunique()),
            **m,
        }
    )

fold_perf = pd.DataFrame(fold_rows)

fold_perf.to_csv(
    PROCESSED_DIR / "step08_stage1_fold_by_activity_metrics.csv",
    index=False,
)

fold_summary_rows = []

for (model, activity, activity_name), g in fold_perf.groupby(
    ["model", "activity", "activity_name"],
    sort=False,
):
    valid = g[g["both_classes_present"] == 1]
    vals = valid["balanced_accuracy"].dropna().to_numpy(float)

    fold_summary_rows.append(
        {
            "model": model,
            "activity": int(activity),
            "activity_name": activity_name,
            "n_outer_folds_total": int(g["outer_fold"].nunique()),
            "n_outer_folds_with_both_classes": int(len(valid)),
            "balanced_accuracy_mean_across_valid_folds": (
                float(np.mean(vals)) if len(vals) else np.nan
            ),
            "balanced_accuracy_sd_across_valid_folds": (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            ),
        }
    )

fold_summary = pd.DataFrame(fold_summary_rows)

fold_summary.to_csv(
    REPORT_DIR / "04_stage1_fold_by_activity_summary.csv",
    index=False,
)

print(
    fold_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


method_lock = {
    "analysis_role": "additional descriptive sensitivity analysis",
    "stage": "Stage 1 locomotor non-FoG vs locomotor FoG",
    "models": EXPECTED_MODELS,
    "activities": ACTIVITY_LABELS,
    "model_retraining": False,
    "retuning": False,
    "threshold_retuning": False,
    "activity_used_as_model_input": False,
    "activity_used_only_for_post_prediction_stratification": True,
    "prediction_source": {
        "traditional_ml": str(TRAD_PRED_PATH),
        "cnn": str(CNN_PRED_PATH),
    },
    "primary_sensitivity_endpoint":
        "within-activity balanced accuracy of frozen outer-test predictions",
    "activity_equal_summary":
        "unweighted mean of pooled within-activity balanced accuracies across activity strata containing both classes",
    "fold_level_summary":
        "descriptive only; reported only where both classes were present within a fold/activity stratum",
    "p_values": False,
}

with open(
    REPORT_DIR / "step08_method_lock.json",
    "w",
) as f:
    json.dump(method_lock, f, indent=2)


header("7. CNN ACTIVITY-CONTEXT SUMMARY")

cnn_activity = (
    activity_perf[
        activity_perf["model"] == "raw_temporal_cnn"
    ]
    .sort_values("activity")
)

for _, r in cnn_activity.iterrows():
    if int(r["both_classes_present"]) == 1:
        print(
            f"{r['activity_name']}: "
            f"N={int(r['n'])} chunks "
            f"(non-FoG={int(r['class0_n'])}, FoG={int(r['class1_n'])}); "
            f"BA={r['balanced_accuracy']:.3f}; "
            f"AUC={r['roc_auc']:.3f}; "
            f"recall non-FoG={r['class0_recall']:.3f}; "
            f"recall FoG={r['class1_recall']:.3f}."
        )
    else:
        print(
            f"{r['activity_name']}: "
            f"N={int(r['n'])} chunks "
            f"(non-FoG={int(r['class0_n'])}, FoG={int(r['class1_n'])}); "
            "both classes not present, so BA/AUC not defined."
        )

cnn_equal = activity_equal[
    activity_equal["model"] == "raw_temporal_cnn"
].iloc[0]

print()
print(
    "CNN activity-equal descriptive BA="
    f"{cnn_equal['activity_equal_balanced_accuracy']:.3f}"
)
print(
    "CNN within-activity BA range="
    f"{cnn_equal['min_within_activity_BA']:.3f}–"
    f"{cnn_equal['max_within_activity_BA']:.3f}"
)

print()
print("Step 08 sensitivity analysis: PASS")
