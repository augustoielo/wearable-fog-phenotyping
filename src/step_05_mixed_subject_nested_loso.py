#!/usr/bin/env python3
"""
Step 05 — Mixed-subject nested LOSO sensitivity analysis.

Evaluates phenotype discrimination in participants exhibiting both target classes.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

TEMPORAL_FEATURES = [
    "acc_mag_std",
    "acc_jerk_vector_rms",
    "gyro_mag_std",
    "gyro_angacc_vector_rms",
]

SPECTRAL_FEATURES = [
    "acc_rel_power_3_8",
    "acc_freeze_index_log10",
    "acc_spectral_entropy",
    "gyro_rel_power_3_8",
    "gyro_spectral_entropy",
]

COMBINED_FEATURES = (
    TEMPORAL_FEATURES
    + SPECTRAL_FEATURES
)

FEATURE_SETS = {
    "temporal": TEMPORAL_FEATURES,
    "spectral": SPECTRAL_FEATURES,
    "combined": COMBINED_FEATURES,
}

C_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

RANDOM_STATE = 20260907

MIXED_SUBJECTS = {
    "Stage 2": [
        1, 2, 3, 9, 11, 12, 19, 21,
    ],
    "Stage 3": [
        9, 12, 13, 20, 21,
    ],
}

LEAF_TARGET_MASS = {
    "Stage 2": {
        "shuffling": 0.25,
        "trembling": 0.25,
        "akinesia": 0.50,
    },
    "Stage 3": {
        "shuffling": 0.50,
        "trembling": 0.50,
    },
}

EXPECTED_COVERAGE = {
    "Stage 2": {
        "n_subjects": 8,
        "n_chunks": 878,
        "class0_chunks": 169,
        "class1_chunks": 709,
    },
    "Stage 3": {
        "n_subjects": 5,
        "n_chunks": 112,
        "class0_chunks": 68,
        "class1_chunks": 44,
    },
}

# Frozen subject-equal outer-LOSO reference summary.
EXPECTED_SUBJECT_EQUAL = {
    ("Stage 2", "combined"): {
        "balanced_accuracy_mean": 0.634349,
        "balanced_accuracy_sd": 0.110004,
        "macro_f1_mean": 0.585582,
        "roc_auc_mean": 0.672908,
    },
    ("Stage 2", "spectral"): {
        "balanced_accuracy_mean": 0.637801,
        "balanced_accuracy_sd": 0.131509,
        "roc_auc_mean": 0.650562,
    },
    ("Stage 2", "temporal"): {
        "balanced_accuracy_mean": 0.631388,
        "balanced_accuracy_sd": 0.097494,
        "roc_auc_mean": 0.685635,
    },
    ("Stage 3", "combined"): {
        "balanced_accuracy_mean": 0.597671,
        "balanced_accuracy_sd": 0.141778,
        "roc_auc_mean": 0.734040,
    },
    ("Stage 3", "spectral"): {
        "balanced_accuracy_mean": 0.571679,
        "balanced_accuracy_sd": 0.182607,
        "roc_auc_mean": 0.596203,
    },
    ("Stage 3", "temporal"): {
        "balanced_accuracy_mean": 0.742485,
        "balanced_accuracy_sd": 0.107020,
        "roc_auc_mean": 0.741983,
    },
}

EXPECTED_POOLED_BA = {
    ("Stage 2", "combined"): 0.743330,
    ("Stage 2", "spectral"): 0.606463,
    ("Stage 2", "temporal"): 0.751671,
    ("Stage 3", "combined"): 0.614305,
    ("Stage 3", "spectral"): 0.591578,
    ("Stage 3", "temporal"): 0.649733,
}

REPRODUCTION_TOL = 8e-4


# =============================================================================
# PATHS
# =============================================================================

def find_project_root() -> Path:
    cwd = Path.cwd().resolve()

    if (
        (cwd / "data").exists()
        and (cwd / "src").exists()
    ):
        return cwd

    if (
        cwd.name.lower() == "src"
        and (cwd.parent / "data").exists()
    ):
        return cwd.parent

    try:
        p = Path(__file__).resolve().parent

        if p.name.lower() == "src":
            return p.parent
    except NameError:
        pass

    raise RuntimeError(
        "Could not determine project root. "
        "Run from the repository root or src/."
    )


ROOT = find_project_root()

STEP02_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pipeline"
    / "step_02_biomechanical_features"
)

FEATURE_PATH = (
    STEP02_DIR
    / "step02_reference_features_9.csv"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pipeline"
    / "step_05_mixed_subject_nested_loso"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "pipeline"
    / "step_05_mixed_subject_nested_loso"
)


for p in [
    PROCESSED_DIR,
    REPORT_DIR,
]:
    p.mkdir(
        parents=True,
        exist_ok=True,
    )

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Required Step-02 feature table not found:\n{FEATURE_PATH}"
    )


# =============================================================================
# PRINT HELPERS
# =============================================================================

def header(title):
    print(
        "\n"
        + "=" * 102
    )
    print(title)
    print(
        "=" * 102
    )


def subheader(title):
    print(
        "\n"
        + "-" * 102
    )
    print(title)
    print(
        "-" * 102
    )


# =============================================================================
# DATA ASSEMBLY
# =============================================================================

def make_stage_table(
    feature_df,
    stage,
):
    mixed = MIXED_SUBJECTS[
        stage
    ]

    if stage == "Stage 2":
        x = feature_df[
            feature_df[
                "subjectID"
            ].isin(
                mixed
            )
            & feature_df[
                "pool_state"
            ].isin(
                [
                    "shuffling",
                    "trembling",
                    "akinesia",
                ]
            )
        ].copy()

        x["y"] = (
            x[
                "pool_state"
            ]
            == "akinesia"
        ).astype(int)

    elif stage == "Stage 3":
        x = feature_df[
            feature_df[
                "subjectID"
            ].isin(
                mixed
            )
            & feature_df[
                "pool_state"
            ].isin(
                [
                    "shuffling",
                    "trembling",
                ]
            )
        ].copy()

        x["y"] = (
            x[
                "pool_state"
            ]
            == "trembling"
        ).astype(int)

    else:
        raise ValueError(
            stage
        )

    return (
        x.sort_values(
            [
                "subjectID",
                "chunk_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# WEIGHTING / SCALING
# =============================================================================

def make_subject_aware_weights(
    train_df,
    stage,
):
    """
    Weighting hierarchy:
      target leaf mass
        -> equal mass across subjects exhibiting leaf
        -> equal mass across subject/leaf chunks
      -> normalize sample weights to mean 1
    """
    target = LEAF_TARGET_MASS[
        stage
    ]

    w = pd.Series(
        0.0,
        index=train_df.index,
        dtype=float,
    )

    for leaf, target_mass in (
        target.items()
    ):
        g_leaf = train_df[
            train_df[
                "pool_state"
            ] == leaf
        ]

        subjects = sorted(
            g_leaf[
                "subjectID"
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        if not subjects:
            raise RuntimeError(
                f"{stage}: no training subject for leaf {leaf}."
            )

        subject_mass = (
            float(target_mass)
            / len(subjects)
        )

        for sid in subjects:
            idx = g_leaf[
                g_leaf[
                    "subjectID"
                ].astype(int)
                == int(sid)
            ].index

            if len(idx) == 0:
                continue

            w.loc[
                idx
            ] = (
                subject_mass
                / len(idx)
            )

    if (
        w <= 0
    ).any():
        raise RuntimeError(
            f"{stage}: non-positive sample weight."
        )

    w = (
        w
        / w.mean()
    )

    return w.to_numpy(
        dtype=float
    )


class WeightedStandardizer:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(
        self,
        X,
        w,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        w = np.asarray(
            w,
            dtype=float,
        )

        sw = float(
            np.sum(w)
        )

        self.mean_ = np.sum(
            X
            * w[:, None],
            axis=0,
        ) / sw

        var = np.sum(
            (
                X
                - self.mean_
            ) ** 2
            * w[:, None],
            axis=0,
        ) / sw

        self.scale_ = np.sqrt(
            np.maximum(
                var,
                1e-12,
            )
        )

        return self

    def transform(
        self,
        X,
    ):
        return (
            np.asarray(
                X,
                dtype=float,
            )
            - self.mean_
        ) / self.scale_


# =============================================================================
# MODEL / METRICS
# =============================================================================

def make_model(
    C,
):
    return LogisticRegression(
        C=float(C),
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def fit_predict(
    train_df,
    eval_df,
    stage,
    features,
    C,
):
    w = make_subject_aware_weights(
        train_df,
        stage,
    )

    X_train = train_df[
        features
    ].to_numpy(
        dtype=float,
    )

    X_eval = eval_df[
        features
    ].to_numpy(
        dtype=float,
    )

    scaler = (
        WeightedStandardizer()
        .fit(
            X_train,
            w,
        )
    )

    Xtr = scaler.transform(
        X_train
    )

    Xev = scaler.transform(
        X_eval
    )

    model = make_model(
        C
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
        )

        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*penalty.*deprecated.*",
        )

        model.fit(
            Xtr,
            train_df[
                "y"
            ].to_numpy(
                dtype=int,
            ),
            sample_weight=w,
        )

    prob = model.predict_proba(
        Xev
    )[:, 1]

    pred = (
        prob
        >= 0.5
    ).astype(int)

    return (
        pred,
        prob,
    )


def metrics(
    y_true,
    y_pred,
    y_prob,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),
        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
        "class0_recall":
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            ),
        "class1_recall":
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            ),
        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob,
            ),
    }


# =============================================================================
# NESTED LOSO
# =============================================================================

def select_C_inner_loso(
    outer_train,
    stage,
    features,
):
    subjects = sorted(
        outer_train[
            "subjectID"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    candidate_rows = []

    for C in C_GRID:
        inner_rows = []

        for inner_val_sid in subjects:
            inner_train = outer_train[
                outer_train[
                    "subjectID"
                ] != inner_val_sid
            ].copy()

            inner_val = outer_train[
                outer_train[
                    "subjectID"
                ] == inner_val_sid
            ].copy()

            if (
                inner_train[
                    "y"
                ].nunique()
                != 2
                or inner_val[
                    "y"
                ].nunique()
                != 2
            ):
                raise RuntimeError(
                    f"{stage}: inner LOSO lost a class for subject "
                    f"{inner_val_sid}."
                )

            pred, prob = fit_predict(
                train_df=inner_train,
                eval_df=inner_val,
                stage=stage,
                features=features,
                C=C,
            )

            m = metrics(
                inner_val[
                    "y"
                ],
                pred,
                prob,
            )

            inner_rows.append({
                "inner_validation_subject":
                    int(
                        inner_val_sid
                    ),
                **m,
            })

        inner_df = pd.DataFrame(
            inner_rows
        )

        candidate_rows.append({
            "C":
                float(C),
            "mean_inner_subject_BA":
                float(
                    inner_df[
                        "balanced_accuracy"
                    ].mean()
                ),
            "mean_inner_subject_macro_f1":
                float(
                    inner_df[
                        "macro_f1"
                    ].mean()
                ),
            "mean_inner_subject_AUC":
                float(
                    inner_df[
                        "roc_auc"
                    ].mean()
                ),
        })

    candidates = pd.DataFrame(
        candidate_rows
    )

    # Frozen selection:
    # highest subject-equal mean BA,
    # macro-F1 tie,
    # smaller C tie.
    selected = (
        candidates.sort_values(
            [
                "mean_inner_subject_BA",
                "mean_inner_subject_macro_f1",
                "C",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return (
        float(
            selected[
                "C"
            ]
        ),
        candidates,
    )


def run_nested_loso(
    stage_df,
    stage,
    feature_set,
    features,
):
    subjects = sorted(
        stage_df[
            "subjectID"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    subject_rows = []
    prediction_rows = []
    selection_rows = []

    for outer_test_sid in subjects:
        outer_train = stage_df[
            stage_df[
                "subjectID"
            ] != outer_test_sid
        ].copy()

        outer_test = stage_df[
            stage_df[
                "subjectID"
            ] == outer_test_sid
        ].copy()

        selected_C, candidates = (
            select_C_inner_loso(
                outer_train=outer_train,
                stage=stage,
                features=features,
            )
        )

        candidates.insert(
            0,
            "outer_test_subject",
            int(
                outer_test_sid
            ),
        )

        candidates.insert(
            0,
            "feature_set",
            feature_set,
        )

        candidates.insert(
            0,
            "stage",
            stage,
        )

        candidates[
            "selected"
        ] = (
            candidates[
                "C"
            ]
            == selected_C
        ).astype(int)

        selection_rows.append(
            candidates
        )

        pred, prob = fit_predict(
            train_df=outer_train,
            eval_df=outer_test,
            stage=stage,
            features=features,
            C=selected_C,
        )

        m = metrics(
            outer_test[
                "y"
            ],
            pred,
            prob,
        )

        subject_rows.append({
            "stage":
                stage,
            "feature_set":
                feature_set,
            "outer_test_subject":
                int(
                    outer_test_sid
                ),
            "selected_C":
                selected_C,
            "n_test_chunks":
                len(
                    outer_test
                ),
            "n_class0":
                int(
                    (
                        outer_test[
                            "y"
                        ] == 0
                    ).sum()
                ),
            "n_class1":
                int(
                    (
                        outer_test[
                            "y"
                        ] == 1
                    ).sum()
                ),
            **m,
        })

        o = outer_test[
            [
                "chunk_id",
                "subjectID",
                "pool_state",
                "y",
            ]
        ].copy()

        o[
            "stage"
        ] = stage

        o[
            "feature_set"
        ] = feature_set

        o[
            "selected_C"
        ] = selected_C

        o[
            "y_pred"
        ] = pred

        o[
            "y_prob"
        ] = prob

        prediction_rows.append(
            o
        )

        print(
            f"{stage} | {feature_set:<8s} | "
            f"test subject {outer_test_sid:>2} | "
            f"C={selected_C:<6g} | "
            f"BA={m['balanced_accuracy']:.3f} | "
            f"F1={m['macro_f1']:.3f} | "
            f"AUC={m['roc_auc']:.3f}"
        )

    return (
        pd.DataFrame(
            subject_rows
        ),
        pd.concat(
            prediction_rows,
            ignore_index=True,
        ),
        pd.concat(
            selection_rows,
            ignore_index=True,
        ),
    )


# =============================================================================
# SUMMARIES
# =============================================================================

def summarize_subject_equal(
    subject_results,
):
    rows = []

    for (
        stage,
        feature_set,
    ), g in subject_results.groupby(
        [
            "stage",
            "feature_set",
        ],
        sort=False,
    ):
        row = {
            "stage":
                stage,
            "feature_set":
                feature_set,
            "n_mixed_subjects":
                len(g),
        }

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "class0_recall",
            "class1_recall",
            "roc_auc",
        ]:
            row[
                metric
                + "_mean"
            ] = float(
                g[
                    metric
                ].mean()
            )

            row[
                metric
                + "_sd"
            ] = float(
                g[
                    metric
                ].std(
                    ddof=1
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def summarize_pooled(
    predictions,
):
    rows = []

    for (
        stage,
        feature_set,
    ), g in predictions.groupby(
        [
            "stage",
            "feature_set",
        ],
        sort=False,
    ):
        m = metrics(
            g[
                "y"
            ],
            g[
                "y_pred"
            ],
            g[
                "y_prob"
            ],
        )

        rows.append({
            "stage":
                stage,
            "feature_set":
                feature_set,
            "n_chunks":
                len(g),
            **m,
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# REPRODUCTION GATE
# =============================================================================

def reproduction_audit(
    subject_summary,
    pooled_summary,
):
    rows = []

    for key, expected in (
        EXPECTED_SUBJECT_EQUAL.items()
    ):
        stage, feature_set = key

        r = subject_summary[
            (
                subject_summary[
                    "stage"
                ] == stage
            )
            & (
                subject_summary[
                    "feature_set"
                ] == feature_set
            )
        ].iloc[0]

        for metric, exp in (
            expected.items()
        ):
            obs = float(
                r[
                    metric
                ]
            )

            diff = abs(
                obs
                - exp
            )

            passed = (
                diff
                <= REPRODUCTION_TOL
            )

            rows.append({
                "summary_type":
                    "subject_equal",
                "stage":
                    stage,
                "feature_set":
                    feature_set,
                "metric":
                    metric,
                "observed":
                    obs,
                "expected":
                    exp,
                "absolute_difference":
                    diff,
                "tolerance":
                    REPRODUCTION_TOL,
                "pass":
                    int(
                        passed
                    ),
            })

            print(
                f"subject-equal | {stage:<7s} | {feature_set:<8s} | "
                f"{metric:<28s} obs={obs:.6f} "
                f"expected={exp:.6f} "
                f"{'PASS' if passed else 'FAIL'}"
            )

    for key, exp in (
        EXPECTED_POOLED_BA.items()
    ):
        stage, feature_set = key

        r = pooled_summary[
            (
                pooled_summary[
                    "stage"
                ] == stage
            )
            & (
                pooled_summary[
                    "feature_set"
                ] == feature_set
            )
        ].iloc[0]

        obs = float(
            r[
                "balanced_accuracy"
            ]
        )

        diff = abs(
            obs
            - exp
        )

        passed = (
            diff
            <= REPRODUCTION_TOL
        )

        rows.append({
            "summary_type":
                "pooled_OOF",
            "stage":
                stage,
            "feature_set":
                feature_set,
            "metric":
                "balanced_accuracy",
            "observed":
                obs,
            "expected":
                exp,
            "absolute_difference":
                diff,
            "tolerance":
                REPRODUCTION_TOL,
            "pass":
                int(
                    passed
                ),
        })

        print(
            f"pooled OOF    | {stage:<7s} | {feature_set:<8s} | "
            f"balanced_accuracy            obs={obs:.6f} "
            f"expected={exp:.6f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

header(
    "STEP 05 — MIXED-SUBJECT NESTED LOSO SENSITIVITY"
)

print(
    "Project root :",
    ROOT,
)

print(
    "Feature table:",
    FEATURE_PATH,
)

print(
    "Processed    :",
    PROCESSED_DIR,
)

print(
    "Reports      :",
    REPORT_DIR,
)


print()
print(
    "This is a sensitivity analysis, not a replacement "
    "for the main five-fold benchmark."
)

print(
    "Model: frozen L2 Logistic Regression only."
)

print(
    "Feature sets: temporal, spectral, combined."
)

print(
    "Nested design: outer LOSO + inner LOSO C selection."
)

print(
    "No p-values."
)


# -----------------------------------------------------------------------------
# 1. Load / coverage
# -----------------------------------------------------------------------------

header(
    "1. MIXED-SUBJECT COVERAGE AUDIT"
)

feature_df = pd.read_csv(
    FEATURE_PATH
)

required = {
    "chunk_id",
    "subjectID",
    "pool_state",
    *COMBINED_FEATURES,
}

missing = (
    required
    - set(
        feature_df.columns
    )
)

if missing:
    raise RuntimeError(
        f"Missing Step-02 columns: {sorted(missing)}"
    )

stage_tables = {}

for stage in [
    "Stage 2",
    "Stage 3",
]:
    x = make_stage_table(
        feature_df,
        stage,
    )

    stage_tables[
        stage
    ] = x

    observed = {
        "n_subjects":
            int(
                x[
                    "subjectID"
                ].nunique()
            ),
        "n_chunks":
            len(x),
        "class0_chunks":
            int(
                (
                    x[
                        "y"
                    ] == 0
                ).sum()
            ),
        "class1_chunks":
            int(
                (
                    x[
                        "y"
                    ] == 1
                ).sum()
            ),
    }

    expected = EXPECTED_COVERAGE[
        stage
    ]

    print(
        f"{stage}: subjects={observed['n_subjects']} "
        f"chunks={observed['n_chunks']} "
        f"class0={observed['class0_chunks']} "
        f"class1={observed['class1_chunks']}"
    )

    if observed != expected:
        raise RuntimeError(
            f"{stage} coverage mismatch:\n"
            f"observed={observed}\n"
            f"expected={expected}"
        )

    actual_subjects = sorted(
        x[
            "subjectID"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    if actual_subjects != MIXED_SUBJECTS[
        stage
    ]:
        raise RuntimeError(
            f"{stage} mixed subject IDs mismatch."
        )

    print(
        "  mixed IDs:",
        actual_subjects,
        "— PASS",
    )


# -----------------------------------------------------------------------------
# 2. Analysis lock
# -----------------------------------------------------------------------------

header(
    "2. STEP 05 ANALYSIS LOCK"
)

lock = {
    "analysis":
        "mixed-subject nested LOSO sensitivity",
    "stages": {
        "Stage 2":
            MIXED_SUBJECTS[
                "Stage 2"
            ],
        "Stage 3":
            MIXED_SUBJECTS[
                "Stage 3"
            ],
    },
    "feature_sets":
        FEATURE_SETS,
    "model":
        "L2 Logistic Regression",
    "solver":
        "liblinear",
    "C_grid":
        C_GRID,
    "inner_selection":
        "mean subject BA, macro-F1 tie, smaller C tie",
    "weights":
        "leaf -> subject -> chunk",
    "scaling":
        "weighted training-only standardization",
    "outer_evaluation":
        "one unseen mixed subject at a time",
    "primary_summary":
        "subject-equal mean +/- SD across held-out subjects",
    "pooled_OOF":
        "secondary",
    "p_values":
        False,
    "post_test_retuning":
        False,
}

with open(
    REPORT_DIR
    / "step05_model_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        lock,
        f,
        indent=2,
    )

print(
    "Analysis lock saved before model execution."
)


# -----------------------------------------------------------------------------
# 3. Nested LOSO
# -----------------------------------------------------------------------------

header(
    "3. NESTED LEAVE-ONE-SUBJECT-OUT"
)

all_subject = []
all_pred = []
all_selection = []

for stage in [
    "Stage 2",
    "Stage 3",
]:
    subheader(
        stage
    )

    for feature_set in [
        "temporal",
        "spectral",
        "combined",
    ]:
        (
            subject_results,
            predictions,
            selections,
        ) = run_nested_loso(
            stage_df=stage_tables[
                stage
            ],
            stage=stage,
            feature_set=feature_set,
            features=FEATURE_SETS[
                feature_set
            ],
        )

        all_subject.append(
            subject_results
        )

        all_pred.append(
            predictions
        )

        all_selection.append(
            selections
        )

subject_results = pd.concat(
    all_subject,
    ignore_index=True,
)

predictions = pd.concat(
    all_pred,
    ignore_index=True,
)

selection_results = pd.concat(
    all_selection,
    ignore_index=True,
)

subject_results.to_csv(
    REPORT_DIR
    / "01_outer_LOSO_subject_metrics.csv",
    index=False,
)

predictions.to_csv(
    PROCESSED_DIR
    / "step05_outer_LOSO_predictions.csv",
    index=False,
)

selection_results.to_csv(
    REPORT_DIR
    / "02_inner_LOSO_C_selection.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# 4. Subject-equal summary
# -----------------------------------------------------------------------------

header(
    "4. SUBJECT-EQUAL OUTER-LOSO SUMMARY"
)

subject_summary = (
    summarize_subject_equal(
        subject_results
    )
)

subject_summary.to_csv(
    REPORT_DIR
    / "03_subject_equal_summary.csv",
    index=False,
)

print(
    subject_summary[
        [
            "stage",
            "feature_set",
            "n_mixed_subjects",
            "balanced_accuracy_mean",
            "balanced_accuracy_sd",
            "macro_f1_mean",
            "roc_auc_mean",
            "roc_auc_sd",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 5. Pooled OOF
# -----------------------------------------------------------------------------

header(
    "5. POOLED OUT-OF-SUBJECT PREDICTIONS — SECONDARY"
)

pooled_summary = summarize_pooled(
    predictions
)

pooled_summary.to_csv(
    REPORT_DIR
    / "04_pooled_OOF_summary.csv",
    index=False,
)

print(
    pooled_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 6. Reproduction gate
# -----------------------------------------------------------------------------

header(
    "6. REPRODUCTION GATE"
)

audit = reproduction_audit(
    subject_summary,
    pooled_summary,
)

audit.to_csv(
    REPORT_DIR
    / "05_nested_loso_reproduction_audit.csv",
    index=False,
)

if not audit[
    "pass"
].all():
    print()
    print(
        "NESTED LOSO REPRODUCTION GATE: FAIL"
    )

    raise RuntimeError(
        "Step 05 did not reproduce the frozen mixed-subject nested LOSO reference results."
    )

print()
print(
    "NESTED LOSO REPRODUCTION GATE: PASS"
)


# -----------------------------------------------------------------------------
# 7. Interpretation lock
# -----------------------------------------------------------------------------

header(
    "7. STEP 05 ANALYSIS LOCK"
)

analysis_lock = {
    "role": "mixed-subject nested LOSO sensitivity analysis",
    "stages": ["Stage 2", "Stage 3"],
    "model": "L2 Logistic Regression",
    "feature_sets": ["combined", "temporal", "spectral"],
    "primary_summary": "subject-equal mean across held-out participants",
    "p_values": False,
    "post_test_policy": "no model or feature-set redefinition after outer-test evaluation",
}

with open(
    REPORT_DIR / "step05_analysis_lock.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(analysis_lock, f, indent=2)


# -----------------------------------------------------------------------------
# FINAL
# -----------------------------------------------------------------------------

print("Step 05 reproduction checks: PASS")
