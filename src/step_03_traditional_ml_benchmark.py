#!/usr/bin/env python3
"""
Step 03 — Traditional machine-learning benchmark.

Evaluates Logistic Regression, RBF-SVM, and Random Forest on the fixed feature representation.
"""

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

FEATURES = [
    "acc_mag_std",
    "acc_jerk_vector_rms",
    "gyro_mag_std",
    "gyro_angacc_vector_rms",
    "acc_rel_power_3_8",
    "acc_freeze_index_log10",
    "acc_spectral_entropy",
    "gyro_rel_power_3_8",
    "gyro_spectral_entropy",
]

STAGES = [
    "Stage 1",
    "Stage 2",
    "Stage 3",
]

STAGE_STEMS = {
    "Stage 1": "stage1",
    "Stage 2": "stage2",
    "Stage 3": "stage3",
}

CLASS_NAMES = {
    "Stage 1": {
        0: "locomotor_non_FoG",
        1: "locomotor_FoG",
    },
    "Stage 2": {
        0: "kinetic_FoG",
        1: "akinetic_FoG",
    },
    "Stage 3": {
        0: "shuffling",
        1: "trembling",
    },
}

# Target total training mass by LEAF manifestation.
# Each target mass is distributed equally across all training windows belonging
# to that leaf, then weights are normalized to mean 1.
LEAF_TARGET_MASS = {
    "Stage 1": {
        "non_fog_locomotor": 0.5,
        "shuffling": 1.0 / 6.0,
        "trembling": 1.0 / 6.0,
        "akinesia": 1.0 / 6.0,
    },
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

# -------------------------------------------------------------------------
# Logistic Regression candidate grid.
# -------------------------------------------------------------------------

LOGISTIC_C_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

# -------------------------------------------------------------------------
# NEW secondary benchmark grids — frozen BEFORE outer-test inspection.
# -------------------------------------------------------------------------

SVM_GRID = [
    {"C": 0.1, "gamma": "scale"},
    {"C": 1.0, "gamma": "scale"},
    {"C": 10.0, "gamma": "scale"},
    {"C": 100.0, "gamma": "scale"},
    {"C": 0.1, "gamma": 0.01},
    {"C": 1.0, "gamma": 0.01},
    {"C": 10.0, "gamma": 0.01},
    {"C": 100.0, "gamma": 0.01},
    {"C": 0.1, "gamma": 0.1},
    {"C": 1.0, "gamma": 0.1},
    {"C": 10.0, "gamma": 0.1},
    {"C": 100.0, "gamma": 0.1},
    {"C": 0.1, "gamma": 1.0},
    {"C": 1.0, "gamma": 1.0},
    {"C": 10.0, "gamma": 1.0},
    {"C": 100.0, "gamma": 1.0},
]

# Candidate order is also the final tie-break order:
# more regularized / simpler trees first.
RF_GRID = [
    {
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    {
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_features": 1.0,
    },
    {
        "max_depth": 6,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "max_depth": 6,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    {
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 1.0,
    },
    {
        "max_depth": 12,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "max_depth": 12,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    {
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": 1.0,
    },
    {
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
]

MODEL_RANDOM_STATE = 20260907

RF_N_ESTIMATORS = 500
RF_RANDOM_STATE = 20260909

# Metric-selection numerical tie tolerance.
SELECTION_TOL = 1e-12

# Frozen Logistic Regression reference summary.
# Logistic Regression MUST reproduce these before the new benchmark is accepted.
EXPECTED_LOGISTIC_REFERENCE = {
    "Stage 1": {
        "balanced_accuracy_mean": 0.833245,
        "balanced_accuracy_sd": 0.030614,
        "roc_auc_mean": 0.912147,
    },
    "Stage 2": {
        "balanced_accuracy_mean": 0.730062,
        "balanced_accuracy_sd": 0.043745,
        "roc_auc_mean": 0.816114,
    },
    "Stage 3": {
        "balanced_accuracy_mean": 0.665623,
        "balanced_accuracy_sd": 0.133452,
        "roc_auc_mean": 0.783860,
    },
}

# Different sklearn versions can produce tiny solver-level differences.
# This is still deliberately tight enough to detect a methodological mismatch.
REPRODUCTION_TOL = 7.5e-4

STAGE2_MIXED = [
    1, 2, 3, 9, 11, 12, 19, 21,
]

STAGE3_MIXED = [
    9, 12, 13, 20, 21,
]

NO_OBSERVED_FOG_SUBJECTS = [
    6, 7, 8, 10, 14, 18,
]


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

INPUT_PATHS = {
    stage:
        STEP02_DIR
        / (
            "step02_"
            + STAGE_STEMS[stage]
            + "_modeling_features.csv"
        )
    for stage in STAGES
}

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pipeline"
    / "step_03_traditional_ml_benchmark"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "pipeline"
    / "step_03_traditional_ml_benchmark"
)


for p in [
    PROCESSED_DIR,
    REPORT_DIR,
]:
    p.mkdir(
        parents=True,
        exist_ok=True,
    )

for stage, p in INPUT_PATHS.items():
    if not p.exists():
        raise FileNotFoundError(
            f"{stage} Step-02 modeling table not found:\n{p}\n"
            "Run Step 02 successfully first."
        )


# =============================================================================
# PRINT HELPERS
# =============================================================================

def header(title):
    print(
        "\n"
        + "=" * 104
    )

    print(title)

    print(
        "=" * 104
    )


def subheader(title):
    print(
        "\n"
        + "-" * 104
    )

    print(title)

    print(
        "-" * 104
    )


# =============================================================================
# WEIGHTED STANDARDIZATION
# =============================================================================

class WeightedStandardizer:
    """
    Weighted training-only z standardization.

    Mean:
        sum(w*x) / sum(w)

    Population variance:
        sum(w*(x-mu)^2) / sum(w)

    Validation/test observations never contribute to mean or scale.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(
        self,
        X,
        sample_weight,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        w = np.asarray(
            sample_weight,
            dtype=float,
        )

        if (
            X.ndim != 2
            or len(X) != len(w)
        ):
            raise ValueError(
                "WeightedStandardizer shape mismatch."
            )

        wsum = float(
            np.sum(w)
        )

        if wsum <= 0:
            raise ValueError(
                "Non-positive weight sum."
            )

        self.mean_ = np.sum(
            X
            * w[:, None],
            axis=0,
        ) / wsum

        var = np.sum(
            (
                X
                - self.mean_
            ) ** 2
            * w[:, None],
            axis=0,
        ) / wsum

        scale = np.sqrt(
            var
        )

        scale[
            ~np.isfinite(scale)
            | (scale <= 0)
        ] = 1.0

        self.scale_ = scale

        return self

    def transform(
        self,
        X,
    ):
        if (
            self.mean_ is None
            or self.scale_ is None
        ):
            raise RuntimeError(
                "Standardizer has not been fit."
            )

        X = np.asarray(
            X,
            dtype=float,
        )

        return (
            X
            - self.mean_
        ) / self.scale_


# =============================================================================
# TRAINING WEIGHTS
# =============================================================================

def make_leaf_weights(
    train_df,
    stage,
):
    """
    Class/leaf/subject-aware weighting.

    Frozen hierarchy:
        desired leaf total mass
          -> equally across subjects exhibiting that leaf
          -> equally across chunks belonging to that subject/leaf.

    Final weights are normalized to mean 1.

    This is intentionally NOT equivalent to assigning one equal weight to every
    chunk in a leaf. Subjects with long recordings must not dominate a leaf.
    """
    target = LEAF_TARGET_MASS[
        stage
    ]

    w = pd.Series(
        0.0,
        index=train_df.index,
        dtype=float,
    )

    audit_rows = []

    for state, target_mass in target.items():
        g_leaf = train_df[
            train_df[
                "pool_state"
            ].astype(str)
            == str(state)
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
                f"{stage}: training fold has no "
                f"windows for required leaf '{state}'."
            )

        per_subject_mass = (
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

            n = len(idx)

            if n <= 0:
                continue

            raw_per_window = (
                per_subject_mass
                / n
            )

            w.loc[
                idx
            ] = raw_per_window

            audit_rows.append({
                "pool_state":
                    state,
                "subjectID":
                    int(sid),
                "n_training_windows":
                    int(n),
                "target_leaf_mass_before_normalization":
                    float(target_mass),
                "n_subjects_in_leaf":
                    int(len(subjects)),
                "subject_mass_before_normalization":
                    float(per_subject_mass),
                "raw_per_window_weight":
                    float(raw_per_window),
            })

    if (
        w <= 0
    ).any():
        bad = train_df.loc[
            w <= 0,
            [
                "subjectID",
                "pool_state",
            ],
        ]

        raise RuntimeError(
            f"{stage}: {len(bad)} training windows "
            "received zero subject-aware weight."
        )

    # Normalize weights to mean 1.
    w = (
        w
        / w.mean()
    )

    audit = pd.DataFrame(
        audit_rows
    )

    if len(audit):
        # Add normalized subject/leaf totals for reproducibility checks.
        normalized = []

        for _, r in audit.iterrows():
            mask = (
                (
                    train_df[
                        "pool_state"
                    ].astype(str)
                    == str(
                        r[
                            "pool_state"
                        ]
                    )
                )
                & (
                    train_df[
                        "subjectID"
                    ].astype(int)
                    == int(
                        r[
                            "subjectID"
                        ]
                    )
                )
            )

            normalized.append(
                float(
                    w.loc[
                        mask
                    ].sum()
                )
            )

        audit[
            "normalized_subject_total_weight"
        ] = normalized

    return (
        w.to_numpy(
            dtype=float
        ),
        audit,
    )


# =============================================================================
# METRICS
# =============================================================================

def binary_metrics(
    y_true,
    y_pred,
    score,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    recall0 = recall_score(
        y_true,
        y_pred,
        labels=[0],
        average=None,
        zero_division=0,
    )[0]

    recall1 = recall_score(
        y_true,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )[0]

    auc = np.nan

    if len(
        np.unique(
            y_true
        )
    ) == 2:
        auc = roc_auc_score(
            y_true,
            score,
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
            float(
                recall0
            ),
        "class1_recall":
            float(
                recall1
            ),
        "roc_auc":
            float(
                auc
            ),
        "tn":
            int(
                cm[0, 0]
            ),
        "fp":
            int(
                cm[0, 1]
            ),
        "fn":
            int(
                cm[1, 0]
            ),
        "tp":
            int(
                cm[1, 1]
            ),
    }


def mean_sd(
    x,
):
    a = np.asarray(
        x,
        dtype=float,
    )

    finite = a[
        np.isfinite(a)
    ]

    if len(finite) == 0:
        return (
            np.nan,
            np.nan,
        )

    if len(finite) == 1:
        return (
            float(finite[0]),
            np.nan,
        )

    return (
        float(
            np.mean(
                finite
            )
        ),
        float(
            np.std(
                finite,
                ddof=1,
            )
        ),
    )


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def make_logistic(
    C,
):
    # Logistic Regression baseline.
    return LogisticRegression(
        C=float(C),
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=MODEL_RANDOM_STATE,
    )


def make_svm(
    params,
):
    return SVC(
        C=float(
            params[
                "C"
            ]
        ),
        kernel="rbf",
        gamma=params[
            "gamma"
        ],
        probability=False,
        shrinking=True,
        tol=1e-3,
        cache_size=1000,
    )


def make_rf(
    params,
):
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        criterion="gini",
        max_depth=params[
            "max_depth"
        ],
        min_samples_leaf=int(
            params[
                "min_samples_leaf"
            ]
        ),
        max_features=params[
            "max_features"
        ],
        bootstrap=True,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )


# =============================================================================
# FIT / SCORE
# =============================================================================

def fit_candidate(
    model_name,
    params,
    X_train,
    y_train,
    w_train,
    X_eval,
):
    """
    Fit one candidate and return predictions/scores.

    Logistic and RBF-SVM:
        weighted standardization.

    Random Forest:
        no standardization (tree splits are invariant to monotonic scaling).
    """
    if model_name in {
        "logistic_regression",
        "rbf_svm",
    }:
        scaler = WeightedStandardizer().fit(
            X_train,
            w_train,
        )

        Xtr = scaler.transform(
            X_train
        )

        Xev = scaler.transform(
            X_eval
        )

    elif model_name == "random_forest":
        scaler = None
        Xtr = X_train
        Xev = X_eval

    else:
        raise ValueError(
            model_name
        )

    if model_name == "logistic_regression":
        model = make_logistic(
            params[
                "C"
            ]
        )

    elif model_name == "rbf_svm":
        model = make_svm(
            params
        )

    else:
        model = make_rf(
            params
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
            y_train,
            sample_weight=w_train,
        )

    if model_name == "logistic_regression":
        # Fixed 0.5 hard-decision rule.
        score = model.predict_proba(
            Xev
        )[:, 1]

        y_pred = (
            score
            >= 0.5
        ).astype(int)

    elif model_name == "rbf_svm":
        score = model.decision_function(
            Xev
        )

        y_pred = model.predict(
            Xev
        ).astype(int)

    else:
        score = model.predict_proba(
            Xev
        )[:, 1]

        y_pred = model.predict(
            Xev
        ).astype(int)

    return (
        model,
        scaler,
        y_pred,
        np.asarray(
            score,
            dtype=float,
        ),
    )


# =============================================================================
# VALIDATION SELECTION
# =============================================================================

def candidate_parameter_list(
    model_name,
):
    if model_name == "logistic_regression":
        return [
            {
                "C": float(C),
            }
            for C in LOGISTIC_C_GRID
        ]

    if model_name == "rbf_svm":
        return [
            dict(x)
            for x in SVM_GRID
        ]

    if model_name == "random_forest":
        return [
            dict(x)
            for x in RF_GRID
        ]

    raise ValueError(
        model_name
    )


def hyperparameter_string(
    params,
):
    return json.dumps(
        params,
        sort_keys=True,
    )


def select_by_validation(
    model_name,
    X_train,
    y_train,
    w_train,
    X_val,
    y_val,
):
    """
    Frozen model-selection hierarchy:
      1. highest validation balanced accuracy
      2. highest validation macro-F1
      3. earlier candidate in the frozen grid

    The outer test data are not touched here.
    """
    candidate_rows = []

    best = None

    for candidate_index, params in enumerate(
        candidate_parameter_list(
            model_name
        )
    ):
        (
            model,
            scaler,
            pred,
            score,
        ) = fit_candidate(
            model_name=model_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_eval=X_val,
        )

        metrics = binary_metrics(
            y_val,
            pred,
            score,
        )

        row = {
            "candidate_index":
                candidate_index,
            "model":
                model_name,
            "params":
                hyperparameter_string(
                    params
                ),
            **metrics,
        }

        candidate_rows.append(
            row
        )

        candidate_key = (
            metrics[
                "balanced_accuracy"
            ],
            metrics[
                "macro_f1"
            ],
            -candidate_index,
        )

        if best is None:
            best = {
                "key":
                    candidate_key,
                "params":
                    params,
                "candidate_index":
                    candidate_index,
                "validation_metrics":
                    metrics,
            }

        else:
            old = best[
                "key"
            ]

            better = False

            if (
                candidate_key[0]
                > old[0]
                + SELECTION_TOL
            ):
                better = True

            elif (
                abs(
                    candidate_key[0]
                    - old[0]
                )
                <= SELECTION_TOL
                and candidate_key[1]
                > old[1]
                + SELECTION_TOL
            ):
                better = True

            elif (
                abs(
                    candidate_key[0]
                    - old[0]
                )
                <= SELECTION_TOL
                and abs(
                    candidate_key[1]
                    - old[1]
                )
                <= SELECTION_TOL
                and candidate_index
                < best[
                    "candidate_index"
                ]
            ):
                better = True

            if better:
                best = {
                    "key":
                        candidate_key,
                    "params":
                        params,
                    "candidate_index":
                        candidate_index,
                    "validation_metrics":
                        metrics,
                }

    return (
        best,
        pd.DataFrame(
            candidate_rows
        ),
    )


# =============================================================================
# ONE STAGE / MODEL
# =============================================================================

def run_stage_model(
    stage,
    model_name,
    df,
):
    fold_rows = []
    selection_rows = []
    prediction_rows = []
    weight_rows = []

    for fold in sorted(
        df[
            "outer_fold"
        ].unique()
    ):
        fold = int(
            fold
        )

        train = df[
            (
                df[
                    "outer_fold"
                ] == fold
            )
            & (
                df[
                    "partition"
                ] == "train"
            )
        ].copy()

        val = df[
            (
                df[
                    "outer_fold"
                ] == fold
            )
            & (
                df[
                    "partition"
                ] == "validation"
            )
        ].copy()

        test = df[
            (
                df[
                    "outer_fold"
                ] == fold
            )
            & (
                df[
                    "partition"
                ] == "test"
            )
        ].copy()

        for name, part in [
            ("train", train),
            ("validation", val),
            ("test", test),
        ]:
            if len(part) == 0:
                raise RuntimeError(
                    f"{stage} fold {fold}: empty {name} partition."
                )

            if part[
                "chunk_id"
            ].duplicated().any():
                raise RuntimeError(
                    f"{stage} fold {fold}: duplicate chunks in {name}."
                )

            if part[
                FEATURES
            ].isna().any().any():
                raise RuntimeError(
                    f"{stage} fold {fold}: missing features in {name}."
                )

        train_subjects = set(
            train[
                "subjectID"
            ].astype(int)
        )

        val_subjects = set(
            val[
                "subjectID"
            ].astype(int)
        )

        test_subjects = set(
            test[
                "subjectID"
            ].astype(int)
        )

        if (
            train_subjects
            & val_subjects
            or train_subjects
            & test_subjects
            or val_subjects
            & test_subjects
        ):
            raise RuntimeError(
                f"{stage} fold {fold}: subject leakage."
            )

        X_train = train[
            FEATURES
        ].to_numpy(
            dtype=float,
        )

        X_val = val[
            FEATURES
        ].to_numpy(
            dtype=float,
        )

        X_test = test[
            FEATURES
        ].to_numpy(
            dtype=float,
        )

        y_train = train[
            "y"
        ].to_numpy(
            dtype=int,
        )

        y_val = val[
            "y"
        ].to_numpy(
            dtype=int,
        )

        y_test = test[
            "y"
        ].to_numpy(
            dtype=int,
        )

        (
            w_train,
            weight_audit,
        ) = make_leaf_weights(
            train,
            stage,
        )

        weight_audit.insert(
            0,
            "outer_fold",
            fold,
        )

        weight_audit.insert(
            0,
            "stage",
            stage,
        )

        weight_audit.insert(
            1,
            "model",
            model_name,
        )

        weight_rows.append(
            weight_audit
        )

        (
            best,
            candidates,
        ) = select_by_validation(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_val=X_val,
            y_val=y_val,
        )

        candidates.insert(
            0,
            "outer_fold",
            fold,
        )

        candidates.insert(
            0,
            "stage",
            stage,
        )

        selection_rows.append(
            candidates
        )

        # Refit the selected hyperparameter on the SAME inner-training data.
        # Validation remains a selection partition; it is NOT merged into model
        # fitting, preserving the frozen protocol.
        (
            model,
            scaler,
            test_pred,
            test_score,
        ) = fit_candidate(
            model_name=model_name,
            params=best[
                "params"
            ],
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_eval=X_test,
        )

        test_metrics = binary_metrics(
            y_test,
            test_pred,
            test_score,
        )

        fold_row = {
            "stage":
                stage,
            "model":
                model_name,
            "outer_fold":
                fold,
            "selected_params":
                hyperparameter_string(
                    best[
                        "params"
                    ]
                ),
            "validation_balanced_accuracy":
                best[
                    "validation_metrics"
                ][
                    "balanced_accuracy"
                ],
            "validation_macro_f1":
                best[
                    "validation_metrics"
                ][
                    "macro_f1"
                ],
            "n_train":
                len(train),
            "n_validation":
                len(val),
            "n_test":
                len(test),
            "n_train_subjects":
                train[
                    "subjectID"
                ].nunique(),
            "n_validation_subjects":
                val[
                    "subjectID"
                ].nunique(),
            "n_test_subjects":
                test[
                    "subjectID"
                ].nunique(),
            **test_metrics,
        }

        fold_rows.append(
            fold_row
        )

        for i, (_, row) in enumerate(
            test.reset_index(
                drop=True
            ).iterrows()
        ):
            prediction_rows.append({
                "stage":
                    stage,
                "model":
                    model_name,
                "outer_fold":
                    fold,
                "chunk_id":
                    int(
                        row[
                            "chunk_id"
                        ]
                    ),
                "subjectID":
                    int(
                        row[
                            "subjectID"
                        ]
                    ),
                "pool_state":
                    row[
                        "pool_state"
                    ],
                "y_true":
                    int(
                        y_test[
                            i
                        ]
                    ),
                "y_pred":
                    int(
                        test_pred[
                            i
                        ]
                    ),
                "score_class1":
                    float(
                        test_score[
                            i
                        ]
                    ),
            })

        print(
            f"{stage} | {model_name:<20s} | fold {fold} | "
            f"selected {hyperparameter_string(best['params'])} | "
            f"val BA={best['validation_metrics']['balanced_accuracy']:.3f} | "
            f"test BA={test_metrics['balanced_accuracy']:.3f} | "
            f"F1={test_metrics['macro_f1']:.3f} | "
            f"AUC={test_metrics['roc_auc']:.3f}"
        )

    return (
        pd.DataFrame(
            fold_rows
        ),
        pd.concat(
            selection_rows,
            ignore_index=True,
        ),
        pd.DataFrame(
            prediction_rows
        ),
        pd.concat(
            weight_rows,
            ignore_index=True,
        ),
    )


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def summarize_fold_performance(
    folds,
):
    rows = []

    for (
        stage,
        model,
    ), g in folds.groupby(
        [
            "stage",
            "model",
        ],
        sort=False,
    ):
        row = {
            "stage":
                stage,
            "model":
                model,
            "n_outer_folds":
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
            mean, sd = mean_sd(
                g[
                    metric
                ]
            )

            row[
                metric
                + "_mean"
            ] = mean

            row[
                metric
                + "_sd"
            ] = sd

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def subject_level_results(
    predictions,
):
    rows = []

    for (
        stage,
        model,
        sid,
    ), g in predictions.groupby(
        [
            "stage",
            "model",
            "subjectID",
        ]
    ):
        # Every subject occurs in exactly one outer test fold.
        if g[
            "outer_fold"
        ].nunique() != 1:
            raise RuntimeError(
                f"{stage}/{model}/subject {sid}: "
                "expected one outer test fold."
            )

        y_true = g[
            "y_true"
        ].to_numpy(
            dtype=int,
        )

        y_pred = g[
            "y_pred"
        ].to_numpy(
            dtype=int,
        )

        score = g[
            "score_class1"
        ].to_numpy(
            dtype=float,
        )

        both = (
            len(
                np.unique(
                    y_true
                )
            )
            == 2
        )

        ba = (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
            if both
            else np.nan
        )

        auc = (
            roc_auc_score(
                y_true,
                score,
            )
            if both
            else np.nan
        )

        row = {
            "stage":
                stage,
            "model":
                model,
            "subjectID":
                int(
                    sid
                ),
            "outer_fold":
                int(
                    g[
                        "outer_fold"
                    ].iloc[0]
                ),
            "n_chunks":
                len(g),
            "n_class0":
                int(
                    np.sum(
                        y_true == 0
                    )
                ),
            "n_class1":
                int(
                    np.sum(
                        y_true == 1
                    )
                ),
            "has_both_classes":
                int(
                    both
                ),
            "accuracy":
                accuracy_score(
                    y_true,
                    y_pred,
                ),
            "balanced_accuracy":
                ba,
            "macro_f1":
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
            "roc_auc":
                auc,
        }

        # Meaningful Stage-1 FPR in subjects whose truth is all non-FoG.
        if (
            stage == "Stage 1"
            and int(
                sid
            )
            in NO_OBSERVED_FOG_SUBJECTS
        ):
            row[
                "stage1_fpr_no_observed_fog"
            ] = float(
                np.mean(
                    y_pred == 1
                )
            )
        else:
            row[
                "stage1_fpr_no_observed_fog"
            ] = np.nan

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def summarize_subject_level(
    subject_df,
):
    rows = []

    for (
        stage,
        model,
    ), g in subject_df.groupby(
        [
            "stage",
            "model",
        ],
        sort=False,
    ):
        acc_mean, acc_sd = mean_sd(
            g[
                "accuracy"
            ]
        )

        mixed = g[
            g[
                "has_both_classes"
            ] == 1
        ]

        mixed_ba_mean, mixed_ba_sd = mean_sd(
            mixed[
                "balanced_accuracy"
            ]
        )

        row = {
            "stage":
                stage,
            "model":
                model,
            "n_subjects":
                len(g),
            "n_mixed_subjects":
                len(mixed),
            "subject_accuracy_mean":
                acc_mean,
            "subject_accuracy_sd":
                acc_sd,
            "mixed_subject_balanced_accuracy_mean":
                mixed_ba_mean,
            "mixed_subject_balanced_accuracy_sd":
                mixed_ba_sd,
        }

        if stage == "Stage 1":
            fpr = g[
                "stage1_fpr_no_observed_fog"
            ].dropna()

            fpr_mean, fpr_sd = mean_sd(
                fpr
            )

            row[
                "n_no_observed_fog_subjects"
            ] = len(fpr)

            row[
                "mean_FPR_no_observed_fog_subjects"
            ] = fpr_mean

            row[
                "sd_FPR_no_observed_fog_subjects"
            ] = fpr_sd

        else:
            row[
                "n_no_observed_fog_subjects"
            ] = np.nan

            row[
                "mean_FPR_no_observed_fog_subjects"
            ] = np.nan

            row[
                "sd_FPR_no_observed_fog_subjects"
            ] = np.nan

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# LOGISTIC REPRODUCTION AUDIT
# =============================================================================

def logistic_reproduction_audit(
    summary,
):
    rows = []

    for stage in STAGES:
        g = summary[
            (
                summary[
                    "stage"
                ] == stage
            )
            & (
                summary[
                    "model"
                ] == "logistic_regression"
            )
        ]

        if len(g) != 1:
            raise RuntimeError(
                f"Missing logistic summary for {stage}."
            )

        r = g.iloc[0]
        expected = EXPECTED_LOGISTIC_REFERENCE[
            stage
        ]

        for metric in [
            "balanced_accuracy_mean",
            "balanced_accuracy_sd",
            "roc_auc_mean",
        ]:
            observed = float(
                r[
                    metric
                ]
            )

            exp = float(
                expected[
                    metric
                ]
            )

            diff = abs(
                observed
                - exp
            )

            passed = (
                diff
                <= REPRODUCTION_TOL
            )

            rows.append({
                "stage":
                    stage,
                "metric":
                    metric,
                "observed":
                    observed,
                "expected_reference":
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
                f"{stage:<8s} {metric:<28s} "
                f"obs={observed:.6f} "
                f"expected={exp:.6f} "
                f"|Δ|={diff:.6f} "
                f"{'PASS' if passed else 'FAIL'}"
            )

    audit = pd.DataFrame(
        rows
    )

    return audit


# =============================================================================
# DESCRIPTIVE MODEL DIFFERENCES
# =============================================================================

def build_paired_model_differences(
    folds,
):
    rows = []

    models = [
        "logistic_regression",
        "rbf_svm",
        "random_forest",
    ]

    for stage in STAGES:
        g = folds[
            folds[
                "stage"
            ] == stage
        ]

        pivot = g.pivot(
            index="outer_fold",
            columns="model",
            values="balanced_accuracy",
        )

        for i in range(
            len(models)
        ):
            for j in range(
                i + 1,
                len(models),
            ):
                a = models[
                    i
                ]

                b = models[
                    j
                ]

                if (
                    a not in pivot.columns
                    or b not in pivot.columns
                ):
                    continue

                diff = (
                    pivot[
                        b
                    ]
                    - pivot[
                        a
                    ]
                )

                rows.append({
                    "stage":
                        stage,
                    "model_A":
                        a,
                    "model_B":
                        b,
                    "difference":
                        "B_minus_A",
                    "mean_paired_BA_difference":
                        float(
                            diff.mean()
                        ),
                    "sd_paired_BA_difference":
                        float(
                            diff.std(
                                ddof=1
                            )
                        ),
                    "folds_favoring_B":
                        int(
                            (
                                diff > 0
                            ).sum()
                        ),
                    "folds_favoring_A":
                        int(
                            (
                                diff < 0
                            ).sum()
                        ),
                    "ties":
                        int(
                            (
                                diff == 0
                            ).sum()
                        ),
                    "n_folds":
                        len(diff),
                })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

header(
    "STEP 03 v2 — TRADITIONAL MACHINE-LEARNING BENCHMARK"
)

print("Project root :", ROOT)
print("Processed    :", PROCESSED_DIR)
print("Reports      :", REPORT_DIR)

print()
print(
    "Frozen features:",
    len(FEATURES),
)

print(
    "Models:"
)

print(
    "  1. Logistic Regression"
)

print(
    "  2. RBF-SVM             — new secondary benchmark"
)

print(
    "  3. Random Forest       — new secondary benchmark"
)

print()
print(
    "Primary metric: balanced accuracy on natural outer-test windows."
)

print(
    "Hyperparameters: selected only on the frozen inner-validation subjects."
)

print(
    "No test-driven retuning and no p-values."
)

print(
    "Training weights: leaf -> subject -> chunk balancing."
)


# -----------------------------------------------------------------------------
# 1. Load and validate
# -----------------------------------------------------------------------------

header(
    "1. INPUT TABLE AUDIT"
)

stage_data = {}

expected_unique = {
    "Stage 1": 3542,
    "Stage 2": 1004,
    "Stage 3": 290,
}

for stage in STAGES:
    df = pd.read_csv(
        INPUT_PATHS[
            stage
        ]
    )

    required = {
        "outer_fold",
        "partition",
        "chunk_id",
        "subjectID",
        "pool_state",
        "y",
        *FEATURES,
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{stage}: missing columns {sorted(missing)}"
        )

    unique_chunks = int(
        df[
            "chunk_id"
        ].nunique()
    )

    if unique_chunks != expected_unique[
        stage
    ]:
        raise RuntimeError(
            f"{stage}: unique chunk mismatch "
            f"{unique_chunks} vs {expected_unique[stage]}"
        )

    if sorted(
        df[
            "outer_fold"
        ].unique()
    ) != [
        1, 2, 3, 4, 5,
    ]:
        raise RuntimeError(
            f"{stage}: invalid outer folds."
        )

    if (
        set(
            df[
                "partition"
            ].unique()
        )
        != {
            "train",
            "validation",
            "test",
        }
    ):
        raise RuntimeError(
            f"{stage}: invalid partition labels."
        )

    if df[
        FEATURES
    ].isna().any().any():
        raise RuntimeError(
            f"{stage}: missing feature values."
        )

    stage_data[
        stage
    ] = df

    print(
        f"{stage}: "
        f"{unique_chunks} unique chunks, "
        f"{df['subjectID'].nunique()} subjects, "
        f"{len(df)} fold-partition rows — PASS"
    )


# -----------------------------------------------------------------------------
# 2. Analysis lock BEFORE model execution
# -----------------------------------------------------------------------------

header(
    "2. BENCHMARK ANALYSIS LOCK"
)

analysis_lock = {
    "features":
        FEATURES,
    "stages":
        {
            "Stage 1":
                "locomotor non-FoG vs locomotor FoG",
            "Stage 2":
                "kinetic vs akinetic FoG",
            "Stage 3":
                "shuffling vs trembling",
        },
    "evaluation":
        "five frozen subject-independent outer folds with frozen inner validation",
    "primary_metric":
        "balanced_accuracy",
    "secondary_metrics": [
        "macro_f1",
        "class0_recall",
        "class1_recall",
        "roc_auc",
        "accuracy",
    ],
    "test_distribution":
        "natural; no downsampling",
    "training_weighting":
        {
            "leaf_target_masses":
                LEAF_TARGET_MASS,
            "subject_balanced_within_leaf":
                True,
            "chunk_balanced_within_subject_leaf":
                True,
            "normalization":
                "final sample weights normalized to mean 1",
        },
    "weighted_scaling":
        "training only for Logistic Regression and RBF-SVM",
    "logistic_grid":
        LOGISTIC_C_GRID,
    "svm_grid":
        SVM_GRID,
    "random_forest": {
        "n_estimators":
            RF_N_ESTIMATORS,
        "random_state":
            RF_RANDOM_STATE,
        "grid":
            RF_GRID,
    },
    "selection_rule":
        "validation balanced accuracy, then validation macro-F1, then earlier frozen candidate",
    "threshold_rule":
        "classifier default hard decision; no validation/test threshold tuning",
    "reproduction_status": {
        "logistic_regression":
            "must reproduce the frozen Logistic Regression reference",
        "rbf_svm":
            "new secondary benchmark",
        "random_forest":
            "new secondary benchmark",
    },
    "post_test_policy":
        "no hyperparameter-grid changes after outer-test inspection",
}

with open(
    REPORT_DIR
    / "step03_model_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        analysis_lock,
        f,
        indent=2,
    )

print(
    "Analysis lock saved BEFORE benchmark execution."
)

print(
    "Logistic C grid:",
    LOGISTIC_C_GRID,
)

print(
    "RBF-SVM candidates:",
    len(
        SVM_GRID
    ),
)

print(
    "Random-Forest candidates:",
    len(
        RF_GRID
    ),
    f"with {RF_N_ESTIMATORS} trees each",
)


# -----------------------------------------------------------------------------
# 3. Execute benchmark
# -----------------------------------------------------------------------------

header(
    "3. INNER-VALIDATION MODEL SELECTION + NATURAL OUTER TEST"
)

all_folds = []
all_selections = []
all_predictions = []
all_weights = []

for stage in STAGES:
    subheader(
        stage
    )

    for model_name in [
        "logistic_regression",
        "rbf_svm",
        "random_forest",
    ]:
        (
            fold_df,
            selection_df,
            pred_df,
            weight_df,
        ) = run_stage_model(
            stage=stage,
            model_name=model_name,
            df=stage_data[
                stage
            ],
        )

        all_folds.append(
            fold_df
        )

        all_selections.append(
            selection_df
        )

        all_predictions.append(
            pred_df
        )

        all_weights.append(
            weight_df
        )

fold_results = pd.concat(
    all_folds,
    ignore_index=True,
)

selection_results = pd.concat(
    all_selections,
    ignore_index=True,
)

predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)

weight_audit = pd.concat(
    all_weights,
    ignore_index=True,
)

fold_results.to_csv(
    REPORT_DIR
    / "01_outer_fold_performance.csv",
    index=False,
)

selection_results.to_csv(
    REPORT_DIR
    / "02_validation_hyperparameter_search.csv",
    index=False,
)

predictions.to_csv(
    PROCESSED_DIR
    / "step03_outer_test_predictions.csv",
    index=False,
)

weight_audit.to_csv(
    REPORT_DIR
    / "03_training_weight_audit.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# 4. Fold summary
# -----------------------------------------------------------------------------

header(
    "4. TRADITIONAL ML PERFORMANCE SUMMARY"
)

summary = summarize_fold_performance(
    fold_results
)

summary.to_csv(
    REPORT_DIR
    / "04_performance_summary.csv",
    index=False,
)

display_cols = [
    "stage",
    "model",
    "balanced_accuracy_mean",
    "balanced_accuracy_sd",
    "macro_f1_mean",
    "macro_f1_sd",
    "class0_recall_mean",
    "class1_recall_mean",
    "roc_auc_mean",
    "roc_auc_sd",
]

print(
    summary[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 5. Logistic Regression reproduction gate
# -----------------------------------------------------------------------------

header(
    "5. LOGISTIC-REGRESSION REPRODUCTION GATE"
)

reproduction = logistic_reproduction_audit(
    summary
)

reproduction.to_csv(
    REPORT_DIR
    / "05_logistic_reproduction_audit.csv",
    index=False,
)

if not reproduction[
    "pass"
].all():
    print()
    print(
        "REPRODUCTION GATE: FAIL"
    )

    raise RuntimeError(
        "Logistic Regression did not reproduce the frozen reference results."
    )

print()
print(
    "REPRODUCTION GATE: PASS"
)

print(
    "Logistic Regression reproduction: PASS"
)


# -----------------------------------------------------------------------------
# 6. Subject-level benchmark
# -----------------------------------------------------------------------------

header(
    "6. SUBJECT-LEVEL PERFORMANCE"
)

subject_df = subject_level_results(
    predictions
)

subject_df.to_csv(
    REPORT_DIR
    / "06_subject_level_performance.csv",
    index=False,
)

subject_summary = (
    summarize_subject_level(
        subject_df
    )
)

subject_summary.to_csv(
    REPORT_DIR
    / "07_subject_level_summary.csv",
    index=False,
)

print(
    subject_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 7. Descriptive paired differences
# -----------------------------------------------------------------------------

header(
    "7. DESCRIPTIVE PAIRED MODEL DIFFERENCES"
)

paired = (
    build_paired_model_differences(
        fold_results
    )
)

paired.to_csv(
    REPORT_DIR
    / "08_paired_model_BA_differences.csv",
    index=False,
)

print(
    paired.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

print()
print(
    "These fold-wise differences are descriptive only; "
    "no p-values are computed."
)


# -----------------------------------------------------------------------------
# 8. Final benchmark table
# -----------------------------------------------------------------------------

header(
    "8. FINAL TRADITIONAL ML TABLE"
)

model_labels = {
    "logistic_regression":
        "Logistic Regression",
    "rbf_svm":
        "RBF-SVM",
    "random_forest":
        "Random Forest",
}

report_rows = []

for _, r in summary.iterrows():
    report_rows.append({
        "Stage":
            r[
                "stage"
            ],
        "Model":
            model_labels[
                r[
                    "model"
                ]
            ],
        "Balanced accuracy":
            (
                f"{r['balanced_accuracy_mean']:.3f} "
                f"± {r['balanced_accuracy_sd']:.3f}"
            ),
        "Macro-F1":
            (
                f"{r['macro_f1_mean']:.3f} "
                f"± {r['macro_f1_sd']:.3f}"
            ),
        "Class 0 recall":
            f"{r['class0_recall_mean']:.3f}",
        "Class 1 recall":
            f"{r['class1_recall_mean']:.3f}",
        "ROC-AUC":
            (
                f"{r['roc_auc_mean']:.3f} "
                f"± {r['roc_auc_sd']:.3f}"
            ),
    })

report_table = pd.DataFrame(
    report_rows
)

report_table.to_csv(
    REPORT_DIR
    / "09_traditional_ML_table.csv",
    index=False,
)

print(
    report_table.to_string(
        index=False
    )
)


# -----------------------------------------------------------------------------
# 9. Final interpretation lock
# -----------------------------------------------------------------------------

header(
    "9. STEP 03 INTERPRETATION LOCK"
)

interpretation = {
    "model_comparison": (
        "descriptive comparison across the same five participant-independent outer folds; "
        "no inferential superiority or equivalence claim"
    ),
    "stage_1": "locomotor non-FoG versus locomotor FoG",
    "stage_2": "kinetic versus akinetic FoG",
    "stage_3": "shuffling versus trembling",
    "post_test_policy": "no model, threshold, split, or sampling retuning after outer-test evaluation",
}

with open(
    REPORT_DIR
    / "step03_analysis_lock.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        interpretation,
        f,
        indent=2,
    )

for key, value in (
    interpretation.items()
):
    print(
        f"- {key}: {value}"
    )


# -----------------------------------------------------------------------------
# FINAL
# -----------------------------------------------------------------------------

print("Step 03 reproduction checks: PASS")
