#!/usr/bin/env python3
"""
Step 04 — Raw temporal CNN benchmark.

Trains the fixed 1D-CNN with five seeds and evaluates probability-level seed ensembles on frozen outer folds.
"""

from pathlib import Path
import copy
import json
import math
import random
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "\nPyTorch is required for Step 04 but is not installed.\n"
        "Install the packages listed in requirements.txt and rerun.\n"
        "For a standard environment:\n"
        "  python -m pip install torch\n"
    ) from exc


# =============================================================================
# CONFIGURATION — FREEZE BEFORE RUNNING
# =============================================================================

FS = 60.0
CHUNK_SEC = 1.0
CHUNK_SAMPLES = 60

LOCOMOTOR_ACTIVITY_CODES = {1, 6, 7}

RIGHT_ANKLE_COLS = [
    "ankleR_acc_x",
    "ankleR_acc_y",
    "ankleR_acc_z",
    "ankleR_gyro_x",
    "ankleR_gyro_y",
    "ankleR_gyro_z",
]

SEEDS = [42, 43, 44, 45, 46]

BATCH_SIZE = 64
SAMPLES_PER_EPOCH = 1024
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20

N_OUTER_FOLDS = 5

STAGE_INFO = {
    "Stage 1": {
        "negative_label": "non_fog",
        "positive_label": "FoG",
        "leaf_probabilities": {
            "non_fog_locomotor": 0.50,
            "shuffling": 1.0 / 6.0,
            "trembling": 1.0 / 6.0,
            "akinesia": 1.0 / 6.0,
        },
    },
    "Stage 2": {
        "negative_label": "kinetic",
        "positive_label": "akinetic",
        "leaf_probabilities": {
            "shuffling": 0.25,
            "trembling": 0.25,
            "akinesia": 0.50,
        },
    },
    "Stage 3": {
        "negative_label": "shuffling",
        "positive_label": "trembling",
        "leaf_probabilities": {
            "shuffling": 0.50,
            "trembling": 0.50,
        },
    },
}


EXPECTED_PARAMETER_COUNT = 11361

EXPECTED_COUNTS = {
    "training_block_rows": 1851,
    "split_manifest_rows": 17800,
    "Stage 1": 3542,
    "Stage 2": 1004,
    "Stage 3": 290,
}

# Frozen CNN reference summary used for reproduction checks.
EXPECTED_CNN_REFERENCE = {
    "Stage 1": {
        "balanced_accuracy_mean": 0.877,
        "balanced_accuracy_sd": 0.034,
        "macro_f1_mean": 0.835,
        "macro_f1_sd": 0.049,
        "roc_auc_mean": 0.934,
        "roc_auc_sd": 0.032,
    },
    "Stage 2": {
        "balanced_accuracy_mean": 0.727,
        "balanced_accuracy_sd": 0.065,
        "macro_f1_mean": 0.684,
        "macro_f1_sd": 0.048,
        "roc_auc_mean": 0.825,
        "roc_auc_sd": 0.053,
    },
    "Stage 3": {
        "balanced_accuracy_mean": 0.656,
        "balanced_accuracy_sd": 0.138,
        "macro_f1_mean": 0.589,
        "macro_f1_sd": 0.171,
        "roc_auc_mean": 0.713,
        "roc_auc_sd": 0.149,
    },
}

# PyTorch/MPS/CUDA kernels can show very small run-to-run or version-level
# numerical variation even with fixed seeds. This gate is intentionally strict
# enough to detect a methodological mismatch but not falsely fail on tiny
# hardware/backend differences.
CNN_REPRODUCTION_TOL = 0.015


# =============================================================================
# PATHS
# =============================================================================

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
        "Could not determine project root. "
        "Run from the repository root or src/."
    )


ROOT = find_project_root()

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

STEP01_DIR = (
    DATA_DIR
    / "processed"
    / "pipeline"
    / "step_01_segments_windows_splits"
)

STEP03_REPORT_DIR = (
    ROOT
    / "reports"
    / "pipeline"
    / "step_03_traditional_ml_benchmark"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
    / "pipeline"
    / "step_04_raw_temporal_cnn_benchmark"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "pipeline"
    / "step_04_raw_temporal_cnn_benchmark"
)


for p in [PROCESSED_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def find_raw_file(filename: str) -> Path:
    preferred = RAW_DIR / filename

    if preferred.exists():
        return preferred

    hits = list(DATA_DIR.rglob(filename))

    if not hits:
        raise FileNotFoundError(
            f"Could not find {filename} under {DATA_DIR}"
        )

    return hits[0]


SENSOR_PATH = find_raw_file("sensor_data.csv")

TRAINING_BLOCKS_PATH = (
    STEP01_DIR
    / "step01_dynamic_training_blocks.csv"
)

SPLIT_MANIFEST_PATH = (
    STEP01_DIR
    / "step01_split_reference_manifest_1s.csv"
)

STEP03_SUMMARY_PATH = (
    STEP03_REPORT_DIR
    / "04_performance_summary.csv"
)


for p in [
    TRAINING_BLOCKS_PATH,
    SPLIT_MANIFEST_PATH,
    STEP03_SUMMARY_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(
            f"Required pipeline file not found:\n{p}"
        )


# =============================================================================
# PRINTING
# =============================================================================

def header(title):
    print("\n" + "=" * 98)
    print(title)
    print("=" * 98)


def subheader(title):
    print("\n" + "-" * 98)
    print(title)
    print("-" * 98)


# =============================================================================
# RECONSTRUCT FROZEN SIGNAL ROW ORDER
# =============================================================================

def add_pool_state(df):
    out = df.copy()

    out["pool_state"] = "unknown"

    out.loc[
        (out["fog"] == 0)
        & out["activity"].isin(
            LOCOMOTOR_ACTIVITY_CODES
        ),
        "pool_state"
    ] = "non_fog_locomotor"

    out.loc[
        (out["fog"] == 0)
        & ~out["activity"].isin(
            LOCOMOTOR_ACTIVITY_CODES
        ),
        "pool_state"
    ] = "excluded_nonlocomotor_nonfog"

    out.loc[
        (out["fog"] == 1)
        & (out["fog_severity"] == 1),
        "pool_state"
    ] = "shuffling"

    out.loc[
        (out["fog"] == 1)
        & (out["fog_severity"] == 2),
        "pool_state"
    ] = "trembling"

    out.loc[
        (out["fog"] == 1)
        & (out["fog_severity"] == 3),
        "pool_state"
    ] = "akinesia"

    return out


def infer_gap_threshold(df):
    diffs = []

    for _, g in df.groupby(
        ["subjectID", "sessionID", "taskID"],
        sort=False,
    ):
        d = (
            g.sort_values("timestamp")["timestamp"]
            .diff()
            .to_numpy(dtype=float)
        )

        d = d[
            np.isfinite(d)
            & (d > 0)
        ]

        if len(d):
            diffs.append(d)

    if not diffs:
        raise RuntimeError(
            "Could not infer timestamp spacing."
        )

    all_d = np.concatenate(diffs)
    median_dt = float(np.median(all_d))

    return median_dt, median_dt * 2.5


def assign_segments(df, gap_threshold):
    pieces = []
    next_segment_id = 0

    for _, g in df.groupby(
        ["subjectID", "sessionID", "taskID"],
        sort=False,
    ):
        g = g.sort_values("timestamp").copy()

        dt = g["timestamp"].diff()

        new_segment = (
            dt.isna()
            | (dt <= 0)
            | (dt > gap_threshold)
            | g["activity"].ne(
                g["activity"].shift()
            )
            | g["pool_state"].ne(
                g["pool_state"].shift()
            )
        )

        local = new_segment.cumsum() - 1

        unique_local = pd.unique(local)

        mapping = {
            old: next_segment_id + i
            for i, old in enumerate(
                unique_local
            )
        }

        g["segment_id"] = (
            local.map(mapping).astype(int)
        )

        next_segment_id += len(
            unique_local
        )

        pieces.append(g)

    out = pd.concat(
        pieces,
        axis=0,
    )

    out = (
        out.sort_values(
            [
                "subjectID",
                "sessionID",
                "taskID",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    out["row_pos"] = np.arange(
        len(out),
        dtype=int,
    )

    return out


# =============================================================================
# DEVICE / REPRODUCIBILITY
# =============================================================================

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = choose_device()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# STAGE LABELS / FILTERS
# =============================================================================

def filter_manifest_for_stage(df, stage):
    if stage == "Stage 1":
        x = df[
            df["activity"].isin(
                LOCOMOTOR_ACTIVITY_CODES
            )
        ].copy()

        x = x[
            x["pool_state"].isin(
                [
                    "non_fog_locomotor",
                    "shuffling",
                    "trembling",
                    "akinesia",
                ]
            )
        ].copy()

        x["y"] = (
            x["pool_state"]
            != "non_fog_locomotor"
        ).astype(int)

        return x

    if stage == "Stage 2":
        x = df[
            df["pool_state"].isin(
                [
                    "shuffling",
                    "trembling",
                    "akinesia",
                ]
            )
        ].copy()

        x["y"] = (
            x["pool_state"]
            == "akinesia"
        ).astype(int)

        return x

    if stage == "Stage 3":
        x = df[
            df["pool_state"].isin(
                [
                    "shuffling",
                    "trembling",
                ]
            )
        ].copy()

        x["y"] = (
            x["pool_state"]
            == "trembling"
        ).astype(int)

        return x

    raise ValueError(stage)


def filter_blocks_for_stage(df, stage):
    if stage == "Stage 1":
        return df[
            df["activity"].isin(
                LOCOMOTOR_ACTIVITY_CODES
            )
            & df["pool_state"].isin(
                [
                    "non_fog_locomotor",
                    "shuffling",
                    "trembling",
                    "akinesia",
                ]
            )
        ].copy()

    if stage == "Stage 2":
        return df[
            df["pool_state"].isin(
                [
                    "shuffling",
                    "trembling",
                    "akinesia",
                ]
            )
        ].copy()

    if stage == "Stage 3":
        return df[
            df["pool_state"].isin(
                [
                    "shuffling",
                    "trembling",
                ]
            )
        ].copy()

    raise ValueError(stage)


# =============================================================================
# RAW WINDOW EXTRACTION
# =============================================================================

def extract_manifest_arrays(
    segmented,
    manifest,
):
    X = np.empty(
        (
            len(manifest),
            CHUNK_SAMPLES,
            len(RIGHT_ANKLE_COLS),
        ),
        dtype=np.float32,
    )

    y = manifest[
        "y"
    ].to_numpy(
        dtype=np.int64
    )

    for i, (_, row) in enumerate(
        manifest.iterrows()
    ):
        start = int(
            row["start_row_pos"]
        )

        end = int(
            row["end_row_pos_exclusive"]
        )

        if end - start != CHUNK_SAMPLES:
            raise RuntimeError(
                "Non-60-sample deterministic chunk encountered."
            )

        g = segmented.iloc[
            start:end
        ]

        if len(g) != CHUNK_SAMPLES:
            raise RuntimeError(
                "Deterministic chunk is out of range."
            )

        if (
            int(g["subjectID"].iloc[0])
            != int(row["subjectID"])
        ):
            raise RuntimeError(
                "Subject mismatch in deterministic chunk."
            )

        if g["pool_state"].nunique() != 1:
            raise RuntimeError(
                "State boundary crossed by deterministic chunk."
            )

        arr = g[
            RIGHT_ANKLE_COLS
        ].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                "Missing right-ankle value in deterministic chunk."
            )

        X[i] = arr

    return X, y


# =============================================================================
# SUBJECT-BALANCED TRAIN STANDARDIZATION
# =============================================================================

def subject_balanced_channel_stats(
    signal_array,
    blocks,
):
    """
    Compute channel mean/std using only current inner-training subjects.

    Each subject contributes equally, independent of recording duration.
    Within a subject, all rows contained in the stage-relevant complete blocks
    are used.
    """

    subjects = sorted(
        blocks["subjectID"]
        .astype(int)
        .unique()
        .tolist()
    )

    subject_means = []
    subject_second_moments = []

    for sid in subjects:
        g = blocks[
            blocks["subjectID"] == sid
        ]

        pieces = []

        for _, row in g.iterrows():
            start = int(
                row["start_row_pos"]
            )

            end = int(
                row["end_row_pos_exclusive"]
            )

            pieces.append(
                signal_array[
                    start:end
                ]
            )

        x = np.concatenate(
            pieces,
            axis=0,
        ).astype(np.float64)

        subject_means.append(
            x.mean(axis=0)
        )

        subject_second_moments.append(
            (x ** 2).mean(axis=0)
        )

    mean = np.mean(
        np.stack(subject_means),
        axis=0,
    )

    second = np.mean(
        np.stack(
            subject_second_moments
        ),
        axis=0,
    )

    var = np.maximum(
        second - mean ** 2,
        1e-12,
    )

    std = np.sqrt(var)

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


# =============================================================================
# DYNAMIC SUBJECT-AWARE SAMPLER
# =============================================================================

class DynamicStageSampler:
    def __init__(
        self,
        signal_array,
        blocks,
        stage,
        channel_mean,
        channel_std,
    ):
        self.signal_array = signal_array
        self.blocks = blocks.copy()
        self.stage = stage

        self.channel_mean = (
            np.asarray(
                channel_mean,
                dtype=np.float32,
            )
        )

        self.channel_std = (
            np.asarray(
                channel_std,
                dtype=np.float32,
            )
        )

        self.leaf_probs = (
            STAGE_INFO[
                stage
            ][
                "leaf_probabilities"
            ]
        )

        self.leaves = list(
            self.leaf_probs.keys()
        )

        self.probs = np.asarray(
            [
                self.leaf_probs[
                    leaf
                ]
                for leaf in self.leaves
            ],
            dtype=float,
        )

        self.probs = (
            self.probs
            / self.probs.sum()
        )

        self.index = {}

        for leaf in self.leaves:
            g_leaf = self.blocks[
                self.blocks[
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

            self.index[leaf] = {}

            for sid in subjects:
                g = (
                    g_leaf[
                        g_leaf[
                            "subjectID"
                        ] == sid
                    ]
                    .copy()
                    .reset_index(
                        drop=True
                    )
                )

                weights = g[
                    "n_possible_start_positions"
                ].to_numpy(
                    dtype=float
                )

                weights = (
                    weights
                    / weights.sum()
                )

                self.index[
                    leaf
                ][
                    sid
                ] = (
                    g,
                    weights,
                )

    def sample_epoch(
        self,
        n_samples,
        seed,
    ):
        rng = np.random.default_rng(
            seed
        )

        X = np.empty(
            (
                n_samples,
                CHUNK_SAMPLES,
                len(RIGHT_ANKLE_COLS),
            ),
            dtype=np.float32,
        )

        y = np.empty(
            n_samples,
            dtype=np.int64,
        )

        for i in range(n_samples):
            leaf = rng.choice(
                self.leaves,
                p=self.probs,
            )

            subject_dict = (
                self.index[leaf]
            )

            subjects = list(
                subject_dict.keys()
            )

            sid = int(
                rng.choice(subjects)
            )

            block_df, block_probs = (
                subject_dict[sid]
            )

            block_i = int(
                rng.choice(
                    len(block_df),
                    p=block_probs,
                )
            )

            row = block_df.iloc[
                block_i
            ]

            block_start = int(
                row["start_row_pos"]
            )

            n_possible = int(
                row[
                    "n_possible_start_positions"
                ]
            )

            relative_start = int(
                rng.integers(
                    0,
                    n_possible,
                )
            )

            start = (
                block_start
                + relative_start
            )

            end = (
                start
                + CHUNK_SAMPLES
            )

            chunk = (
                self.signal_array[
                    start:end
                ]
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            if (
                chunk.shape
                != (
                    CHUNK_SAMPLES,
                    len(
                        RIGHT_ANKLE_COLS
                    ),
                )
            ):
                raise RuntimeError(
                    "Dynamic sampling produced invalid chunk shape."
                )

            X[i] = (
                chunk
                - self.channel_mean
            ) / self.channel_std

            if self.stage == "Stage 1":
                y[i] = (
                    0
                    if leaf
                    == "non_fog_locomotor"
                    else 1
                )

            elif self.stage == "Stage 2":
                y[i] = (
                    1
                    if leaf
                    == "akinesia"
                    else 0
                )

            elif self.stage == "Stage 3":
                y[i] = (
                    1
                    if leaf
                    == "trembling"
                    else 0
                )

        return X, y


# =============================================================================
# MODEL
# =============================================================================

class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                6,
                32,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.dropout = nn.Dropout(
            p=DROPOUT
        )

        self.classifier = nn.Linear(
            64,
            1,
        )

    def forward(self, x):
        # input: batch x time x channels
        x = x.transpose(1, 2)

        z = self.features(x)
        z = z.squeeze(-1)
        z = self.dropout(z)

        return self.classifier(
            z
        ).squeeze(-1)


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =============================================================================
# METRICS / PREDICTION
# =============================================================================

def binary_metrics(
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

    out = {
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
        "mcc":
            matthews_corrcoef(
                y_true,
                y_pred,
            ),
    }

    if len(
        np.unique(y_true)
    ) == 2:
        out["roc_auc"] = (
            roc_auc_score(
                y_true,
                y_prob,
            )
        )
    else:
        out["roc_auc"] = np.nan

    return out


@torch.no_grad()
def predict_probabilities(
    model,
    X,
    batch_size=256,
):
    model.eval()

    X_t = torch.from_numpy(
        X.astype(
            np.float32,
            copy=False,
        )
    )

    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=False,
    )

    probs = []

    for (xb,) in loader:
        xb = xb.to(DEVICE)

        logits = model(xb)

        p = torch.sigmoid(
            logits
        )

        probs.append(
            p.detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        probs
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_one_seed(
    sampler,
    X_val,
    y_val,
    seed,
):
    set_seed(seed)

    model = TemporalCNN().to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_epoch = None
    best_val_ba = -np.inf
    best_val_f1 = -np.inf

    patience_counter = 0

    history_rows = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        X_train, y_train = (
            sampler.sample_epoch(
                SAMPLES_PER_EPOCH,
                seed=(
                    seed * 100_000
                    + epoch
                ),
            )
        )

        X_t = torch.from_numpy(
            X_train
        )

        y_t = torch.from_numpy(
            y_train.astype(
                np.float32
            )
        )

        generator = torch.Generator()
        generator.manual_seed(
            seed * 10_000 + epoch
        )

        loader = DataLoader(
            TensorDataset(
                X_t,
                y_t,
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
        )

        model.train()

        losses = []

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            optimizer.step()

            losses.append(
                float(
                    loss.detach()
                    .cpu()
                    .item()
                )
            )

        val_prob = (
            predict_probabilities(
                model,
                X_val,
            )
        )

        val_pred = (
            val_prob >= 0.5
        ).astype(int)

        val_m = binary_metrics(
            y_val,
            val_pred,
            val_prob,
        )

        history_rows.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": float(
                np.mean(losses)
            ),
            "val_balanced_accuracy":
                val_m[
                    "balanced_accuracy"
                ],
            "val_macro_f1":
                val_m["macro_f1"],
            "val_roc_auc":
                val_m["roc_auc"],
        })

        improved = False

        if (
            val_m[
                "balanced_accuracy"
            ]
            > best_val_ba
            + 1e-12
        ):
            improved = True

        elif (
            abs(
                val_m[
                    "balanced_accuracy"
                ]
                - best_val_ba
            )
            <= 1e-12
            and val_m[
                "macro_f1"
            ]
            > best_val_f1
            + 1e-12
        ):
            improved = True

        if improved:
            best_val_ba = (
                val_m[
                    "balanced_accuracy"
                ]
            )

            best_val_f1 = (
                val_m[
                    "macro_f1"
                ]
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:
            patience_counter += 1

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "No best CNN state selected."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_epoch,
        best_val_ba,
        pd.DataFrame(
            history_rows
        ),
    )


# =============================================================================
# SUBJECT-LEVEL METRICS
# =============================================================================

def subject_level_metrics(
    ensemble_predictions,
):
    rows = []

    for (
        stage,
        sid
    ), g in ensemble_predictions.groupby(
        ["stage", "subjectID"]
    ):
        y_true = g[
            "y"
        ].to_numpy(
            dtype=int
        )

        y_pred = g[
            "y_pred"
        ].to_numpy(
            dtype=int
        )

        y_prob = g[
            "y_prob"
        ].to_numpy(
            dtype=float
        )

        classes = np.unique(
            y_true
        )

        row = {
            "stage": stage,
            "subjectID": int(sid),
            "n_chunks": len(g),
            "n_classes_present":
                len(classes),
            "accuracy":
                accuracy_score(
                    y_true,
                    y_pred,
                ),
            "class0_recall":
                (
                    recall_score(
                        y_true,
                        y_pred,
                        pos_label=0,
                        zero_division=0,
                    )
                    if 0 in classes
                    else np.nan
                ),
            "class1_recall":
                (
                    recall_score(
                        y_true,
                        y_pred,
                        pos_label=1,
                        zero_division=0,
                    )
                    if 1 in classes
                    else np.nan
                ),
        }

        if len(classes) == 2:
            row[
                "balanced_accuracy_if_mixed"
            ] = (
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            )

            row[
                "roc_auc_if_mixed"
            ] = roc_auc_score(
                y_true,
                y_prob,
            )

        else:
            row[
                "balanced_accuracy_if_mixed"
            ] = np.nan

            row[
                "roc_auc_if_mixed"
            ] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_subject_metrics(
    subject_df,
):
    rows = []

    for stage, g in subject_df.groupby(
        "stage"
    ):
        mixed = g[
            g["n_classes_present"] == 2
        ]

        row = {
            "stage": stage,
            "n_subjects": len(g),
            "n_mixed_subjects": len(mixed),
            "mean_subject_accuracy": float(
                g["accuracy"].mean()
            ),
            "sd_subject_accuracy": float(
                g["accuracy"].std(
                    ddof=1
                )
            ),
            "mean_mixed_subject_BA":
                (
                    float(
                        mixed[
                            "balanced_accuracy_if_mixed"
                        ].mean()
                    )
                    if len(mixed)
                    else np.nan
                ),
            "sd_mixed_subject_BA":
                (
                    float(
                        mixed[
                            "balanced_accuracy_if_mixed"
                        ].std(
                            ddof=1
                        )
                    )
                    if len(mixed) > 1
                    else np.nan
                ),
        }

        if stage == "Stage 1":
            fog_free = g[
                (g["n_classes_present"] == 1)
                & g[
                    "class0_recall"
                ].notna()
            ]

            if len(fog_free):
                fpr = (
                    1.0
                    - fog_free[
                        "class0_recall"
                    ]
                )

                row[
                    "mean_FPR_in_FoG_free_subjects"
                ] = float(
                    fpr.mean()
                )

                row[
                    "sd_FPR_in_FoG_free_subjects"
                ] = float(
                    fpr.std(
                        ddof=1
                    )
                )

                row[
                    "n_FoG_free_subjects"
                ] = len(
                    fog_free
                )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_ensemble_fold_metrics(
    fold_metrics,
):
    rows = []

    for stage, g in fold_metrics.groupby(
        "stage"
    ):
        row = {
            "stage": stage,
            "n_outer_folds": len(g),
        }

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "class0_recall",
            "class1_recall",
            "roc_auc",
            "mcc",
        ]:
            x = g[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.nanmean(x)
            )

            row[
                f"{metric}_sd"
            ] = float(
                np.nanstd(
                    x,
                    ddof=1,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# REPRODUCTION AUDIT
# =============================================================================

def cnn_reproduction_audit(cnn_summary):
    rows = []

    for stage in ["Stage 1", "Stage 2", "Stage 3"]:
        g = cnn_summary[cnn_summary["stage"] == stage]
        if len(g) != 1:
            raise RuntimeError(f"Missing Step-04 CNN summary for {stage}.")

        r = g.iloc[0]
        for metric, expected in EXPECTED_CNN_REFERENCE[stage].items():
            observed = float(r[metric])
            diff = abs(observed - expected)
            passed = diff <= CNN_REPRODUCTION_TOL

            rows.append({
                "stage": stage,
                "metric": metric,
                "observed": observed,
                "expected_reference": expected,
                "absolute_difference": diff,
                "tolerance": CNN_REPRODUCTION_TOL,
                "pass": int(passed),
            })

            print(
                f"{stage:<8s} {metric:<28s} "
                f"obs={observed:.6f} expected={expected:.6f} "
                f"|Delta|={diff:.6f} "
                f"{'PASS' if passed else 'FAIL'}"
            )

    return pd.DataFrame(rows)


def build_ml_vs_cnn_table(step03_summary, cnn_summary):
    rows = []

    model_labels = {
        "logistic_regression": "Logistic Regression",
        "rbf_svm": "RBF-SVM",
        "random_forest": "Random Forest",
    }

    for _, r in step03_summary.iterrows():
        rows.append({
            "stage": r["stage"],
            "model": model_labels.get(r["model"], r["model"]),
            "input": "9 interpretable biomechanical features",
            "balanced_accuracy_mean": r["balanced_accuracy_mean"],
            "balanced_accuracy_sd": r["balanced_accuracy_sd"],
            "macro_f1_mean": r["macro_f1_mean"],
            "macro_f1_sd": r["macro_f1_sd"],
            "class0_recall_mean": r["class0_recall_mean"],
            "class1_recall_mean": r["class1_recall_mean"],
            "roc_auc_mean": r["roc_auc_mean"],
            "roc_auc_sd": r["roc_auc_sd"],
        })

    for _, r in cnn_summary.iterrows():
        rows.append({
            "stage": r["stage"],
            "model": "Raw temporal 1D-CNN",
            "input": "raw 60x6 right-ankle ACC+GYRO window",
            "balanced_accuracy_mean": r["balanced_accuracy_mean"],
            "balanced_accuracy_sd": r["balanced_accuracy_sd"],
            "macro_f1_mean": r["macro_f1_mean"],
            "macro_f1_sd": r["macro_f1_sd"],
            "class0_recall_mean": r["class0_recall_mean"],
            "class1_recall_mean": r["class1_recall_mean"],
            "roc_auc_mean": r["roc_auc_mean"],
            "roc_auc_sd": r["roc_auc_sd"],
        })

    out = pd.DataFrame(rows)

    stage_order = {
        "Stage 1": 1,
        "Stage 2": 2,
        "Stage 3": 3,
    }

    model_order = {
        "Logistic Regression": 1,
        "RBF-SVM": 2,
        "Random Forest": 3,
        "Raw temporal 1D-CNN": 4,
    }

    out["_stage_order"] = out["stage"].map(stage_order)
    out["_model_order"] = out["model"].map(model_order)

    out = (
        out.sort_values(["_stage_order", "_model_order"])
        .drop(columns=["_stage_order", "_model_order"])
        .reset_index(drop=True)
    )

    return out


def best_traditional_vs_cnn(benchmark):
    rows = []

    for stage in ["Stage 1", "Stage 2", "Stage 3"]:
        g = benchmark[
            benchmark["stage"] == stage
        ]

        traditional = g[
            g["model"] != "Raw temporal 1D-CNN"
        ]

        best = traditional.loc[
            traditional["balanced_accuracy_mean"].idxmax()
        ]

        cnn = g[
            g["model"] == "Raw temporal 1D-CNN"
        ].iloc[0]

        rows.append({
            "stage": stage,
            "best_traditional_model": best["model"],
            "best_traditional_BA": float(best["balanced_accuracy_mean"]),
            "CNN_BA": float(cnn["balanced_accuracy_mean"]),
            "CNN_minus_best_traditional_BA": float(
                cnn["balanced_accuracy_mean"]
                - best["balanced_accuracy_mean"]
            ),
            "best_traditional_AUC": float(best["roc_auc_mean"]),
            "CNN_AUC": float(cnn["roc_auc_mean"]),
            "CNN_minus_best_traditional_AUC": float(
                cnn["roc_auc_mean"]
                - best["roc_auc_mean"]
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

header(
    "STEP 04 — RAW-SIGNAL TEMPORAL CNN BENCHMARK"
)

print("Project root       :", ROOT)
print("Sensor file        :", SENSOR_PATH)
print("Training blocks    :", TRAINING_BLOCKS_PATH)
print("Split manifest     :", SPLIT_MANIFEST_PATH)
print("Processed          :", PROCESSED_DIR)
print("Reports            :", REPORT_DIR)
print("Torch version      :", torch.__version__)
print("Device             :", DEVICE)
print()
print("Frozen architecture:")
print("  Conv1d 6->32 k5")
print("  Conv1d 32->64 k5")
print("  global average pooling")
print("  dropout =", DROPOUT)
print("  binary linear head")
print()
print("Seeds              :", SEEDS)
print("Samples/epoch      :", SAMPLES_PER_EPOCH)
print("Max epochs         :", MAX_EPOCHS)
print("Early-stop patience:", EARLY_STOPPING_PATIENCE)
print("Learning rate      :", LEARNING_RATE)
print("Weight decay       :", WEIGHT_DECAY)


# -----------------------------------------------------------------------------
# 1. LOAD / REBUILD
# -----------------------------------------------------------------------------

header(
    "1. REBUILDING FROZEN SIGNAL ROW ORDER"
)

raw = pd.read_csv(
    SENSOR_PATH
)

required = {
    "timestamp",
    "activity",
    "fog",
    "fog_severity",
    "subjectID",
    "sessionID",
    "taskID",
    *RIGHT_ANKLE_COLS,
}

missing = (
    required
    - set(raw.columns)
)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

raw = add_pool_state(
    raw
)

if (
    raw["pool_state"]
    == "unknown"
).any():
    raise RuntimeError(
        "Unknown state rows detected."
    )

median_dt, gap_threshold = (
    infer_gap_threshold(
        raw
    )
)

segmented = assign_segments(
    raw,
    gap_threshold,
)

signal_array = segmented[
    RIGHT_ANKLE_COLS
].to_numpy(
    dtype=np.float32
)

training_blocks = pd.read_csv(
    TRAINING_BLOCKS_PATH
)

split_manifest = pd.read_csv(
    SPLIT_MANIFEST_PATH
)

print(
    "Rows:",
    len(segmented),
)

print(
    "Training block rows:",
    len(training_blocks),
)

print(
    "Split manifest rows:",
    len(split_manifest),
)

print(
    "Signal reconstruction: PASSED"
)

if len(training_blocks) != EXPECTED_COUNTS["training_block_rows"]:
    raise RuntimeError(
        f"Training-block row mismatch: {len(training_blocks)} vs "
        f"{EXPECTED_COUNTS['training_block_rows']}"
    )

if len(split_manifest) != EXPECTED_COUNTS["split_manifest_rows"]:
    raise RuntimeError(
        f"Split-manifest row mismatch: {len(split_manifest)} vs "
        f"{EXPECTED_COUNTS['split_manifest_rows']}"
    )

print("Step-01 row-count audit: PASS")


# Stage-specific unique deterministic test/validation universe audit.
for stage in ["Stage 1", "Stage 2", "Stage 3"]:
    x = filter_manifest_for_stage(
        split_manifest,
        stage,
    )

    n_unique = int(x["chunk_id"].nunique())
    expected = EXPECTED_COUNTS[stage]

    print(
        f"{stage}: {n_unique} unique deterministic chunks "
        f"/ expected {expected} "
        f"{'PASS' if n_unique == expected else 'FAIL'}"
    )

    if n_unique != expected:
        raise RuntimeError(
            f"{stage}: deterministic chunk universe mismatch."
        )


# -----------------------------------------------------------------------------
# 2. MODEL PARAMETER COUNT
# -----------------------------------------------------------------------------

model_probe = TemporalCNN()

parameter_count = count_parameters(
    model_probe
)

header(
    "2. FIXED TEMPORAL CNN"
)

print(
    "Trainable parameters:",
    f"{parameter_count:,}",
)

if parameter_count != EXPECTED_PARAMETER_COUNT:
    raise RuntimeError(
        f"CNN parameter-count mismatch: {parameter_count} vs "
        f"{EXPECTED_PARAMETER_COUNT}."
    )

print("Frozen CNN parameter-count audit: PASS")

del model_probe


# -----------------------------------------------------------------------------
# 3. RUN OUTER CV
# -----------------------------------------------------------------------------

header(
    "3. FIVE-FOLD SUBJECT-INDEPENDENT TEMPORAL CNN"
)

seed_metric_rows = []
seed_prediction_frames = []
history_frames = []
fold_ensemble_rows = []
fold_ensemble_prediction_frames = []

for outer_fold in range(
    1,
    N_OUTER_FOLDS + 1,
):
    print()
    print(
        f"OUTER FOLD {outer_fold}"
    )
    print(
        "-" * 50
    )

    fold_blocks = training_blocks[
        training_blocks[
            "outer_fold"
        ] == outer_fold
    ].copy()

    fold_manifest = split_manifest[
        split_manifest[
            "outer_fold"
        ] == outer_fold
    ].copy()

    for stage in [
        "Stage 1",
        "Stage 2",
        "Stage 3",
    ]:
        print(
            f"\n  {stage}"
        )

        stage_blocks = (
            filter_blocks_for_stage(
                fold_blocks,
                stage,
            )
        )

        val_manifest = (
            filter_manifest_for_stage(
                fold_manifest[
                    fold_manifest[
                        "partition"
                    ]
                    == "validation"
                ],
                stage,
            )
        )

        test_manifest = (
            filter_manifest_for_stage(
                fold_manifest[
                    fold_manifest[
                        "partition"
                    ]
                    == "test"
                ],
                stage,
            )
        )

        if (
            val_manifest["y"].nunique()
            != 2
        ):
            raise RuntimeError(
                f"{stage} fold {outer_fold}: validation lacks a class."
            )

        if (
            test_manifest["y"].nunique()
            != 2
        ):
            raise RuntimeError(
                f"{stage} fold {outer_fold}: test lacks a class."
            )

        mean, std = (
            subject_balanced_channel_stats(
                signal_array,
                stage_blocks,
            )
        )

        X_val_raw, y_val = (
            extract_manifest_arrays(
                segmented,
                val_manifest,
            )
        )

        X_test_raw, y_test = (
            extract_manifest_arrays(
                segmented,
                test_manifest,
            )
        )

        X_val = (
            X_val_raw
            - mean
        ) / std

        X_test = (
            X_test_raw
            - mean
        ) / std

        sampler = (
            DynamicStageSampler(
                signal_array=signal_array,
                blocks=stage_blocks,
                stage=stage,
                channel_mean=mean,
                channel_std=std,
            )
        )

        print(
            f"    train subjects={stage_blocks['subjectID'].nunique()}, "
            f"val chunks={len(val_manifest)}, "
            f"test chunks={len(test_manifest)}"
        )

        seed_probs = []

        for seed in SEEDS:
            (
                model,
                best_epoch,
                best_val_ba,
                history,
            ) = train_one_seed(
                sampler=sampler,
                X_val=X_val,
                y_val=y_val,
                seed=seed,
            )

            test_prob = (
                predict_probabilities(
                    model,
                    X_test,
                )
            )

            test_pred = (
                test_prob
                >= 0.5
            ).astype(int)

            m = binary_metrics(
                y_test,
                test_pred,
                test_prob,
            )

            seed_metric_rows.append({
                "outer_fold":
                    outer_fold,
                "stage": stage,
                "seed": seed,
                "best_epoch":
                    best_epoch,
                "best_validation_BA":
                    best_val_ba,
                "n_validation":
                    len(y_val),
                "n_test":
                    len(y_test),
                **m,
            })

            history.insert(
                0,
                "outer_fold",
                outer_fold,
            )

            history.insert(
                1,
                "stage",
                stage,
            )

            history_frames.append(
                history
            )

            pred = test_manifest[
                [
                    "chunk_id",
                    "subjectID",
                    "pool_state",
                    "activity",
                    "taskID",
                    "y",
                ]
            ].copy()

            pred.insert(
                0,
                "outer_fold",
                outer_fold,
            )

            pred.insert(
                1,
                "stage",
                stage,
            )

            pred.insert(
                2,
                "seed",
                seed,
            )

            pred[
                "y_prob"
            ] = test_prob

            pred[
                "y_pred"
            ] = test_pred

            seed_prediction_frames.append(
                pred
            )

            seed_probs.append(
                test_prob
            )

            print(
                f"      seed {seed}: "
                f"epoch={best_epoch:>2}, "
                f"valBA={best_val_ba:.3f}, "
                f"testBA={m['balanced_accuracy']:.3f}, "
                f"AUC={m['roc_auc']:.3f}"
            )

            del model

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        ensemble_prob = (
            np.mean(
                np.stack(
                    seed_probs,
                    axis=0,
                ),
                axis=0,
            )
        )

        ensemble_pred = (
            ensemble_prob
            >= 0.5
        ).astype(int)

        em = binary_metrics(
            y_test,
            ensemble_pred,
            ensemble_prob,
        )

        fold_ensemble_rows.append({
            "outer_fold":
                outer_fold,
            "stage": stage,
            "n_test":
                len(y_test),
            **em,
        })

        ep = test_manifest[
            [
                "chunk_id",
                "subjectID",
                "pool_state",
                "activity",
                "taskID",
                "y",
            ]
        ].copy()

        ep.insert(
            0,
            "outer_fold",
            outer_fold,
        )

        ep.insert(
            1,
            "stage",
            stage,
        )

        ep[
            "y_prob"
        ] = ensemble_prob

        ep[
            "y_pred"
        ] = ensemble_pred

        fold_ensemble_prediction_frames.append(
            ep
        )

        print(
            f"    ENSEMBLE: BA={em['balanced_accuracy']:.3f}, "
            f"F1={em['macro_f1']:.3f}, "
            f"AUC={em['roc_auc']:.3f}"
        )


seed_metrics = pd.DataFrame(
    seed_metric_rows
)

seed_predictions = pd.concat(
    seed_prediction_frames,
    ignore_index=True,
)

history_df = pd.concat(
    history_frames,
    ignore_index=True,
)

ensemble_fold_metrics = pd.DataFrame(
    fold_ensemble_rows
)

ensemble_predictions = pd.concat(
    fold_ensemble_prediction_frames,
    ignore_index=True,
)

seed_metrics.to_csv(
    REPORT_DIR
    / "01_seed_fold_metrics.csv",
    index=False,
)

history_df.to_csv(
    REPORT_DIR
    / "02_training_histories.csv",
    index=False,
)

seed_predictions.to_csv(
    PROCESSED_DIR
    / "step04_seed_level_outer_predictions.csv",
    index=False,
)

ensemble_fold_metrics.to_csv(
    REPORT_DIR
    / "03_seed_ensemble_outer_fold_metrics.csv",
    index=False,
)

ensemble_predictions.to_csv(
    PROCESSED_DIR
    / "step04_seed_ensemble_outer_predictions.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# 4. SUMMARY
# -----------------------------------------------------------------------------

header(
    "4. SEED-ENSEMBLE OUTER-FOLD SUMMARY"
)

cnn_summary = (
    summarize_ensemble_fold_metrics(
        ensemble_fold_metrics
    )
)

cnn_summary.to_csv(
    REPORT_DIR
    / "04_cnn_performance_summary.csv",
    index=False,
)

display_cols = [
    "stage",
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
    cnn_summary[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 5. TRAINING STOCHASTICITY
# -----------------------------------------------------------------------------

header(
    "5. TRAINING-SEED VARIABILITY"
)

seed_fold_summary = (
    seed_metrics.groupby(
        [
            "outer_fold",
            "stage",
        ]
    )
    .agg(
        seed_BA_mean=(
            "balanced_accuracy",
            "mean",
        ),
        seed_BA_sd=(
            "balanced_accuracy",
            "std",
        ),
        seed_AUC_mean=(
            "roc_auc",
            "mean",
        ),
        seed_AUC_sd=(
            "roc_auc",
            "std",
        ),
        mean_best_epoch=(
            "best_epoch",
            "mean",
        ),
    )
    .reset_index()
)

seed_fold_summary.to_csv(
    REPORT_DIR
    / "05_seed_variability_by_fold.csv",
    index=False,
)

print(
    seed_fold_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 6. SUBJECT LEVEL
# -----------------------------------------------------------------------------

header(
    "6. SUBJECT-LEVEL GENERALIZATION"
)

subject_metrics = (
    subject_level_metrics(
        ensemble_predictions
    )
)

subject_metrics.to_csv(
    REPORT_DIR
    / "06_subject_level_metrics.csv",
    index=False,
)

subject_summary = (
    summarize_subject_metrics(
        subject_metrics
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

subheader(
    "Mixed-subject balanced accuracy"
)

print(
    subject_metrics[
        subject_metrics[
            "n_classes_present"
        ] == 2
    ][
        [
            "stage",
            "subjectID",
            "n_chunks",
            "balanced_accuracy_if_mixed",
            "roc_auc_if_mixed",
        ]
    ]
    .sort_values(
        [
            "stage",
            "subjectID",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 7. CNN reproduction gate
# -----------------------------------------------------------------------------

header(
    "7. CNN REPRODUCTION GATE"
)

reproduction = cnn_reproduction_audit(
    cnn_summary
)

reproduction.to_csv(
    REPORT_DIR
    / "08_cnn_reproduction_audit.csv",
    index=False,
)

if not reproduction["pass"].all():
    print()
    print("CNN REPRODUCTION GATE: FAIL")
    raise RuntimeError(
        "Step 04 did not reproduce the frozen CNN reference within the hardware-aware tolerance."
    )

print()
print("CNN REPRODUCTION GATE: PASS")


# -----------------------------------------------------------------------------
# 8. Traditional ML vs CNN benchmark
# -----------------------------------------------------------------------------

header(
    "8. TRADITIONAL ML vs RAW TEMPORAL CNN"
)

step03_summary = pd.read_csv(
    STEP03_SUMMARY_PATH
)

benchmark = build_ml_vs_cnn_table(
    step03_summary,
    cnn_summary,
)

benchmark.to_csv(
    REPORT_DIR
    / "09_traditional_ML_vs_CNN_benchmark.csv",
    index=False,
)

print(
    benchmark[
        [
            "stage",
            "model",
            "balanced_accuracy_mean",
            "balanced_accuracy_sd",
            "macro_f1_mean",
            "roc_auc_mean",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

best_vs_cnn = best_traditional_vs_cnn(
    benchmark
)

best_vs_cnn.to_csv(
    REPORT_DIR
    / "10_CNN_vs_best_traditional_by_stage.csv",
    index=False,
)

subheader(
    "CNN minus best traditional model"
)

print(
    best_vs_cnn.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 9. FINAL REPORT / ANALYSIS LOCK
# -----------------------------------------------------------------------------

header(
    "9. STEP 04 ANALYSIS LOCK"
)

design = {
    "step": "04",
    "model": {
        "type": "1D temporal CNN",
        "trainable_parameters": parameter_count,
        "architecture": [
            "Conv1d(6,32,kernel=5,padding=2)",
            "ReLU",
            "Conv1d(32,64,kernel=5,padding=2)",
            "ReLU",
            "AdaptiveAvgPool1d(1)",
            f"Dropout({DROPOUT})",
            "Linear(64,1)",
        ],
    },
    "training": {
        "seeds": SEEDS,
        "batch_size": BATCH_SIZE,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "sampling": "leaf -> subject -> block weighted by valid starts -> random start",
        "normalization": "subject-balanced channel statistics from inner-training blocks only",
    },
    "evaluation": {
        "outer_folds": 5,
        "seed_ensemble": "mean probability across five seeds",
        "threshold": 0.5,
        "primary_metric": "balanced_accuracy",
        "validation_test_distribution": "natural deterministic Step-01 reference chunks",
    },
    "reproduction_tolerance": CNN_REPRODUCTION_TOL,
    "post_test_policy": "no retuning",
}

with open(
    REPORT_DIR / "step04_analysis_lock.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(design, f, indent=2)


# -----------------------------------------------------------------------------
# FINAL
# -----------------------------------------------------------------------------

print("Step 04 reproduction checks: PASS")
