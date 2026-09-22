#!/usr/bin/env python3
"""
Step 02 — Interpretable biomechanical features.

Extracts the fixed nine-feature representation and participant-level phenotype summaries.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

FS = 60.0
WINDOW_SAMPLES = 60
WINDOW_SEC = 1.0

LOCOMOTOR_ACTIVITY_CODES = {1, 6, 7}

RIGHT_ANKLE_COLS = [
    "ankleR_acc_x",
    "ankleR_acc_y",
    "ankleR_acc_z",
    "ankleR_gyro_x",
    "ankleR_gyro_y",
    "ankleR_gyro_z",
]

ACC_COLS = RIGHT_ANKLE_COLS[:3]
GYRO_COLS = RIGHT_ANKLE_COLS[3:]

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

STATE_ORDER = [
    "non_fog_locomotor",
    "shuffling",
    "trembling",
    "akinesia",
]

ACTIVITY_NAMES = {
    1: "walking",
    2: "sitting",
    3: "standing",
    4: "sit_to_stand",
    5: "stand_to_sit",
    6: "turn_right",
    7: "turn_left",
}

# Fixed mixed-subject identities already established in Step 01.
EXPECTED_STAGE2_MIXED = [1, 2, 3, 9, 11, 12, 19, 21]
EXPECTED_STAGE3_MIXED = [9, 12, 13, 20, 21]

# Frozen numerical sentinels used for reproduction checks.
EXPECTED_SENTINELS = {
    # subject-level median across subjects in each state
    ("non_fog_locomotor", "acc_mag_std"): 0.293895,
    ("shuffling", "acc_mag_std"): 0.151009,
    ("trembling", "acc_mag_std"): 0.175556,
    ("akinesia", "acc_mag_std"): 0.0688491,

    # Stage 2 kinetic-minus-akinetic median paired differences
    ("stage2", "acc_jerk_vector_rms"): 9.88754,
    ("stage2", "gyro_mag_std"): 18.5853,

    # Stage 3 shuffling-minus-trembling median paired difference
    ("stage3", "gyro_mag_std"): 12.0093,
}

SENTINEL_ABS_TOL = 5e-4


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
        "Could not determine project root. Run from the repository root or src/."
    )


ROOT = find_project_root()

SENSOR_PATH = ROOT / "data" / "raw" / "sensor_data.csv"

STEP01_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pipeline"
    / "step_01_segments_windows_splits"
)

REFERENCE_PATH = (
    STEP01_DIR
    / "step01_reference_chunks_1s.csv"
)

STAGE_MANIFEST_PATHS = {
    "Stage 1":
        STEP01_DIR
        / "step01_stage1_split_manifest.csv",
    "Stage 2":
        STEP01_DIR
        / "step01_stage2_split_manifest.csv",
    "Stage 3":
        STEP01_DIR
        / "step01_stage3_split_manifest.csv",
}

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pipeline"
    / "step_02_biomechanical_features"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "pipeline"
    / "step_02_biomechanical_features"
)


for p in [PROCESSED_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


for p in [SENSOR_PATH, REFERENCE_PATH, *STAGE_MANIFEST_PATHS.values()]:
    if not p.exists():
        raise FileNotFoundError(
            f"Required input not found:\n{p}\n"
            "Run Step 01 first."
        )


# =============================================================================
# PRINT HELPERS
# =============================================================================

def header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def subheader(title):
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


# =============================================================================
# REBUILD STEP-01 ROW ORDER
# =============================================================================

def add_pool_state(df):
    out = df.copy()
    out["pool_state"] = "excluded"

    out.loc[
        (out["fog"] == 0)
        & out["activity"].isin(LOCOMOTOR_ACTIVITY_CODES),
        "pool_state",
    ] = "non_fog_locomotor"

    out.loc[
        (out["fog"] == 1)
        & (out["fog_severity"] == 1),
        "pool_state",
    ] = "shuffling"

    out.loc[
        (out["fog"] == 1)
        & (out["fog_severity"] == 2),
        "pool_state",
    ] = "trembling"

    out.loc[
        (out["fog"] == 1)
        & (out["fog_severity"] == 3),
        "pool_state",
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


def assign_homogeneous_segments(df, gap_threshold):
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
            | g["activity"].ne(g["activity"].shift())
            | g["pool_state"].ne(g["pool_state"].shift())
        )

        local = new_segment.cumsum() - 1
        unique_local = pd.unique(local)

        mapping = {
            old: next_segment_id + i
            for i, old in enumerate(unique_local)
        }

        g["segment_id"] = local.map(mapping).astype(int)
        next_segment_id += len(unique_local)
        pieces.append(g)

    out = pd.concat(pieces, axis=0)

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
# FEATURE DEFINITIONS
# =============================================================================

def rms(x):
    x = np.asarray(x, dtype=float)
    return float(
        np.sqrt(
            np.mean(x ** 2)
        )
    )


def periodogram_features(x, fs=FS):
    """
    Hann-window FFT periodogram on a 1-s centered magnitude signal.

    Analysis range:
      0.5–15 Hz

    Bands:
      locomotor : 0.5 <= f < 3 Hz
      FoG       : 3 <= f < 8 Hz

    NOTE:
      With 60 samples at 60 Hz, FFT spacing is ~1 Hz.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    window = np.hanning(len(x))
    xw = x * window

    fft = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(
        len(x),
        d=1.0 / fs,
    )

    power = np.abs(fft) ** 2

    analysis_mask = (
        (freqs >= 0.5)
        & (freqs <= 15.0)
    )

    ap = power[analysis_mask]
    total = float(np.sum(ap))
    eps = 1e-12

    if total <= eps:
        return {
            "rel_power_3_8": 0.0,
            "freeze_index_log10": 0.0,
            "spectral_entropy": 0.0,
        }

    p = ap / total
    p_nonzero = p[p > 0]

    if len(p_nonzero) <= 1:
        entropy = 0.0
    else:
        entropy = float(
            -np.sum(
                p_nonzero
                * np.log(p_nonzero)
            )
            / np.log(
                len(p_nonzero)
            )
        )

    low_mask = (
        (freqs >= 0.5)
        & (freqs < 3.0)
    )

    freeze_mask = (
        (freqs >= 3.0)
        & (freqs < 8.0)
    )

    p_low = float(
        np.sum(
            power[low_mask]
        )
    )

    p_freeze = float(
        np.sum(
            power[freeze_mask]
        )
    )

    rel_freeze = (
        p_freeze
        / (total + eps)
    )

    freeze_index = float(
        np.log10(
            (p_freeze + eps)
            / (p_low + eps)
        )
    )

    return {
        "rel_power_3_8": rel_freeze,
        "freeze_index_log10": freeze_index,
        "spectral_entropy": entropy,
    }


def extract_selected_features(chunk):
    acc = chunk[
        ACC_COLS
    ].to_numpy(dtype=float)

    gyro = chunk[
        GYRO_COLS
    ].to_numpy(dtype=float)

    if acc.shape != (WINDOW_SAMPLES, 3):
        raise RuntimeError(
            f"Unexpected ACC shape {acc.shape}"
        )

    if gyro.shape != (WINDOW_SAMPLES, 3):
        raise RuntimeError(
            f"Unexpected GYRO shape {gyro.shape}"
        )

    acc_mag = np.linalg.norm(
        acc,
        axis=1,
    )

    gyro_mag = np.linalg.norm(
        gyro,
        axis=1,
    )

    acc_dyn = (
        acc_mag
        - np.mean(acc_mag)
    )

    gyro_dyn = (
        gyro_mag
        - np.mean(gyro_mag)
    )

    acc_diff = (
        np.diff(
            acc,
            axis=0,
        )
        * FS
    )

    gyro_diff = (
        np.diff(
            gyro,
            axis=0,
        )
        * FS
    )

    acc_jerk_mag = np.linalg.norm(
        acc_diff,
        axis=1,
    )

    gyro_angacc_mag = np.linalg.norm(
        gyro_diff,
        axis=1,
    )

    acc_spec = periodogram_features(
        acc_dyn,
        fs=FS,
    )

    gyro_spec = periodogram_features(
        gyro_dyn,
        fs=FS,
    )

    return {
        "acc_mag_std":
            float(
                np.std(
                    acc_mag,
                    ddof=0,
                )
            ),
        "acc_jerk_vector_rms":
            rms(
                acc_jerk_mag
            ),
        "gyro_mag_std":
            float(
                np.std(
                    gyro_mag,
                    ddof=0,
                )
            ),
        "gyro_angacc_vector_rms":
            rms(
                gyro_angacc_mag
            ),
        "acc_rel_power_3_8":
            float(
                acc_spec[
                    "rel_power_3_8"
                ]
            ),
        "acc_freeze_index_log10":
            float(
                acc_spec[
                    "freeze_index_log10"
                ]
            ),
        "acc_spectral_entropy":
            float(
                acc_spec[
                    "spectral_entropy"
                ]
            ),
        "gyro_rel_power_3_8":
            float(
                gyro_spec[
                    "rel_power_3_8"
                ]
            ),
        "gyro_spectral_entropy":
            float(
                gyro_spec[
                    "spectral_entropy"
                ]
            ),
    }


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def validate_reference_chunk(segmented, row):
    start = int(
        row["start_row_pos"]
    )

    end = int(
        row[
            "end_row_pos_exclusive"
        ]
    )

    if end - start != WINDOW_SAMPLES:
        raise RuntimeError(
            f"Chunk {row['chunk_id']} does not contain 60 samples."
        )

    g = segmented.iloc[
        start:end
    ]

    if len(g) != WINDOW_SAMPLES:
        raise RuntimeError(
            f"Chunk {row['chunk_id']} is outside reconstructed data."
        )

    if g["subjectID"].nunique() != 1:
        raise RuntimeError(
            f"Chunk {row['chunk_id']} crosses subjects."
        )

    if (
        int(
            g["subjectID"].iloc[0]
        )
        != int(
            row["subjectID"]
        )
    ):
        raise RuntimeError(
            f"Chunk {row['chunk_id']} subject mismatch."
        )

    if g["pool_state"].nunique() != 1:
        raise RuntimeError(
            f"Chunk {row['chunk_id']} crosses states."
        )

    if (
        g["pool_state"].iloc[0]
        != row["pool_state"]
    ):
        raise RuntimeError(
            f"Chunk {row['chunk_id']} state mismatch."
        )

    if not g[
        RIGHT_ANKLE_COLS
    ].notna().all().all():
        raise RuntimeError(
            f"Chunk {row['chunk_id']} has incomplete right-ankle data."
        )

    return g


def build_feature_table(
    segmented,
    reference,
):
    rows = []
    total = len(reference)

    for i, (_, row) in enumerate(
        reference.iterrows(),
        start=1,
    ):
        g = validate_reference_chunk(
            segmented,
            row,
        )

        feat = extract_selected_features(
            g
        )

        out = {
            "chunk_id":
                int(
                    row["chunk_id"]
                ),
            "subjectID":
                int(
                    row["subjectID"]
                ),
            "segment_id":
                int(
                    row["segment_id"]
                ),
            # Step 01 recreates reference windows at the original

            # homogeneous-segment level before the right-ankle filter.

            # Therefore block_id is optional metadata here.

            "block_id":

                (

                    int(row["block_id"])

                    if (

                        "block_id" in row.index

                        and pd.notna(row["block_id"])

                    )

                    else np.nan

                ),
            "sessionID":
                row["sessionID"],
            "taskID":
                row["taskID"],
            "activity":
                int(
                    row["activity"]
                ),
            "activity_name":
                ACTIVITY_NAMES.get(
                    int(
                        row["activity"]
                    ),
                    "unknown",
                ),
            "pool_state":
                row["pool_state"],
            "start_row_pos":
                int(
                    row["start_row_pos"]
                ),
            "end_row_pos_exclusive":
                int(
                    row[
                        "end_row_pos_exclusive"
                    ]
                ),
            "reference_phase_offset_samples":
                (
                    int(
                        row[
                            "reference_phase_offset_samples"
                        ]
                    )
                    if (
                        "reference_phase_offset_samples"
                        in row.index
                        and pd.notna(
                            row[
                                "reference_phase_offset_samples"
                            ]
                        )
                    )
                    else np.nan
                ),
        }

        out.update(feat)
        rows.append(out)

        if (
            i % 500 == 0
            or i == total
        ):
            print(
                f"  extracted {i:,}/{total:,} chunks"
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# SUBJECT-LEVEL SUMMARIES
# =============================================================================

def build_subject_state_medians(
    feature_df,
):
    rows = []

    for (
        sid,
        state,
    ), g in feature_df.groupby(
        [
            "subjectID",
            "pool_state",
        ]
    ):
        row = {
            "subjectID":
                int(sid),
            "pool_state":
                state,
            "n_chunks":
                len(g),
        }

        for feat in FEATURES:
            row[
                feat
            ] = float(
                g[
                    feat
                ].median()
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def build_state_summary(
    subject_state,
):
    rows = []

    for state in STATE_ORDER:
        g = subject_state[
            subject_state[
                "pool_state"
            ] == state
        ]

        for feat in FEATURES:
            x = (
                g[
                    feat
                ]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            q25, med, q75 = np.percentile(
                x,
                [25, 50, 75],
            )

            rows.append({
                "pool_state":
                    state,
                "feature":
                    feat,
                "n_subjects":
                    len(x),
                "median_subject_value":
                    float(med),
                "q25_subject_value":
                    float(q25),
                "q75_subject_value":
                    float(q75),
                "iqr_subject_value":
                    float(
                        q75 - q25
                    ),
            })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PAIRED STAGE CONTRASTS
# =============================================================================

def subject_stage2_medians(
    feature_df,
):
    fog = feature_df[
        feature_df[
            "pool_state"
        ].isin(
            [
                "shuffling",
                "trembling",
                "akinesia",
            ]
        )
    ].copy()

    fog["stage2_state"] = np.where(
        fog[
            "pool_state"
        ].isin(
            [
                "shuffling",
                "trembling",
            ]
        ),
        "kinetic",
        "akinetic",
    )

    rows = []

    for (
        sid,
        state,
    ), g in fog.groupby(
        [
            "subjectID",
            "stage2_state",
        ]
    ):
        row = {
            "subjectID":
                int(sid),
            "stage2_state":
                state,
            "n_chunks":
                len(g),
        }

        for feat in FEATURES:
            row[
                feat
            ] = float(
                g[
                    feat
                ].median()
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def subject_stage3_medians(
    feature_df,
):
    kin = feature_df[
        feature_df[
            "pool_state"
        ].isin(
            [
                "shuffling",
                "trembling",
            ]
        )
    ].copy()

    rows = []

    for (
        sid,
        state,
    ), g in kin.groupby(
        [
            "subjectID",
            "pool_state",
        ]
    ):
        row = {
            "subjectID":
                int(sid),
            "stage3_state":
                state,
            "n_chunks":
                len(g),
        }

        for feat in FEATURES:
            row[
                feat
            ] = float(
                g[
                    feat
                ].median()
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def paired_contrast(
    subject_df,
    state_col,
    state_a,
    state_b,
    comparison,
):
    a = (
        subject_df[
            subject_df[
                state_col
            ] == state_a
        ]
        .set_index(
            "subjectID"
        )
    )

    b = (
        subject_df[
            subject_df[
                state_col
            ] == state_b
        ]
        .set_index(
            "subjectID"
        )
    )

    common = sorted(
        set(
            a.index
        )
        & set(
            b.index
        )
    )

    detail_rows = []
    summary_rows = []

    for feat in FEATURES:
        diffs = []

        for sid in common:
            va = float(
                a.loc[
                    sid,
                    feat,
                ]
            )

            vb = float(
                b.loc[
                    sid,
                    feat,
                ]
            )

            diff = (
                va - vb
            )

            diffs.append(diff)

            detail_rows.append({
                "comparison":
                    comparison,
                "subjectID":
                    int(sid),
                "feature":
                    feat,
                "state_A":
                    state_a,
                "state_B":
                    state_b,
                "value_A":
                    va,
                "value_B":
                    vb,
                "difference_A_minus_B":
                    diff,
            })

        diffs = np.asarray(
            diffs,
            dtype=float,
        )

        q25, med, q75 = np.percentile(
            diffs,
            [25, 50, 75],
        )

        summary_rows.append({
            "comparison":
                comparison,
            "feature":
                feat,
            "n_mixed_subjects":
                len(diffs),
            "median_paired_difference_A_minus_B":
                float(med),
            "q25_difference":
                float(q25),
            "q75_difference":
                float(q75),
            "positive_subjects":
                int(
                    np.sum(
                        diffs > 0
                    )
                ),
            "negative_subjects":
                int(
                    np.sum(
                        diffs < 0
                    )
                ),
            "zero_subjects":
                int(
                    np.sum(
                        diffs == 0
                    )
                ),
        })

    return (
        pd.DataFrame(
            detail_rows
        ),
        pd.DataFrame(
            summary_rows
        ),
        common,
    )


# =============================================================================
# CONTEXT AUDIT
# =============================================================================

def build_context_audits(
    feature_df,
):
    activity = (
        feature_df.groupby(
            [
                "pool_state",
                "activity",
                "activity_name",
            ]
        )
        .agg(
            n_chunks=(
                "chunk_id",
                "size",
            ),
            n_subjects=(
                "subjectID",
                "nunique",
            ),
        )
        .reset_index()
    )

    activity[
        "state_total_chunks"
    ] = (
        activity.groupby(
            "pool_state"
        )[
            "n_chunks"
        ]
        .transform(
            "sum"
        )
    )

    activity[
        "within_state_percent"
    ] = (
        100.0
        * activity[
            "n_chunks"
        ]
        / activity[
            "state_total_chunks"
        ]
    )

    task = (
        feature_df.groupby(
            [
                "pool_state",
                "taskID",
            ]
        )
        .agg(
            n_chunks=(
                "chunk_id",
                "size",
            ),
            n_subjects=(
                "subjectID",
                "nunique",
            ),
        )
        .reset_index()
    )

    task[
        "state_total_chunks"
    ] = (
        task.groupby(
            "pool_state"
        )[
            "n_chunks"
        ]
        .transform(
            "sum"
        )
    )

    task[
        "within_state_percent"
    ] = (
        100.0
        * task[
            "n_chunks"
        ]
        / task[
            "state_total_chunks"
        ]
    )

    fog = feature_df[
        feature_df[
            "pool_state"
        ].isin(
            [
                "shuffling",
                "trembling",
                "akinesia",
            ]
        )
    ].copy()

    fog[
        "walking_turning_context"
    ] = fog[
        "activity"
    ].isin(
        LOCOMOTOR_ACTIVITY_CODES
    )

    fog_context = (
        fog.groupby(
            "pool_state"
        )
        .agg(
            total_fog_chunks=(
                "chunk_id",
                "size",
            ),
            chunks_in_walking_turning=(
                "walking_turning_context",
                "sum",
            ),
        )
        .reset_index()
    )

    fog_context[
        "percent_in_walking_turning"
    ] = (
        100.0
        * fog_context[
            "chunks_in_walking_turning"
        ]
        / fog_context[
            "total_fog_chunks"
        ]
    )

    return (
        activity,
        task,
        fog_context,
    )


# =============================================================================
# SENTINEL VALIDATION
# =============================================================================

def validate_sentinels(
    state_summary,
    stage2_summary,
    stage3_summary,
):
    rows = []

    for (
        state,
        feat,
    ), expected in EXPECTED_SENTINELS.items():

        if state == "stage2":
            g = stage2_summary[
                stage2_summary[
                    "feature"
                ] == feat
            ]

            observed = float(
                g.iloc[0][
                    "median_paired_difference_A_minus_B"
                ]
            )

        elif state == "stage3":
            g = stage3_summary[
                stage3_summary[
                    "feature"
                ] == feat
            ]

            observed = float(
                g.iloc[0][
                    "median_paired_difference_A_minus_B"
                ]
            )

        else:
            g = state_summary[
                (
                    state_summary[
                        "pool_state"
                    ] == state
                )
                & (
                    state_summary[
                        "feature"
                    ] == feat
                )
            ]

            observed = float(
                g.iloc[0][
                    "median_subject_value"
                ]
            )

        abs_diff = abs(
            observed - expected
        )

        passed = (
            abs_diff
            <= SENTINEL_ABS_TOL
        )

        rows.append({
            "sentinel_group":
                state,
            "feature":
                feat,
            "observed":
                observed,
            "expected":
                expected,
            "absolute_difference":
                abs_diff,
            "tolerance":
                SENTINEL_ABS_TOL,
            "pass":
                int(passed),
        })

        print(
            f"{state:<20s} {feat:<26s} "
            f"obs={observed:.6g} "
            f"expected={expected:.6g} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "pass"
    ].all():
        raise RuntimeError(
            "Step 02 sentinel results do not match the frozen reference values."
        )

    return audit


# =============================================================================
# MODELING TABLES FOR STEP 03
# =============================================================================

def create_stage_modeling_table(
    stage_name,
    feature_df,
):
    manifest = pd.read_csv(
        STAGE_MANIFEST_PATHS[
            stage_name
        ]
    )

    # Manifest already contains one row per chunk per outer fold.
    out = manifest.merge(
        feature_df[
            [
                "chunk_id",
                *FEATURES,
            ]
        ],
        on="chunk_id",
        how="left",
        validate="many_to_one",
    )

    if out[
        FEATURES
    ].isna().any().any():
        raise RuntimeError(
            f"{stage_name}: missing feature values after manifest merge."
        )

    return out


# =============================================================================
# MAIN
# =============================================================================

header(
    "STEP 02 — INTERPRETABLE BIOMECHANICAL FEATURES"
)

print("Project root      :", ROOT)
print("Sensor file       :", SENSOR_PATH)
print("Reference windows :", REFERENCE_PATH)
print("Processed         :", PROCESSED_DIR)
print("Reports           :", REPORT_DIR)
print()
print("Frozen selected features:", len(FEATURES))
for feat in FEATURES:
    print(" -", feat)

print()
print(
    "No classifier training occurs in Step 02."
)


# -----------------------------------------------------------------------------
# 1. Rebuild signal row order
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
    - set(
        raw.columns
    )
)

if missing:
    raise ValueError(
        f"Missing required sensor columns: {sorted(missing)}"
    )

raw["subjectID"] = (
    raw[
        "subjectID"
    ].astype(int)
)

raw = add_pool_state(
    raw
)

median_dt, gap_threshold = (
    infer_gap_threshold(
        raw
    )
)

segmented = (
    assign_homogeneous_segments(
        raw,
        gap_threshold,
    )
)

reference = pd.read_csv(
    REFERENCE_PATH
)

print(
    "Raw rows:",
    len(raw),
)

print(
    "Reconstructed rows:",
    len(segmented),
)

print(
    "Reference chunks:",
    len(reference),
)

print(
    "Subjects:",
    reference[
        "subjectID"
    ].nunique(),
)

if len(reference) != 3560:
    raise RuntimeError(
        f"Expected 3560 reference chunks, found {len(reference)}."
    )

print(
    "Input validation: PASS"
)


# -----------------------------------------------------------------------------
# 2. Extract features
# -----------------------------------------------------------------------------

header(
    "2. EXTRACTING THE 9 FROZEN FEATURES"
)

feature_df = build_feature_table(
    segmented,
    reference,
)

feature_path = (
    PROCESSED_DIR
    / "step02_reference_features_9.csv"
)

feature_df.to_csv(
    feature_path,
    index=False,
)

print()
print(
    "Feature table rows:",
    len(feature_df),
)

print(
    "Selected feature columns:",
    len(FEATURES),
)

print(
    "Feature table:",
    feature_path,
)


# -----------------------------------------------------------------------------
# 3. Context audit
# -----------------------------------------------------------------------------

header(
    "3. ACTIVITY / TASK CONTEXT AUDIT"
)

(
    activity_audit,
    task_audit,
    fog_context,
) = build_context_audits(
    feature_df
)

activity_audit.to_csv(
    REPORT_DIR
    / "02_activity_state_audit.csv",
    index=False,
)

task_audit.to_csv(
    REPORT_DIR
    / "03_task_state_audit.csv",
    index=False,
)

fog_context.to_csv(
    REPORT_DIR
    / "04_fog_activity_context_match.csv",
    index=False,
)

print(
    fog_context.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

all_fog = feature_df[
    feature_df[
        "pool_state"
    ].isin(
        [
            "shuffling",
            "trembling",
            "akinesia",
        ]
    )
]

overall_context_match = (
    100.0
    * all_fog[
        "activity"
    ]
    .isin(
        LOCOMOTOR_ACTIVITY_CODES
    )
    .mean()
)

print()
print(
    "All FoG reference chunks in walking/turning:",
    f"{overall_context_match:.2f}%",
)

if abs(
    overall_context_match
    - 98.21
) > 0.01:
    raise RuntimeError(
        "Walking/turning context result does not match the frozen reference value."
    )

print(
    "Stage-1 context audit reproduction: PASS"
)


# -----------------------------------------------------------------------------
# 3. Subject-level state phenotypes
# -----------------------------------------------------------------------------

header(
    "3. SUBJECT-LEVEL STATE PHENOTYPES"
)

subject_state = (
    build_subject_state_medians(
        feature_df
    )
)

subject_state.to_csv(
    PROCESSED_DIR
    / "step02_subject_state_feature_medians.csv",
    index=False,
)

state_summary = (
    build_state_summary(
        subject_state
    )
)

state_summary.to_csv(
    REPORT_DIR
    / "05_subject_level_state_feature_summary.csv",
    index=False,
)

print(
    state_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


# -----------------------------------------------------------------------------
# 3. Stage 2 paired mixed-subject contrast
# -----------------------------------------------------------------------------

header(
    "3. STAGE 2 — KINETIC MINUS AKINETIC PAIRED CONTRAST"
)

stage2_subject = (
    subject_stage2_medians(
        feature_df
    )
)

stage2_subject.to_csv(
    PROCESSED_DIR
    / "step02_stage2_subject_feature_medians.csv",
    index=False,
)

(
    stage2_detail,
    stage2_summary,
    stage2_ids,
) = paired_contrast(
    stage2_subject,
    state_col="stage2_state",
    state_a="kinetic",
    state_b="akinetic",
    comparison="Stage2_kinetic_minus_akinetic",
)

if stage2_ids != EXPECTED_STAGE2_MIXED:
    raise RuntimeError(
        f"Stage 2 mixed IDs mismatch: {stage2_ids}"
    )

stage2_detail.to_csv(
    REPORT_DIR
    / "06_stage2_paired_detail.csv",
    index=False,
)

stage2_summary.to_csv(
    REPORT_DIR
    / "07_stage2_paired_summary.csv",
    index=False,
)

print(
    "Mixed subjects:",
    stage2_ids,
)

print(
    stage2_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


# -----------------------------------------------------------------------------
# 3. Stage 3 paired mixed-subject contrast
# -----------------------------------------------------------------------------

header(
    "3. STAGE 3 — SHUFFLING MINUS TREMBLING PAIRED CONTRAST"
)

stage3_subject = (
    subject_stage3_medians(
        feature_df
    )
)

stage3_subject.to_csv(
    PROCESSED_DIR
    / "step02_stage3_subject_feature_medians.csv",
    index=False,
)

(
    stage3_detail,
    stage3_summary,
    stage3_ids,
) = paired_contrast(
    stage3_subject,
    state_col="stage3_state",
    state_a="shuffling",
    state_b="trembling",
    comparison="Stage3_shuffling_minus_trembling",
)

if stage3_ids != EXPECTED_STAGE3_MIXED:
    raise RuntimeError(
        f"Stage 3 mixed IDs mismatch: {stage3_ids}"
    )

stage3_detail.to_csv(
    REPORT_DIR
    / "08_stage3_paired_detail.csv",
    index=False,
)

stage3_summary.to_csv(
    REPORT_DIR
    / "09_stage3_paired_summary.csv",
    index=False,
)

print(
    "Mixed subjects:",
    stage3_ids,
)

print(
    stage3_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


# -----------------------------------------------------------------------------
# 3. Sentinel reproduction
# -----------------------------------------------------------------------------

header(
    "3. VALIDATED RESULT SENTINELS"
)

sentinel_audit = validate_sentinels(
    state_summary,
    stage2_summary,
    stage3_summary,
)

sentinel_audit.to_csv(
    REPORT_DIR
    / "10_reference_result_sentinels.csv",
    index=False,
)

print(
    "Reference result sentinels: PASS"
)


# -----------------------------------------------------------------------------
# 3. Stage-specific modeling tables for Step 03
# -----------------------------------------------------------------------------

header(
    "3. STAGE-SPECIFIC FEATURE TABLES FOR TRADITIONAL ML"
)

for stage_name, stem in [
    (
        "Stage 1",
        "stage1",
    ),
    (
        "Stage 2",
        "stage2",
    ),
    (
        "Stage 3",
        "stage3",
    ),
]:
    model_table = (
        create_stage_modeling_table(
            stage_name,
            feature_df,
        )
    )

    out_path = (
        PROCESSED_DIR
        / f"step02_{stem}_modeling_features.csv"
    )

    model_table.to_csv(
        out_path,
        index=False,
    )

    unique_chunks = (
        model_table[
            "chunk_id"
        ].nunique()
    )

    print(
        f"{stage_name}: "
        f"{unique_chunks} unique chunks, "
        f"{len(model_table)} fold-partition rows"
    )


# -----------------------------------------------------------------------------
# 3. Analysis lock
# -----------------------------------------------------------------------------

header(
    "3. STEP 02 ANALYSIS LOCK"
)

feature_dictionary = pd.DataFrame([
    {
        "feature":
            "acc_mag_std",
        "family":
            "temporal",
        "interpretation":
            "within-window acceleration-magnitude variability",
    },
    {
        "feature":
            "acc_jerk_vector_rms",
        "family":
            "temporal",
        "interpretation":
            "RMS magnitude of acceleration derivative",
    },
    {
        "feature":
            "gyro_mag_std",
        "family":
            "temporal",
        "interpretation":
            "within-window angular-velocity-magnitude variability",
    },
    {
        "feature":
            "gyro_angacc_vector_rms",
        "family":
            "temporal",
        "interpretation":
            "RMS magnitude of angular-velocity derivative",
    },
    {
        "feature":
            "acc_rel_power_3_8",
        "family":
            "spectral",
        "interpretation":
            "relative acceleration-magnitude power from 3 to <8 Hz",
    },
    {
        "feature":
            "acc_freeze_index_log10",
        "family":
            "spectral",
        "interpretation":
            "log10 ratio of acceleration 3–<8 Hz power to 0.5–<3 Hz power",
    },
    {
        "feature":
            "acc_spectral_entropy",
        "family":
            "spectral",
        "interpretation":
            "normalized spectral entropy of acceleration magnitude, 0.5–15 Hz",
    },
    {
        "feature":
            "gyro_rel_power_3_8",
        "family":
            "spectral",
        "interpretation":
            "relative angular-velocity-magnitude power from 3 to <8 Hz",
    },
    {
        "feature":
            "gyro_spectral_entropy",
        "family":
            "spectral",
        "interpretation":
            "normalized spectral entropy of angular-velocity magnitude, 0.5–15 Hz",
    },
])

feature_dictionary.to_csv(
    REPORT_DIR
    / "11_feature_dictionary.csv",
    index=False,
)

lock = {
    "features":
        FEATURES,
    "n_features":
        len(FEATURES),
    "removed_redundant_feature":
        "acc_mag_dynamic_rms",
    "reason_removed":
        "numerically redundant with acc_mag_std for centered magnitude signal",
    "window_seconds":
        WINDOW_SEC,
    "sampling_rate_hz":
        FS,
    "spectral_resolution_note":
        "1-s FFT at 60 Hz gives approximately 1-Hz bin spacing",
    "stage2_mixed_subjects":
        stage2_ids,
    "stage3_mixed_subjects":
        stage3_ids,
    "inference":
        "descriptive subject-level medians and paired contrasts; no p-values",
    "status":
        "matches the frozen feature definitions and reference sentinels",
}

with open(
    REPORT_DIR
    / "step02_analysis_lock.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        lock,
        f,
        indent=2,
    )


# -----------------------------------------------------------------------------
# FINAL
# -----------------------------------------------------------------------------

print("Step 02 reproduction checks: PASS")
