#!/usr/bin/env python3
"""
Step 01 — Window construction and subject-independent splits.

Builds deterministic 1-s reference windows, complete training blocks, and the frozen participant partitions.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# =============================================================================
# FROZEN CONSTANTS
# =============================================================================

FS = 60.0
WINDOW_SEC = 1.0
WINDOW_SAMPLES = 60

# Frozen deterministic evaluation-window phase.
REFERENCE_CHUNK_SEED = 20260907

LOCOMOTOR_ACTIVITY_CODES = {1, 6, 7}

RIGHT_ANKLE_COLS = [
    "ankleR_acc_x",
    "ankleR_acc_y",
    "ankleR_acc_z",
    "ankleR_gyro_x",
    "ankleR_gyro_y",
    "ankleR_gyro_z",
]

# Frozen outer folds.
OUTER_FOLDS = {
    1: [5, 7, 12, 15, 16],
    2: [1, 8, 13, 17, 22],
    3: [4, 6, 19, 20],
    4: [3, 9, 11, 14],
    5: [2, 10, 18, 21],
}

# Frozen inner-validation sets.
INNER_VALIDATION = {
    1: [3, 9, 10, 19],
    2: [9, 10, 11, 19],
    3: [1, 9, 10, 15],
    4: [7, 12, 13, 19],
    5: [4, 7, 13, 19],
}

ALL_SUBJECTS = list(range(1, 23))

EXPECTED_COMPLETE_BLOCKS = {
    "non_fog_locomotor": {
        "n_blocks": 506,
        "seconds": 2772.05,
        "possible_starts": 136469,
    },
    "shuffling": {
        "n_blocks": 30,
        "seconds": 96.68333333333334,
        "possible_starts": 4031,
    },
    "trembling": {
        "n_blocks": 38,
        "seconds": 224.03333333333333,
        "possible_starts": 11200,
    },
    "akinesia": {
        "n_blocks": 45,
        "seconds": 734.0333333333333,
        "possible_starts": 41387,
    },
}

EXPECTED_REFERENCE_CHUNKS_BEFORE_SENSOR_FILTER = {
    "non_fog_locomotor": 2679,
    "shuffling": 83,
    "trembling": 207,
    "akinesia": 714,
}

EXPECTED_REFERENCE_CHUNKS = {
    "non_fog_locomotor": 2556,
    "shuffling": 83,
    "trembling": 207,
    "akinesia": 714,
}

EXPECTED_DYNAMIC_POSSIBLE_STARTS_BY_FOLD = {
    1: {
        "akinesia": 27671,
        "non_fog_locomotor": 72919,
        "shuffling": 1762,
        "trembling": 5886,
    },
    2: {
        "akinesia": 22174,
        "non_fog_locomotor": 81985,
        "shuffling": 2181,
        "trembling": 6403,
    },
    3: {
        "akinesia": 23919,
        "non_fog_locomotor": 79039,
        "shuffling": 1796,
        "trembling": 6020,
    },
    4: {
        "akinesia": 26221,
        "non_fog_locomotor": 79328,
        "shuffling": 1521,
        "trembling": 6953,
    },
    5: {
        "akinesia": 18300,
        "non_fog_locomotor": 76461,
        "shuffling": 2612,
        "trembling": 6252,
    },
}

# Deterministic 1-s chunk counts in each partition/fold.
EXPECTED_SPLIT_COUNTS = {
    1: {
        "test": {
            "akinesia": 38, "non_fog_locomotor": 581,
            "shuffling": 15, "trembling": 56,
        },
        "train": {
            "akinesia": 475, "non_fog_locomotor": 1377,
            "shuffling": 39, "trembling": 107,
        },
        "validation": {
            "akinesia": 201, "non_fog_locomotor": 598,
            "shuffling": 29, "trembling": 44,
        },
    },
    2: {
        "test": {
            "akinesia": 171, "non_fog_locomotor": 400,
            "shuffling": 10, "trembling": 34,
        },
        "train": {
            "akinesia": 384, "non_fog_locomotor": 1549,
            "shuffling": 44, "trembling": 120,
        },
        "validation": {
            "akinesia": 159, "non_fog_locomotor": 607,
            "shuffling": 29, "trembling": 53,
        },
    },
    3: {
        "test": {
            "akinesia": 126, "non_fog_locomotor": 499,
            "shuffling": 15, "trembling": 41,
        },
        "train": {
            "akinesia": 415, "non_fog_locomotor": 1494,
            "shuffling": 37, "trembling": 111,
        },
        "validation": {
            "akinesia": 173, "non_fog_locomotor": 563,
            "shuffling": 31, "trembling": 55,
        },
    },
    4: {
        "test": {
            "akinesia": 106, "non_fog_locomotor": 523,
            "shuffling": 29, "trembling": 32,
        },
        "train": {
            "akinesia": 449, "non_fog_locomotor": 1502,
            "shuffling": 34, "trembling": 129,
        },
        "validation": {
            "akinesia": 159, "non_fog_locomotor": 531,
            "shuffling": 20, "trembling": 46,
        },
    },
    5: {
        "test": {
            "akinesia": 273, "non_fog_locomotor": 553,
            "shuffling": 14, "trembling": 44,
        },
        "train": {
            "akinesia": 315, "non_fog_locomotor": 1435,
            "shuffling": 54, "trembling": 118,
        },
        "validation": {
            "akinesia": 126, "non_fog_locomotor": 568,
            "shuffling": 15, "trembling": 45,
        },
    },
}


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

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pipeline"
    / "step_01_segments_windows_splits"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "pipeline"
    / "step_01_segments_windows_splits"
)

for p in [PROCESSED_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

if not SENSOR_PATH.exists():
    raise FileNotFoundError(SENSOR_PATH)


# =============================================================================
# HELPERS
# =============================================================================

def header(title):
    print("\n" + "=" * 102)
    print(title)
    print("=" * 102)


def subheader(title):
    print("\n" + "-" * 102)
    print(title)
    print("-" * 102)


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

    median_dt = float(
        np.median(all_d)
    )

    gap_threshold = (
        median_dt
        * 2.5
    )

    return median_dt, gap_threshold


def assign_homogeneous_segments(
    df,
    gap_threshold,
):
    pieces = []
    next_segment_id = 0

    for _, g in df.groupby(
        ["subjectID", "sessionID", "taskID"],
        sort=False,
    ):
        g = (
            g.sort_values("timestamp")
            .copy()
        )

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

        local = (
            new_segment.cumsum()
            - 1
        )

        unique_local = pd.unique(
            local
        )

        mapping = {
            old:
                next_segment_id + i
            for i, old
            in enumerate(unique_local)
        }

        g["segment_id"] = (
            local.map(mapping)
            .astype(int)
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


def build_segment_catalog(segmented):
    rows = []

    keep_states = {
        "non_fog_locomotor",
        "shuffling",
        "trembling",
        "akinesia",
    }

    for segment_id, g in segmented.groupby(
        "segment_id",
        sort=True,
    ):
        state = g[
            "pool_state"
        ].iloc[0]

        if state not in keep_states:
            continue

        rows.append({
            "segment_id":
                int(segment_id),
            "subjectID":
                int(
                    g["subjectID"].iloc[0]
                ),
            "sessionID":
                g["sessionID"].iloc[0],
            "taskID":
                g["taskID"].iloc[0],
            "activity":
                int(
                    g["activity"].iloc[0]
                ),
            "pool_state":
                state,
            "start_row_pos":
                int(
                    g["row_pos"].iloc[0]
                ),
            "end_row_pos_exclusive":
                int(
                    g["row_pos"].iloc[-1]
                    + 1
                ),
            "n_samples":
                int(len(g)),
            "duration_sec":
                float(
                    len(g)
                    / FS
                ),
            "start_timestamp":
                float(
                    g["timestamp"].iloc[0]
                ),
            "end_timestamp":
                float(
                    g["timestamp"].iloc[-1]
                ),
        })

    return pd.DataFrame(rows)


def build_complete_blocks(
    segmented,
    segment_catalog,
):
    """
    Split every homogeneous segment into contiguous runs for which all six
    right-ankle channels are present. Runs shorter than one full 1-s window
    are retained in the raw run audit but excluded from dynamic/window pools.
    """

    right_complete = (
        segmented[
            RIGHT_ANKLE_COLS
        ]
        .notna()
        .all(axis=1)
        .to_numpy()
    )

    run_rows = []
    block_rows = []
    next_block_id = 0
    next_run_id = 0

    for _, seg in segment_catalog.iterrows():
        start = int(
            seg[
                "start_row_pos"
            ]
        )

        end = int(
            seg[
                "end_row_pos_exclusive"
            ]
        )

        flags = right_complete[
            start:end
        ]

        if len(flags) == 0:
            continue

        # Run boundaries whenever completeness status changes.
        change = np.empty(
            len(flags),
            dtype=bool,
        )

        change[0] = True

        if len(flags) > 1:
            change[1:] = (
                flags[1:]
                != flags[:-1]
            )

        local_run_ids = (
            np.cumsum(change)
            - 1
        )

        for local_run in np.unique(
            local_run_ids
        ):
            idx = np.flatnonzero(
                local_run_ids
                == local_run
            )

            run_start = (
                start
                + int(idx[0])
            )

            run_end = (
                start
                + int(idx[-1])
                + 1
            )

            complete = bool(
                flags[
                    idx[0]
                ]
            )

            n = int(
                run_end
                - run_start
            )

            run_rows.append({
                "run_id":
                    next_run_id,
                "segment_id":
                    int(
                        seg[
                            "segment_id"
                        ]
                    ),
                "subjectID":
                    int(
                        seg[
                            "subjectID"
                        ]
                    ),
                "sessionID":
                    seg[
                        "sessionID"
                    ],
                "taskID":
                    seg[
                        "taskID"
                    ],
                "activity":
                    int(
                        seg[
                            "activity"
                        ]
                    ),
                "pool_state":
                    seg[
                        "pool_state"
                    ],
                "is_right_ankle_complete":
                    int(complete),
                "start_row_pos":
                    run_start,
                "end_row_pos_exclusive":
                    run_end,
                "n_samples":
                    n,
                "duration_sec":
                    n / FS,
            })

            next_run_id += 1

            if (
                complete
                and n >= WINDOW_SAMPLES
            ):
                n_possible = (
                    n
                    - WINDOW_SAMPLES
                    + 1
                )

                n_reference = (
                    n
                    // WINDOW_SAMPLES
                )

                block_rows.append({
                    "block_id":
                        next_block_id,
                    "segment_id":
                        int(
                            seg[
                                "segment_id"
                            ]
                        ),
                    "subjectID":
                        int(
                            seg[
                                "subjectID"
                            ]
                        ),
                    "sessionID":
                        seg[
                            "sessionID"
                        ],
                    "taskID":
                        seg[
                            "taskID"
                        ],
                    "activity":
                        int(
                            seg[
                                "activity"
                            ]
                        ),
                    "pool_state":
                        seg[
                            "pool_state"
                        ],
                    "start_row_pos":
                        run_start,
                    "end_row_pos_exclusive":
                        run_end,
                    "n_samples":
                        n,
                    "duration_sec":
                        n / FS,
                    "n_possible_start_positions":
                        int(
                            n_possible
                        ),
                    "n_deterministic_reference_chunks":
                        int(
                            n_reference
                        ),
                })

                next_block_id += 1

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(block_rows),
    )


def stable_rng(seed, segment_id, n_samples):
    """
    Deterministic per-segment RNG used for reference-window placement.
    """
    mixed = (
        int(seed)
        + int(segment_id) * 1_000_003
        + int(n_samples) * 9_973
    ) % (2**32 - 1)

    return np.random.default_rng(
        mixed
    )


def build_reference_chunks_from_segments(
    segmented,
    segment_catalog,
    length_sec=WINDOW_SEC,
    seed=REFERENCE_CHUNK_SEED,
):
    """
    Reproduce the frozen deterministic reference-window design.

    IMPORTANT:
    Reference windows are generated at the ORIGINAL HOMOGENEOUS SEGMENT level,
    not at the right-ankle-complete-block level.

    For each eligible homogeneous segment:
        n_chunks = floor(segment_samples / chunk_samples)

    A deterministic pseudo-random phase is selected uniformly from the
    residual samples:
        residual = segment_samples - n_chunks * chunk_samples

    The maximum number of non-overlapping chunks is then placed starting from
    that offset.

    Only after all reference chunks are generated do we evaluate
    right-ankle completeness and filter incomplete chunks.

    This detail matters: anchoring windows at the starts of complete sensor
    blocks preserves the same counts but changes the actual 1-s signals and
    therefore the biomechanical features.
    """
    chunk_samples = int(
        round(
            length_sec
            * FS
        )
    )

    if not np.isclose(
        chunk_samples / FS,
        length_sec,
    ):
        raise ValueError(
            f"Length {length_sec} s cannot be represented exactly at {FS} Hz."
        )

    rows = []
    chunk_counter = 0

    right_complete = (
        segmented[
            RIGHT_ANKLE_COLS
        ]
        .notna()
        .all(axis=1)
        .to_numpy()
    )

    for _, seg in segment_catalog.iterrows():
        n_samples = int(
            seg[
                "n_samples"
            ]
        )

        n_chunks = (
            n_samples
            // chunk_samples
        )

        if n_chunks <= 0:
            continue

        residual = (
            n_samples
            - n_chunks
            * chunk_samples
        )

        rng = stable_rng(
            seed=(
                seed
                + int(
                    round(
                        length_sec
                        * 1000
                    )
                )
            ),
            segment_id=int(
                seg[
                    "segment_id"
                ]
            ),
            n_samples=n_samples,
        )

        offset = (
            int(
                rng.integers(
                    0,
                    residual + 1,
                )
            )
            if residual > 0
            else 0
        )

        base = int(
            seg[
                "start_row_pos"
            ]
        )

        for k in range(
            n_chunks
        ):
            start = (
                base
                + offset
                + k
                * chunk_samples
            )

            end = (
                start
                + chunk_samples
            )

            complete = bool(
                np.all(
                    right_complete[
                        start:end
                    ]
                )
            )

            rows.append({
                "chunk_id":
                    chunk_counter,
                "length_sec":
                    float(
                        length_sec
                    ),
                "n_samples":
                    chunk_samples,
                "segment_id":
                    int(
                        seg[
                            "segment_id"
                        ]
                    ),
                "subjectID":
                    int(
                        seg[
                            "subjectID"
                        ]
                    ),
                "sessionID":
                    seg[
                        "sessionID"
                    ],
                "taskID":
                    seg[
                        "taskID"
                    ],
                "activity":
                    int(
                        seg[
                            "activity"
                        ]
                    ),
                "pool_state":
                    seg[
                        "pool_state"
                    ],
                "start_row_pos":
                    start,
                "end_row_pos_exclusive":
                    end,
                "reference_phase_offset_samples":
                    offset,
                "ankleR_complete":
                    complete,
                "duration_sec":
                    float(
                        length_sec
                    ),
            })

            chunk_counter += 1

    return pd.DataFrame(
        rows
    )


def validate_reference_chunks_before_filter(
    reference_all,
):
    counts = (
        reference_all[
            "pool_state"
        ]
        .value_counts()
        .to_dict()
    )

    rows = []

    for state, expected in (
        EXPECTED_REFERENCE_CHUNKS_BEFORE_SENSOR_FILTER.items()
    ):
        observed = int(
            counts.get(
                state,
                0,
            )
        )

        passed = (
            observed
            == expected
        )

        print(
            f"{state:<22s}: "
            f"{observed:>5} / {expected:<5} "
            f"{'PASS' if passed else 'FAIL'}"
        )

        rows.append({
            "pool_state":
                state,
            "observed_chunks":
                observed,
            "expected_chunks":
                expected,
            "pass":
                int(
                    passed
                ),
        })

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "pass"
    ].all():
        raise RuntimeError(
            "Pre-sensor-filter reference chunks do not match the frozen design."
        )

    return audit


def build_subject_roles():
    rows = []

    for fold in sorted(
        OUTER_FOLDS
    ):
        test = set(
            OUTER_FOLDS[fold]
        )

        validation = set(
            INNER_VALIDATION[fold]
        )

        overlap = (
            test
            & validation
        )

        if overlap:
            raise RuntimeError(
                f"Fold {fold}: test/validation overlap {sorted(overlap)}"
            )

        train = (
            set(ALL_SUBJECTS)
            - test
            - validation
        )

        for sid in ALL_SUBJECTS:
            if sid in test:
                role = "test"
            elif sid in validation:
                role = "validation"
            elif sid in train:
                role = "train"
            else:
                raise RuntimeError(
                    "Unassigned subject."
                )

            rows.append({
                "outer_fold":
                    fold,
                "subjectID":
                    sid,
                "partition":
                    role,
            })

    return pd.DataFrame(
        rows
    )


def build_split_manifest(
    reference_chunks,
    subject_roles,
):
    pieces = []

    for fold in sorted(
        OUTER_FOLDS
    ):
        roles = subject_roles[
            subject_roles[
                "outer_fold"
            ] == fold
        ][
            [
                "subjectID",
                "partition",
            ]
        ]

        x = reference_chunks.merge(
            roles,
            on="subjectID",
            how="left",
            validate="many_to_one",
        )

        if x[
            "partition"
        ].isna().any():
            raise RuntimeError(
                f"Fold {fold}: missing subject role."
            )

        x.insert(
            0,
            "outer_fold",
            fold,
        )

        pieces.append(x)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def build_dynamic_training_blocks(
    complete_blocks,
    subject_roles,
):
    pieces = []

    for fold in sorted(
        OUTER_FOLDS
    ):
        train_ids = (
            subject_roles.loc[
                (
                    subject_roles[
                        "outer_fold"
                    ] == fold
                )
                & (
                    subject_roles[
                        "partition"
                    ] == "train"
                ),
                "subjectID",
            ]
            .astype(int)
            .tolist()
        )

        x = complete_blocks[
            complete_blocks[
                "subjectID"
            ].isin(
                train_ids
            )
        ].copy()

        x.insert(
            0,
            "outer_fold",
            fold,
        )

        x.insert(
            1,
            "partition",
            "train",
        )

        pieces.append(x)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def validate_complete_blocks(
    complete_blocks,
):
    rows = []

    for state, expected in (
        EXPECTED_COMPLETE_BLOCKS.items()
    ):
        g = complete_blocks[
            complete_blocks[
                "pool_state"
            ] == state
        ]

        observed_blocks = len(g)

        observed_sec = float(
            g[
                "n_samples"
            ].sum()
            / FS
        )

        observed_starts = int(
            g[
                "n_possible_start_positions"
            ].sum()
        )

        pass_blocks = (
            observed_blocks
            == expected[
                "n_blocks"
            ]
        )

        pass_seconds = (
            abs(
                observed_sec
                - expected[
                    "seconds"
                ]
            )
            < 1e-9
        )

        pass_starts = (
            observed_starts
            == expected[
                "possible_starts"
            ]
        )

        print(
            f"{state:<22s} "
            f"blocks {observed_blocks:>4}/"
            f"{expected['n_blocks']:<4} "
            f"seconds {observed_sec:>9.3f}/"
            f"{expected['seconds']:<9.3f} "
            f"starts {observed_starts:>7}/"
            f"{expected['possible_starts']:<7} "
            f"{'PASS' if (pass_blocks and pass_seconds and pass_starts) else 'FAIL'}"
        )

        rows.append({
            "pool_state":
                state,
            "observed_blocks":
                observed_blocks,
            "expected_blocks":
                expected[
                    "n_blocks"
                ],
            "observed_seconds":
                observed_sec,
            "expected_seconds":
                expected[
                    "seconds"
                ],
            "observed_possible_starts":
                observed_starts,
            "expected_possible_starts":
                expected[
                    "possible_starts"
                ],
            "pass":
                int(
                    pass_blocks
                    and pass_seconds
                    and pass_starts
                ),
        })

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "pass"
    ].all():
        raise RuntimeError(
            "Complete-block audit did not match the frozen design."
        )

    return audit


def validate_reference_chunks(
    reference_chunks,
):
    counts = (
        reference_chunks[
            "pool_state"
        ]
        .value_counts()
        .to_dict()
    )

    rows = []

    for state, expected in (
        EXPECTED_REFERENCE_CHUNKS.items()
    ):
        observed = int(
            counts.get(
                state,
                0,
            )
        )

        passed = (
            observed
            == expected
        )

        print(
            f"{state:<22s}: "
            f"{observed:>5} / {expected:<5} "
            f"{'PASS' if passed else 'FAIL'}"
        )

        rows.append({
            "pool_state":
                state,
            "observed_chunks":
                observed,
            "expected_chunks":
                expected,
            "pass":
                int(passed),
        })

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "pass"
    ].all():
        raise RuntimeError(
            "Reference-chunk audit did not match the frozen design."
        )

    return audit


def validate_split_counts(
    split_manifest,
):
    rows = []

    for fold, partitions in (
        EXPECTED_SPLIT_COUNTS.items()
    ):
        for partition, state_counts in (
            partitions.items()
        ):
            g = split_manifest[
                (
                    split_manifest[
                        "outer_fold"
                    ] == fold
                )
                & (
                    split_manifest[
                        "partition"
                    ] == partition
                )
            ]

            observed = (
                g[
                    "pool_state"
                ]
                .value_counts()
                .to_dict()
            )

            for state, expected in (
                state_counts.items()
            ):
                obs = int(
                    observed.get(
                        state,
                        0,
                    )
                )

                passed = (
                    obs
                    == expected
                )

                rows.append({
                    "outer_fold":
                        fold,
                    "partition":
                        partition,
                    "pool_state":
                        state,
                    "observed":
                        obs,
                    "expected":
                        expected,
                    "pass":
                        int(passed),
                })

                if not passed:
                    print(
                        f"FAIL fold={fold} "
                        f"partition={partition} "
                        f"state={state}: "
                        f"{obs} vs {expected}"
                    )

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "pass"
    ].all():
        raise RuntimeError(
            "Split chunk counts do not match the frozen design."
        )

    return audit


def validate_dynamic_starts(
    dynamic_blocks,
):
    rows = []

    for fold, expected_states in (
        EXPECTED_DYNAMIC_POSSIBLE_STARTS_BY_FOLD.items()
    ):
        g = dynamic_blocks[
            dynamic_blocks[
                "outer_fold"
            ] == fold
        ]

        observed = (
            g.groupby(
                "pool_state"
            )[
                "n_possible_start_positions"
            ]
            .sum()
            .to_dict()
        )

        for state, expected in (
            expected_states.items()
        ):
            obs = int(
                observed.get(
                    state,
                    0,
                )
            )

            passed = (
                obs
                == expected
            )

            rows.append({
                "outer_fold":
                    fold,
                "pool_state":
                    state,
                "observed_possible_starts":
                    obs,
                "expected_possible_starts":
                    expected,
                "pass":
                    int(passed),
            })

            if not passed:
                print(
                    f"FAIL fold={fold} state={state}: "
                    f"{obs} vs {expected}"
                )

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "pass"
    ].all():
        raise RuntimeError(
            "Dynamic-training start-position counts do not match the frozen design."
        )

    return audit


# =============================================================================
# MAIN
# =============================================================================

header(
    "STEP 01 — HOMOGENEOUS SEGMENTS, 1-s WINDOWS, AND FROZEN SUBJECT SPLITS"
)

print("Project root :", ROOT)
print("Sensor file  :", SENSOR_PATH)
print("Processed    :", PROCESSED_DIR)
print("Reports      :", REPORT_DIR)
print()
print("No model training occurs in Step 01.")


# -----------------------------------------------------------------------------
# 1. Load / rebuild row order
# -----------------------------------------------------------------------------

header(
    "1. RAW DATA + FROZEN ROW ORDER"
)

sensor = pd.read_csv(
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
        sensor.columns
    )
)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

sensor["subjectID"] = (
    sensor[
        "subjectID"
    ].astype(int)
)

sensor = add_pool_state(
    sensor
)

median_dt, gap_threshold = (
    infer_gap_threshold(
        sensor
    )
)

segmented = (
    assign_homogeneous_segments(
        sensor,
        gap_threshold,
    )
)

print(
    "Rows:",
    len(segmented),
)

print(
    "Median positive timestamp increment:",
    f"{median_dt:.9f}",
)

print(
    "Temporal gap threshold:",
    f"{gap_threshold:.9f}",
)

print(
    "Homogeneous segments (all states):",
    segmented[
        "segment_id"
    ].nunique(),
)


# -----------------------------------------------------------------------------
# 2. Homogeneous segment catalog
# -----------------------------------------------------------------------------

header(
    "2. ELIGIBLE HOMOGENEOUS SEGMENT CATALOG"
)

segment_catalog = (
    build_segment_catalog(
        segmented
    )
)

segment_catalog.to_csv(
    PROCESSED_DIR
    / "step01_homogeneous_segment_catalog.csv",
    index=False,
)

seg_summary = (
    segment_catalog.groupby(
        "pool_state"
    )
    .agg(
        n_segments=(
            "segment_id",
            "count",
        ),
        n_subjects=(
            "subjectID",
            "nunique",
        ),
        seconds=(
            "duration_sec",
            "sum",
        ),
    )
    .reset_index()
)

print(
    seg_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 3. Complete right-ankle blocks
# -----------------------------------------------------------------------------

header(
    "3. COMPLETE RIGHT-ANKLE BLOCKS"
)

(
    completeness_runs,
    complete_blocks,
) = build_complete_blocks(
    segmented,
    segment_catalog,
)

completeness_runs.to_csv(
    PROCESSED_DIR
    / "step01_right_ankle_completeness_runs.csv",
    index=False,
)

complete_blocks.to_csv(
    PROCESSED_DIR
    / "step01_complete_sampling_blocks_global.csv",
    index=False,
)

block_audit = (
    validate_complete_blocks(
        complete_blocks
    )
)

block_audit.to_csv(
    REPORT_DIR
    / "01_complete_block_reproduction_audit.csv",
    index=False,
)

print(
    "Complete-block reproduction: PASS"
)


# -----------------------------------------------------------------------------
# 4. Deterministic reference windows
# -----------------------------------------------------------------------------

header(
    "4. DETERMINISTIC RANDOM-PHASE 1-s REFERENCE WINDOWS + SENSOR FILTER"
)

reference_all = (
    build_reference_chunks_from_segments(
        segmented,
        segment_catalog,
        length_sec=WINDOW_SEC,
        seed=REFERENCE_CHUNK_SEED,
    )
)

reference_all.to_csv(
    PROCESSED_DIR
    / "step01_reference_chunks_1s_before_sensor_filter.csv",
    index=False,
)

print(
    "Before right-ankle completeness filter:"
)

reference_before_audit = (
    validate_reference_chunks_before_filter(
        reference_all
    )
)

reference_before_audit.to_csv(
    REPORT_DIR
    / "02a_reference_chunk_before_sensor_filter_audit.csv",
    index=False,
)

reference_chunks = (
    reference_all[
        reference_all[
            "ankleR_complete"
        ]
    ]
    .copy()
)

# Preserve reference chunk_id values, including gaps created by sensor filtering.
reference_chunks.to_csv(
    PROCESSED_DIR
    / "step01_reference_chunks_1s.csv",
    index=False,
)

print()
print(
    "After right-ankle completeness filter:"
)

reference_audit = (
    validate_reference_chunks(
        reference_chunks
    )
)

reference_audit.to_csv(
    REPORT_DIR
    / "02_reference_chunk_reproduction_audit.csv",
    index=False,
)

print()
print(
    "Total reference chunks before sensor filter:",
    len(
        reference_all
    ),
)

print(
    "Expected before filter:",
    sum(
        EXPECTED_REFERENCE_CHUNKS_BEFORE_SENSOR_FILTER.values()
    ),
)

print(
    "Total right-ankle-complete reference chunks:",
    len(
        reference_chunks
    ),
)

print(
    "Expected after filter:",
    sum(
        EXPECTED_REFERENCE_CHUNKS.values()
    ),
)

if len(
    reference_all
) != 3683:
    raise RuntimeError(
        f"Expected 3683 pre-filter reference chunks, found {len(reference_all)}."
    )

if len(
    reference_chunks
) != 3560:
    raise RuntimeError(
        f"Expected 3560 right-ankle-complete reference chunks, found {len(reference_chunks)}."
    )

print(
    "Reference-window reproduction: PASS"
)


# -----------------------------------------------------------------------------
# 5. Frozen subject roles
# -----------------------------------------------------------------------------

header(
    "5. EXACT FROZEN OUTER / INNER SUBJECT SPLITS"
)

subject_roles = (
    build_subject_roles()
)

subject_roles.to_csv(
    PROCESSED_DIR
    / "step01_subject_split_roles.csv",
    index=False,
)

for fold in sorted(
    OUTER_FOLDS
):
    g = subject_roles[
        subject_roles[
            "outer_fold"
        ] == fold
    ]

    train = g.loc[
        g[
            "partition"
        ] == "train",
        "subjectID",
    ].tolist()

    val = g.loc[
        g[
            "partition"
        ] == "validation",
        "subjectID",
    ].tolist()

    test = g.loc[
        g[
            "partition"
        ] == "test",
        "subjectID",
    ].tolist()

    print(
        f"Fold {fold}:"
    )

    print(
        "  train:",
        train,
    )

    print(
        "  val  :",
        val,
    )

    print(
        "  test :",
        test,
    )


# -----------------------------------------------------------------------------
# 6. Split reference manifest
# -----------------------------------------------------------------------------

header(
    "6. DETERMINISTIC REFERENCE WINDOWS BY FOLD / PARTITION"
)

split_manifest = (
    build_split_manifest(
        reference_chunks,
        subject_roles,
    )
)

split_manifest.to_csv(
    PROCESSED_DIR
    / "step01_split_reference_manifest_1s.csv",
    index=False,
)

split_audit = (
    validate_split_counts(
        split_manifest
    )
)

split_audit.to_csv(
    REPORT_DIR
    / "03_split_chunk_reproduction_audit.csv",
    index=False,
)

split_counts = (
    split_manifest.groupby(
        [
            "outer_fold",
            "partition",
            "pool_state",
        ]
    )
    .size()
    .rename(
        "n_chunks"
    )
    .reset_index()
)

split_counts.to_csv(
    REPORT_DIR
    / "04_split_chunk_counts.csv",
    index=False,
)

print(
    split_counts.to_string(
        index=False
    )
)

print()
print(
    "Split reference reproduction: PASS"
)


# -----------------------------------------------------------------------------
# 7. Dynamic training blocks
# -----------------------------------------------------------------------------

header(
    "7. DYNAMIC TRAINING BLOCKS — INNER-TRAIN SUBJECTS ONLY"
)

dynamic_blocks = (
    build_dynamic_training_blocks(
        complete_blocks,
        subject_roles,
    )
)

dynamic_blocks.to_csv(
    PROCESSED_DIR
    / "step01_dynamic_training_blocks.csv",
    index=False,
)

dynamic_audit = (
    validate_dynamic_starts(
        dynamic_blocks
    )
)

dynamic_audit.to_csv(
    REPORT_DIR
    / "05_dynamic_training_reproduction_audit.csv",
    index=False,
)

dynamic_summary = (
    dynamic_blocks.groupby(
        [
            "outer_fold",
            "pool_state",
        ]
    )
    .agg(
        n_blocks=(
            "block_id",
            "count",
        ),
        n_subjects=(
            "subjectID",
            "nunique",
        ),
        possible_starts=(
            "n_possible_start_positions",
            "sum",
        ),
    )
    .reset_index()
)

dynamic_summary.to_csv(
    REPORT_DIR
    / "06_dynamic_training_summary.csv",
    index=False,
)

print(
    dynamic_summary.to_string(
        index=False
    )
)

print()
print(
    "Dynamic-training reproduction: PASS"
)


# -----------------------------------------------------------------------------
# 8. Three-stage reference eligibility
# -----------------------------------------------------------------------------

header(
    "8. THREE-STAGE REFERENCE ELIGIBILITY"
)

# Stage 1 strict context match: all classes must be walking/turning.
stage1 = reference_chunks[
    reference_chunks[
        "activity"
    ].isin(
        LOCOMOTOR_ACTIVITY_CODES
    )
].copy()

stage1["y"] = (
    stage1[
        "pool_state"
    ]
    != "non_fog_locomotor"
).astype(int)

stage2 = reference_chunks[
    reference_chunks[
        "pool_state"
    ].isin(
        [
            "shuffling",
            "trembling",
            "akinesia",
        ]
    )
].copy()

stage2["y"] = (
    stage2[
        "pool_state"
    ]
    == "akinesia"
).astype(int)

stage3 = reference_chunks[
    reference_chunks[
        "pool_state"
    ].isin(
        [
            "shuffling",
            "trembling",
        ]
    )
].copy()

stage3["y"] = (
    stage3[
        "pool_state"
    ]
    == "trembling"
).astype(int)

stage_rows = []

for stage_name, x in [
    ("Stage 1", stage1),
    ("Stage 2", stage2),
    ("Stage 3", stage3),
]:
    counts = (
        x[
            "y"
        ]
        .value_counts()
        .to_dict()
    )

    stage_rows.append({
        "stage":
            stage_name,
        "n_chunks":
            len(x),
        "n_subjects":
            x[
                "subjectID"
            ].nunique(),
        "class0_chunks":
            int(
                counts.get(
                    0,
                    0,
                )
            ),
        "class1_chunks":
            int(
                counts.get(
                    1,
                    0,
                )
            ),
    })

stage_eligibility = pd.DataFrame(
    stage_rows
)

stage_eligibility.to_csv(
    REPORT_DIR
    / "07_stage_reference_eligibility.csv",
    index=False,
)

print(
    stage_eligibility.to_string(
        index=False
    )
)

# Frozen stage totals.
expected_stage = {
    "Stage 1": (3542, 22, 2556, 986),
    "Stage 2": (1004, 16, 290, 714),
    "Stage 3": (290, 15, 83, 207),
}

for _, r in stage_eligibility.iterrows():
    expected = expected_stage[
        r["stage"]
    ]

    observed = (
        int(
            r[
                "n_chunks"
            ]
        ),
        int(
            r[
                "n_subjects"
            ]
        ),
        int(
            r[
                "class0_chunks"
            ]
        ),
        int(
            r[
                "class1_chunks"
            ]
        ),
    )

    if observed != expected:
        raise RuntimeError(
            f"{r['stage']} eligibility mismatch: "
            f"{observed} vs {expected}"
        )

print()
print(
    "Three-stage reference eligibility: PASS"
)

standing_akinesia_chunks = (
    reference_chunks[
        (
            reference_chunks[
                "pool_state"
            ] == "akinesia"
        )
        & ~reference_chunks[
            "activity"
        ].isin(
            LOCOMOTOR_ACTIVITY_CODES
        )
    ]
)

print(
    "Akinesia chunks excluded from Stage 1 only due non-locomotor context:",
    len(
        standing_akinesia_chunks
    ),
)

if len(
    standing_akinesia_chunks
) != 18:
    raise RuntimeError(
        "Expected exactly 18 non-locomotor akinesia reference chunks."
    )


# -----------------------------------------------------------------------------
# 9. Save stage-specific split manifests
# -----------------------------------------------------------------------------

header(
    "9. STAGE-SPECIFIC SPLIT MANIFESTS"
)

for stage_name, stage_df in [
    ("stage1", stage1),
    ("stage2", stage2),
    ("stage3", stage3),
]:
    ids = stage_df[
        [
            "chunk_id",
            "y",
        ]
    ]

    out = split_manifest.merge(
        ids,
        on="chunk_id",
        how="inner",
        validate="many_to_one",
    )

    out.to_csv(
        PROCESSED_DIR
        / f"step01_{stage_name}_split_manifest.csv",
        index=False,
    )

    print(
        f"{stage_name}: "
        f"{len(stage_df)} unique chunks, "
        f"{len(out)} fold-partition rows"
    )


# -----------------------------------------------------------------------------
# 10. Analysis lock
# -----------------------------------------------------------------------------

header(
    "10. STEP 01 ANALYSIS LOCK"
)

lock = {
    "sampling_rate_hz":
        FS,
    "window_seconds":
        WINDOW_SEC,
    "window_samples":
        WINDOW_SAMPLES,
    "right_ankle_channels":
        RIGHT_ANKLE_COLS,
    "gap_threshold_rule":
        "2.5 * median positive timestamp increment",
    "homogeneous_boundaries": [
        "subjectID",
        "sessionID",
        "taskID",
        "activity",
        "pool_state",
        "temporal continuity",
    ],
    "missing_data_policy":
        "split at incomplete right-ankle rows; no imputation",
    "training":
        "dynamic random 1-s starts from complete inner-training blocks",
    "reference_chunk_seed":
        REFERENCE_CHUNK_SEED,
    "reference_chunk_phase":
        "deterministic random phase within each original homogeneous segment before sensor filtering",
    "validation_test":
        "deterministic non-overlapping random-phase 1-s chunks, followed by right-ankle completeness filtering",
    "outer_folds":
        OUTER_FOLDS,
    "inner_validation":
        INNER_VALIDATION,
    "stage_1":
        "walking/turning non-FoG vs walking/turning FoG; exclude 18 non-locomotor akinesia chunks",
    "stage_2":
        "kinetic vs akinetic among all FoG contexts",
    "stage_3":
        "shuffling vs trembling among kinetic FoG",
    "status":
        "matches the frozen window and split design",
}

with open(
    REPORT_DIR
    / "step01_analysis_lock.json",
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

print("Step 01 reproduction checks: PASS")
