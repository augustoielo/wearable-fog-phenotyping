#!/usr/bin/env python3
"""
Step 06 — Clinical digital phenotype analysis.

Computes protocol-observed digital FoG measures and participant-level clinical associations.
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

FS = 60.0

LOCOMOTOR_ACTIVITY_CODES = {
    1, 6, 7,
}

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260909

PRIMARY_ASSOCIATIONS = [
    {
        "biomarker":
            "locomotor_fog_burden_pct",
        "clinical":
            "fog_q",
        "population":
            "all",
        "label":
            "Locomotor FoG burden vs FoG-Q",
    },
    {
        "biomarker":
            "locomotor_fog_burden_pct",
        "clinical":
            "updrs_iii",
        "population":
            "all",
        "label":
            "Locomotor FoG burden vs MDS-UPDRS III",
    },
    {
        "biomarker":
            "akinetic_fraction_of_fog_pct",
        "clinical":
            "fog_q",
        "population":
            "observed_fog",
        "label":
            "Akinetic fraction vs FoG-Q",
    },
]

EXPLORATORY_ASSOCIATIONS = []

for clinical in [
    "h_y",
    "fes_i",
    "pdq_8",
    "disease_duration",
    "moca",
]:
    EXPLORATORY_ASSOCIATIONS.append({
        "biomarker":
            "locomotor_fog_burden_pct",
        "clinical":
            clinical,
        "population":
            "all",
        "label":
            f"Locomotor FoG burden vs {clinical}",
    })

for clinical in [
    "updrs_iii",
    "h_y",
    "fes_i",
    "pdq_8",
    "disease_duration",
    "moca",
]:
    EXPLORATORY_ASSOCIATIONS.append({
        "biomarker":
            "akinetic_fraction_of_fog_pct",
        "clinical":
            clinical,
        "population":
            "observed_fog",
        "label":
            f"Akinetic fraction vs {clinical}",
    })

for clinical in [
    "fog_q",
    "updrs_iii",
    "h_y",
    "fes_i",
    "pdq_8",
    "disease_duration",
    "moca",
]:
    EXPLORATORY_ASSOCIATIONS.append({
        "biomarker":
            "trembling_fraction_of_kinetic_pct",
        "clinical":
            clinical,
        "population":
            "observed_kinetic",
        "label":
            f"Trembling fraction vs {clinical}",
    })

for clinical in [
    "fog_q",
    "updrs_iii",
    "h_y",
    "fes_i",
    "pdq_8",
    "disease_duration",
    "moca",
]:
    EXPLORATORY_ASSOCIATIONS.append({
        "biomarker":
            "manifestation_entropy_normalized",
        "clinical":
            clinical,
        "population":
            "observed_fog",
        "label":
            f"Manifestation entropy vs {clinical}",
    })


# Frozen primary Spearman coefficients used for reproduction checks.
EXPECTED_PRIMARY_RHO = {
    (
        "locomotor_fog_burden_pct",
        "fog_q",
    ): 0.536,
    (
        "locomotor_fog_burden_pct",
        "updrs_iii",
    ): 0.785,
    (
        "akinetic_fraction_of_fog_pct",
        "fog_q",
    ): -0.206,
}

RHO_REPRODUCTION_TOL = 0.0025

EXPECTED_COVERAGE = {
    "n_subjects": 22,
    "n_observed_fog": 16,
    "n_no_observed_fog": 6,
    "n_observed_kinetic": 15,
}

EXPECTED_LOCOMOTOR_EXPOSURE_MINUTES = {
    "min": 1.19,
    "median": 3.09,
    "q25": 1.84,
    "q75": 3.66,
    "max": 7.02,
}

EXPOSURE_TOL_MINUTES = 0.03

# Reference confidence intervals are recorded for inspection but are not used as exact numerical gates.
PRIMARY_CI_REFERENCE = {
    (
        "locomotor_fog_burden_pct",
        "fog_q",
    ): [
        0.131,
        0.794,
    ],
    (
        "locomotor_fog_burden_pct",
        "updrs_iii",
    ): [
        0.556,
        0.899,
    ],
    (
        "akinetic_fraction_of_fog_pct",
        "fog_q",
    ): [
        -0.745,
        0.375,
    ],
}


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

SENSOR_PATH = (
    ROOT
    / "data"
    / "raw"
    / "sensor_data.csv"
)

CLINICAL_PATH = (
    ROOT
    / "data"
    / "raw"
    / "clinical_data.csv"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pipeline"
    / "step_06_clinical_digital_phenotype"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "pipeline"
    / "step_06_clinical_digital_phenotype"
)


for p in [
    PROCESSED_DIR,
    REPORT_DIR,
]:
    p.mkdir(
        parents=True,
        exist_ok=True,
    )

for p in [
    SENSOR_PATH,
    CLINICAL_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(
            p
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
# CLINICAL COLUMN NORMALIZATION
# =============================================================================

def normalize_col_name(
    name,
):
    x = str(
        name
    ).strip()

    x = x.lower()

    x = re.sub(
        r"[^a-z0-9]+",
        "_",
        x,
    )

    x = x.strip(
        "_"
    )

    aliases = {
        "subjectid":
            "subjectID",
        "subject_id":
            "subjectID",
        "fes_i":
            "fes_i",
        "fesi":
            "fes_i",
        "pdq8":
            "pdq_8",
        "pdq_8":
            "pdq_8",
        "updrsiii":
            "updrs_iii",
        "updrs_iii":
            "updrs_iii",
        "fogq":
            "fog_q",
        "fog_q":
            "fog_q",
        "hy":
            "h_y",
        "h_y":
            "h_y",
    }

    return aliases.get(
        x,
        x,
    )


def canonicalize_clinical(
    clinical,
):
    out = clinical.copy()

    out.columns = [
        normalize_col_name(
            c
        )
        for c in out.columns
    ]

    if "subjectID" not in out.columns:
        raise RuntimeError(
            "Could not identify clinical subjectID."
        )

    out[
        "subjectID"
    ] = pd.to_numeric(
        out[
            "subjectID"
        ],
        errors="raise",
    ).astype(int)

    for col in [
        "age",
        "disease_duration",
        "h_y",
        "updrs_iii",
        "fog_q",
        "moca",
        "fes_i",
        "pdq_8",
    ]:
        if col in out.columns:
            out[
                col
            ] = pd.to_numeric(
                out[
                    col
                ],
                errors="coerce",
            )

    return out


# =============================================================================
# DIGITAL BIOMARKERS
# =============================================================================

def normalized_entropy(
    counts,
):
    x = np.asarray(
        counts,
        dtype=float,
    )

    total = float(
        np.sum(x)
    )

    if total <= 0:
        return np.nan

    p = (
        x[
            x > 0
        ]
        / total
    )

    h = -float(
        np.sum(
            p
            * np.log(
                p
            )
        )
    )

    return (
        h
        / math.log(
            3.0
        )
    )


def build_digital_phenotype(
    sensor,
):
    rows = []

    for sid, g in sensor.groupby(
        "subjectID",
        sort=True,
    ):
        locomotor = g[
            "activity"
        ].isin(
            LOCOMOTOR_ACTIVITY_CODES
        )

        fog = (
            g[
                "fog"
            ]
            == 1
        )

        locomotor_samples = int(
            locomotor.sum()
        )

        locomotor_fog_samples = int(
            (
                locomotor
                & fog
            ).sum()
        )

        shuffling_samples = int(
            (
                fog
                & (
                    g[
                        "fog_severity"
                    ]
                    == 1
                )
            ).sum()
        )

        trembling_samples = int(
            (
                fog
                & (
                    g[
                        "fog_severity"
                    ]
                    == 2
                )
            ).sum()
        )

        akinesia_samples = int(
            (
                fog
                & (
                    g[
                        "fog_severity"
                    ]
                    == 3
                )
            ).sum()
        )

        fog_samples = (
            shuffling_samples
            + trembling_samples
            + akinesia_samples
        )

        kinetic_samples = (
            shuffling_samples
            + trembling_samples
        )

        if locomotor_samples <= 0:
            raise RuntimeError(
                f"Subject {sid} has no locomotor observation."
            )

        burden = (
            100.0
            * locomotor_fog_samples
            / locomotor_samples
        )

        akinetic_fraction = (
            100.0
            * akinesia_samples
            / fog_samples
            if fog_samples > 0
            else np.nan
        )

        trembling_fraction = (
            100.0
            * trembling_samples
            / kinetic_samples
            if kinetic_samples > 0
            else np.nan
        )

        entropy = normalized_entropy(
            [
                shuffling_samples,
                trembling_samples,
                akinesia_samples,
            ]
        )

        rows.append({
            "subjectID":
                int(
                    sid
                ),
            "locomotor_observation_samples":
                locomotor_samples,
            "locomotor_observation_seconds":
                locomotor_samples
                / FS,
            "locomotor_observation_minutes":
                locomotor_samples
                / FS
                / 60.0,
            "locomotor_fog_samples":
                locomotor_fog_samples,
            "locomotor_fog_seconds":
                locomotor_fog_samples
                / FS,
            "locomotor_fog_burden_pct":
                burden,
            "shuffling_samples":
                shuffling_samples,
            "shuffling_seconds":
                shuffling_samples
                / FS,
            "trembling_samples":
                trembling_samples,
            "trembling_seconds":
                trembling_samples
                / FS,
            "akinesia_samples":
                akinesia_samples,
            "akinesia_seconds":
                akinesia_samples
                / FS,
            "all_fog_samples":
                fog_samples,
            "all_fog_seconds":
                fog_samples
                / FS,
            "kinetic_samples":
                kinetic_samples,
            "kinetic_seconds":
                kinetic_samples
                / FS,
            "observed_fog":
                int(
                    fog_samples > 0
                ),
            "observed_kinetic":
                int(
                    kinetic_samples > 0
                ),
            "akinetic_fraction_of_fog_pct":
                akinetic_fraction,
            "trembling_fraction_of_kinetic_pct":
                trembling_fraction,
            "manifestation_entropy_normalized":
                entropy,
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# DESCRIPTIVES
# =============================================================================

def median_iqr_row(
    df,
    variable,
    population,
):
    x = pd.to_numeric(
        df[
            variable
        ],
        errors="coerce",
    ).dropna()

    q25, med, q75 = np.percentile(
        x,
        [
            25,
            50,
            75,
        ],
    )

    return {
        "population":
            population,
        "variable":
            variable,
        "N":
            len(x),
        "median":
            float(
                med
            ),
        "q25":
            float(
                q25
            ),
        "q75":
            float(
                q75
            ),
        "minimum":
            float(
                x.min()
            ),
        "maximum":
            float(
                x.max()
            ),
    }


def build_cohort_table(
    merged,
):
    variables = [
        "age",
        "disease_duration",
        "h_y",
        "updrs_iii",
        "fog_q",
        "moca",
        "fes_i",
        "pdq_8",
    ]

    rows = []

    populations = [
        (
            "overall",
            merged,
        ),
        (
            "observed_FoG",
            merged[
                merged[
                    "observed_fog"
                ]
                == 1
            ],
        ),
        (
            "no_observed_FoG",
            merged[
                merged[
                    "observed_fog"
                ]
                == 0
            ],
        ),
    ]

    for name, df in populations:
        for variable in variables:
            if variable not in df.columns:
                continue

            rows.append(
                median_iqr_row(
                    df,
                    variable,
                    name,
                )
            )

    return pd.DataFrame(
        rows
    )


def summarize_gender(
    merged,
):
    if (
        "gender"
        not in merged.columns
    ):
        return pd.DataFrame()

    values = (
        merged[
            "gender"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    rows = []

    for raw_value, n in values.value_counts(
        dropna=False
    ).items():
        rows.append({
            "raw_gender_value":
                raw_value,
            "N":
                int(
                    n
                ),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# SPEARMAN + BOOTSTRAP
# =============================================================================

def get_population(
    merged,
    population,
):
    if population == "all":
        return merged.copy()

    if population == "observed_fog":
        return merged[
            merged[
                "observed_fog"
            ]
            == 1
        ].copy()

    if population == "observed_kinetic":
        return merged[
            merged[
                "observed_kinetic"
            ]
            == 1
        ].copy()

    raise ValueError(
        population
    )


def safe_spearman(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(
            np.unique(
                x
            )
        )
        < 2
        or len(
            np.unique(
                y
            )
        )
        < 2
    ):
        return np.nan

    rho = spearmanr(
        x,
        y,
        nan_policy="omit",
    ).statistic

    return float(
        rho
    )


def bootstrap_spearman_ci(
    x,
    y,
    seed,
    n_boot=N_BOOTSTRAP,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    n = len(
        x
    )

    rng = np.random.default_rng(
        seed
    )

    values = []

    attempts = 0
    max_attempts = (
        n_boot
        * 4
    )

    while (
        len(
            values
        )
        < n_boot
        and attempts
        < max_attempts
    ):
        attempts += 1

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        rho = safe_spearman(
            x[
                idx
            ],
            y[
                idx
            ],
        )

        if np.isfinite(
            rho
        ):
            values.append(
                rho
            )

    if len(
        values
    ) < n_boot:
        raise RuntimeError(
            f"Bootstrap produced only {len(values)} valid replicates."
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    low, high = np.percentile(
        values,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(
            low
        ),
        float(
            high
        ),
    )


def run_association(
    merged,
    spec,
    association_index,
    family,
):
    df = get_population(
        merged,
        spec[
            "population"
        ],
    )

    complete = df[
        [
            "subjectID",
            spec[
                "biomarker"
            ],
            spec[
                "clinical"
            ],
        ]
    ].dropna()

    x = complete[
        spec[
            "biomarker"
        ]
    ].to_numpy(
        dtype=float,
    )

    y = complete[
        spec[
            "clinical"
        ]
    ].to_numpy(
        dtype=float,
    )

    rho = safe_spearman(
        x,
        y,
    )

    low, high = bootstrap_spearman_ci(
        x,
        y,
        seed=(
            BOOTSTRAP_SEED
            + association_index
            * 1009
        ),
    )

    return {
        "family":
            family,
        "label":
            spec[
                "label"
            ],
        "biomarker":
            spec[
                "biomarker"
            ],
        "clinical_variable":
            spec[
                "clinical"
            ],
        "population":
            spec[
                "population"
            ],
        "N":
            len(
                complete
            ),
        "spearman_rho":
            rho,
        "bootstrap_CI_low":
            low,
        "bootstrap_CI_high":
            high,
        "bootstrap_resamples":
            N_BOOTSTRAP,
        "p_value":
            np.nan,
    }


# =============================================================================
# REPRODUCTION AUDITS
# =============================================================================

def coverage_audit(
    digital,
):
    observed = {
        "n_subjects":
            len(
                digital
            ),
        "n_observed_fog":
            int(
                digital[
                    "observed_fog"
                ].sum()
            ),
        "n_no_observed_fog":
            int(
                (
                    digital[
                        "observed_fog"
                    ]
                    == 0
                ).sum()
            ),
        "n_observed_kinetic":
            int(
                digital[
                    "observed_kinetic"
                ].sum()
            ),
    }

    for key, expected in (
        EXPECTED_COVERAGE.items()
    ):
        obs = int(
            observed[
                key
            ]
        )

        print(
            f"{key:<24s}: "
            f"{obs} / {expected} "
            f"{'PASS' if obs == expected else 'FAIL'}"
        )

        if obs != expected:
            raise RuntimeError(
                f"Coverage mismatch for {key}."
            )


def exposure_audit(
    digital,
):
    x = digital[
        "locomotor_observation_minutes"
    ].to_numpy(
        dtype=float,
    )

    q25, med, q75 = np.percentile(
        x,
        [
            25,
            50,
            75,
        ],
    )

    observed = {
        "min":
            float(
                np.min(
                    x
                )
            ),
        "median":
            float(
                med
            ),
        "q25":
            float(
                q25
            ),
        "q75":
            float(
                q75
            ),
        "max":
            float(
                np.max(
                    x
                )
            ),
    }

    rows = []

    for key, expected in (
        EXPECTED_LOCOMOTOR_EXPOSURE_MINUTES.items()
    ):
        obs = observed[
            key
        ]

        diff = abs(
            obs
            - expected
        )

        passed = (
            diff
            <= EXPOSURE_TOL_MINUTES
        )

        rows.append({
            "statistic":
                key,
            "observed_minutes":
                obs,
            "expected_minutes":
                expected,
            "absolute_difference":
                diff,
            "tolerance":
                EXPOSURE_TOL_MINUTES,
            "pass":
                int(
                    passed
                ),
        })

        print(
            f"{key:<8s}: "
            f"obs={obs:.3f} min "
            f"expected≈{expected:.2f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "pass"
    ].all():
        raise RuntimeError(
            "Locomotor exposure audit failed."
        )

    return audit


def primary_rho_audit(
    association_df,
):
    rows = []

    primary = association_df[
        association_df[
            "family"
        ]
        == "primary"
    ]

    for key, expected in (
        EXPECTED_PRIMARY_RHO.items()
    ):
        biomarker, clinical = key

        r = primary[
            (
                primary[
                    "biomarker"
                ]
                == biomarker
            )
            & (
                primary[
                    "clinical_variable"
                ]
                == clinical
            )
        ]

        if len(
            r
        ) != 1:
            raise RuntimeError(
                f"Missing primary association {key}."
            )

        obs = float(
            r.iloc[0][
                "spearman_rho"
            ]
        )

        diff = abs(
            obs
            - expected
        )

        passed = (
            diff
            <= RHO_REPRODUCTION_TOL
        )

        ci_ref = PRIMARY_CI_REFERENCE[
            key
        ]

        rows.append({
            "biomarker":
                biomarker,
            "clinical_variable":
                clinical,
            "observed_rho":
                obs,
            "expected_rho":
                expected,
            "absolute_difference":
                diff,
            "rho_tolerance":
                RHO_REPRODUCTION_TOL,
            "reference_CI_low":
                ci_ref[
                    0
                ],
            "reference_CI_high":
                ci_ref[
                    1
                ],
            "pass":
                int(
                    passed
                ),
        })

        print(
            f"{biomarker:<36s} vs {clinical:<12s} "
            f"rho={obs:.6f} expected≈{expected:.3f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "pass"
    ].all():
        raise RuntimeError(
            "Primary Spearman reproduction gate failed."
        )

    return audit


# =============================================================================
# MAIN
# =============================================================================

header(
    "STEP 06 — CLINICAL DIGITAL PHENOTYPE"
)

print(
    "Project root :",
    ROOT,
)

print(
    "Sensor file  :",
    SENSOR_PATH,
)

print(
    "Clinical file:",
    CLINICAL_PATH,
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
    "No classifier development occurs in Step 06."
)

print(
    "Spearman associations only; no p-values."
)

print(
    "Bootstrap:",
    N_BOOTSTRAP,
    "subject-level resamples.",
)


# -----------------------------------------------------------------------------
# 1. Load / schema
# -----------------------------------------------------------------------------

header(
    "1. RAW INPUT + CLINICAL SCHEMA AUDIT"
)

sensor = pd.read_csv(
    SENSOR_PATH
)

clinical_raw = pd.read_csv(
    CLINICAL_PATH
)

required_sensor = {
    "subjectID",
    "activity",
    "fog",
    "fog_severity",
}

missing = (
    required_sensor
    - set(
        sensor.columns
    )
)

if missing:
    raise RuntimeError(
        f"Missing sensor columns: {sorted(missing)}"
    )

sensor[
    "subjectID"
] = pd.to_numeric(
    sensor[
        "subjectID"
    ],
    errors="raise",
).astype(int)

clinical = canonicalize_clinical(
    clinical_raw
)

print(
    "Sensor rows:",
    len(
        sensor
    ),
)

print(
    "Sensor subjects:",
    sensor[
        "subjectID"
    ].nunique(),
)

print(
    "Clinical subjects:",
    clinical[
        "subjectID"
    ].nunique(),
)

print(
    "Canonical clinical columns:",
    list(
        clinical.columns
    ),
)

sensor_ids = set(
    sensor[
        "subjectID"
    ]
)

clinical_ids = set(
    clinical[
        "subjectID"
    ]
)

if sensor_ids != clinical_ids:
    raise RuntimeError(
        "Sensor/clinical subject IDs do not match."
    )

print(
    "Sensor/clinical subject-ID match: PASS"
)


# -----------------------------------------------------------------------------
# 2. Digital phenotype
# -----------------------------------------------------------------------------

header(
    "2. SUBJECT-LEVEL DIGITAL PHENOTYPE"
)

digital = build_digital_phenotype(
    sensor
)

coverage_audit(
    digital
)

digital.to_csv(
    PROCESSED_DIR
    / "step06_subject_digital_phenotype.csv",
    index=False,
)

print()
print(
    digital[
        [
            "subjectID",
            "locomotor_observation_minutes",
            "locomotor_fog_burden_pct",
            "observed_fog",
            "akinetic_fraction_of_fog_pct",
            "trembling_fraction_of_kinetic_pct",
            "manifestation_entropy_normalized",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# -----------------------------------------------------------------------------
# 3. Exposure audit
# -----------------------------------------------------------------------------

header(
    "3. PROTOCOL LOCOMOTOR EXPOSURE AUDIT"
)

exposure = exposure_audit(
    digital
)

exposure.to_csv(
    REPORT_DIR
    / "01_locomotor_exposure_reproduction_audit.csv",
    index=False,
)

print()
print(
    "IMPORTANT: exposure differs substantially across participants; "
    "the biomarker is protocol-observed burden, not real-world prevalence."
)


# -----------------------------------------------------------------------------
# 4. Merge clinical
# -----------------------------------------------------------------------------

header(
    "4. CLINICAL + DIGITAL MERGE"
)

merged = clinical.merge(
    digital,
    on="subjectID",
    how="inner",
    validate="one_to_one",
)

if len(
    merged
) != 22:
    raise RuntimeError(
        f"Expected 22 merged participants, found {len(merged)}."
    )

merged.to_csv(
    PROCESSED_DIR
    / "step06_clinical_digital_merged.csv",
    index=False,
)

print(
    "Merged participants:",
    len(
        merged
    ),
)


# -----------------------------------------------------------------------------
# 5. Clinical descriptives
# -----------------------------------------------------------------------------

header(
    "5. COHORT CLINICAL DESCRIPTIVES"
)

cohort_table = build_cohort_table(
    merged
)

cohort_table.to_csv(
    REPORT_DIR
    / "02_clinical_descriptives.csv",
    index=False,
)

print(
    cohort_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

gender_table = summarize_gender(
    merged
)

if len(
    gender_table
):
    gender_table.to_csv(
        REPORT_DIR
        / "03_gender_distribution.csv",
        index=False,
    )

    print()
    print(
        "Gender distribution:"
    )

    print(
        gender_table.to_string(
            index=False
        )
    )


# -----------------------------------------------------------------------------
# 6. Associations
# -----------------------------------------------------------------------------

header(
    "6. SPEARMAN ASSOCIATIONS WITH SUBJECT-LEVEL BOOTSTRAP"
)

rows = []

association_index = 0

subheader(
    "PRIMARY"
)

for spec in PRIMARY_ASSOCIATIONS:
    r = run_association(
        merged=merged,
        spec=spec,
        association_index=association_index,
        family="primary",
    )

    association_index += 1

    rows.append(
        r
    )

    print(
        f"{r['label']:<52s} "
        f"N={r['N']:>2} "
        f"rho={r['spearman_rho']:+.3f} "
        f"95% CI [{r['bootstrap_CI_low']:+.3f}, "
        f"{r['bootstrap_CI_high']:+.3f}]"
    )

subheader(
    "EXPLORATORY"
)

for spec in EXPLORATORY_ASSOCIATIONS:
    if (
        spec[
            "clinical"
        ]
        not in merged.columns
    ):
        continue

    r = run_association(
        merged=merged,
        spec=spec,
        association_index=association_index,
        family="exploratory",
    )

    association_index += 1

    rows.append(
        r
    )

    print(
        f"{r['label']:<52s} "
        f"N={r['N']:>2} "
        f"rho={r['spearman_rho']:+.3f} "
        f"95% CI [{r['bootstrap_CI_low']:+.3f}, "
        f"{r['bootstrap_CI_high']:+.3f}]"
    )

associations = pd.DataFrame(
    rows
)

associations.to_csv(
    REPORT_DIR
    / "04_clinical_associations.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# 7. Primary central-estimate reproduction gate
# -----------------------------------------------------------------------------

header(
    "7. PRIMARY ASSOCIATION REPRODUCTION GATE"
)

rho_audit = primary_rho_audit(
    associations
)

rho_audit.to_csv(
    REPORT_DIR
    / "05_primary_rho_reproduction_audit.csv",
    index=False,
)

print()
print(
    "PRIMARY CENTRAL-ESTIMATE GATE: PASS"
)

print()
print(
    "Bootstrap confidence intervals are recomputed with the fixed "
    "10,000-resample participant-level procedure."
)


# -----------------------------------------------------------------------------
# 8. Primary clinical table
# -----------------------------------------------------------------------------

header(
    "8. PRIMARY CLINICAL TABLE"
)

primary = associations[
    associations[
        "family"
    ]
    == "primary"
].copy()

primary_table = pd.DataFrame({
    "Digital biomarker":
        primary[
            "biomarker"
        ],
    "Clinical variable":
        primary[
            "clinical_variable"
        ],
    "Population":
        primary[
            "population"
        ],
    "N":
        primary[
            "N"
        ].astype(int),
    "Spearman rho":
        primary[
            "spearman_rho"
        ].map(
            lambda x:
                f"{x:.3f}"
        ),
    "95% bootstrap CI":
        primary.apply(
            lambda r:
                (
                    f"[{r['bootstrap_CI_low']:.3f}, "
                    f"{r['bootstrap_CI_high']:.3f}]"
                ),
            axis=1,
        ),
})

primary_table.to_csv(
    REPORT_DIR
    / "06_primary_clinical_table.csv",
    index=False,
)

print(
    primary_table.to_string(
        index=False
    )
)


# -----------------------------------------------------------------------------
# 9. Analysis lock
# -----------------------------------------------------------------------------

header(
    "9. STEP 06 ANALYSIS LOCK"
)

lock = {
    "step": "06",
    "biomarkers": {
        "locomotor_fog_burden_pct": "100 * FoG locomotor samples / all locomotor samples",
        "akinetic_fraction_of_fog_pct": "100 * akinesia samples / all FoG samples",
        "trembling_fraction_of_kinetic_pct": "100 * trembling samples / kinetic FoG samples",
        "manifestation_entropy_normalized": "Shannon entropy across three manifestations / log(3)",
    },
    "primary_associations": PRIMARY_ASSOCIATIONS,
    "bootstrap": {
        "n_resamples": N_BOOTSTRAP,
        "seed": BOOTSTRAP_SEED,
        "unit": "participant",
        "confidence_interval": "percentile 2.5%-97.5%",
    },
    "p_values": False,
}

with open(
    REPORT_DIR / "step06_analysis_lock.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(lock, f, indent=2)


# -----------------------------------------------------------------------------
# FINAL
# -----------------------------------------------------------------------------

print("Step 06 reproduction checks: PASS")
