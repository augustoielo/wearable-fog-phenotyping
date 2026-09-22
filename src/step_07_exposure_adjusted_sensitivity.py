#!/usr/bin/env python3
"""
Step 07 — Exposure-adjusted sensitivity analysis.

Assesses clinical associations after accounting for participant-level locomotor exposure duration.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

N_BOOTSTRAP = 10_000

# Keep the same base bootstrap seed used in Step 06.
BOOTSTRAP_SEED = 20260909

# Separate deterministic offsets for the three sensitivity analyses.
BOOTSTRAP_SEED_OFFSETS = {
    "burden_vs_exposure": 50_000,
    "partial_fogq": 51_009,
    "partial_updrs_iii": 52_018,
}

EXPECTED_N = 22

# Same Step-06 reproduction targets.
EXPECTED_PRIMARY_RHO = {
    "fog_q": 0.536,
    "updrs_iii": 0.785,
}

RHO_REPRODUCTION_TOL = 0.0025

# Same Step-06 exposure gate.
EXPECTED_LOCOMOTOR_EXPOSURE_MINUTES = {
    "min": 1.19,
    "median": 3.09,
    "q25": 1.84,
    "q75": 3.66,
    "max": 7.02,
}

EXPOSURE_TOL_MINUTES = 0.03


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

STEP06_PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pipeline"
    / "step_06_clinical_digital_phenotype"
)

INPUT_PATH = (
    STEP06_PROCESSED_DIR
    / "step06_clinical_digital_merged.csv"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pipeline"
    / "step_07_exposure_adjusted_sensitivity"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "pipeline"
    / "step_07_exposure_adjusted_sensitivity"
)

for p in [PROCESSED_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Required Step-06 participant-level input was not found:\n"
        f"{INPUT_PATH}\n\n"
        "Run Step 06 first. That step creates "
        "'step06_clinical_digital_merged.csv'."
    )


# =============================================================================
# PRINT HELPERS
# =============================================================================

def header(title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def safe_spearman(x, y):
    """
    Spearman rho only. No p-value is retained or reported.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if (
        len(x) < 3
        or len(np.unique(x)) < 2
        or len(np.unique(y)) < 2
    ):
        return np.nan

    return float(
        spearmanr(
            x,
            y,
            nan_policy="omit",
        ).statistic
    )


def partial_spearman_one_covariate(x, y, z):
    """
    Partial Spearman correlation between x and y controlling for z.

    Implementation:
    1. average-rank transform x, y, and z;
    2. residualize ranked x on ranked z using OLS with intercept;
    3. residualize ranked y on ranked z using OLS with intercept;
    4. compute Pearson correlation between the two residual vectors.

    This is partial Pearson correlation applied to ranks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]

    if len(x) < 4:
        return np.nan

    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rz = rankdata(z, method="average")

    if (
        len(np.unique(rx)) < 2
        or len(np.unique(ry)) < 2
        or len(np.unique(rz)) < 2
    ):
        return np.nan

    design = np.column_stack(
        [
            np.ones(len(rz), dtype=float),
            rz,
        ]
    )

    beta_x = np.linalg.lstsq(
        design,
        rx,
        rcond=None,
    )[0]

    beta_y = np.linalg.lstsq(
        design,
        ry,
        rcond=None,
    )[0]

    resid_x = rx - design @ beta_x
    resid_y = ry - design @ beta_y

    if (
        np.std(resid_x, ddof=0) == 0
        or np.std(resid_y, ddof=0) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            resid_x,
            resid_y,
        )[0, 1]
    )


def bootstrap_ci(
    x,
    y,
    z=None,
    *,
    statistic,
    seed,
    n_boot=N_BOOTSTRAP,
):
    """
    Participant-level nonparametric bootstrap.

    The full statistic, including rank transformation and residualization for
    partial Spearman, is recomputed after each participant resample.

    Invalid resamples are discarded and redrawn until exactly n_boot valid
    replicates are obtained, mirroring the Step-06 bootstrap style.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if z is not None:
        z = np.asarray(z, dtype=float)

    n = len(x)

    if len(y) != n:
        raise RuntimeError("x and y lengths do not match.")

    if z is not None and len(z) != n:
        raise RuntimeError("x and z lengths do not match.")

    rng = np.random.default_rng(seed)

    values = []
    attempts = 0
    max_attempts = n_boot * 4

    while len(values) < n_boot and attempts < max_attempts:
        attempts += 1

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        if z is None:
            value = statistic(
                x[idx],
                y[idx],
            )
        else:
            value = statistic(
                x[idx],
                y[idx],
                z[idx],
            )

        if np.isfinite(value):
            values.append(value)

    if len(values) < n_boot:
        raise RuntimeError(
            f"Bootstrap produced only {len(values)} valid replicates "
            f"out of the requested {n_boot}."
        )

    values = np.asarray(values, dtype=float)

    low, high = np.percentile(
        values,
        [2.5, 97.5],
    )

    return (
        float(low),
        float(high),
        values,
        attempts,
    )


# =============================================================================
# REPRODUCTION GATES
# =============================================================================

def exposure_reproduction_gate(exposure_minutes):
    x = np.asarray(
        exposure_minutes,
        dtype=float,
    )

    q25, median, q75 = np.percentile(
        x,
        [25, 50, 75],
    )

    observed = {
        "min": float(np.min(x)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "max": float(np.max(x)),
    }

    rows = []

    for key, expected in EXPECTED_LOCOMOTOR_EXPOSURE_MINUTES.items():
        obs = observed[key]
        diff = abs(obs - expected)
        passed = diff <= EXPOSURE_TOL_MINUTES

        rows.append(
            {
                "statistic": key,
                "observed_minutes": obs,
                "expected_minutes": expected,
                "absolute_difference": diff,
                "tolerance_minutes": EXPOSURE_TOL_MINUTES,
                "pass": int(passed),
            }
        )

        print(
            f"{key:<10s} "
            f"observed={obs:.6f} "
            f"expected≈{expected:.2f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    audit = pd.DataFrame(rows)

    if not audit["pass"].all():
        raise RuntimeError(
            "Step-06 locomotor-exposure reproduction gate failed."
        )

    return audit, observed


def primary_rho_reproduction_gate(df):
    rows = []

    for clinical, expected in EXPECTED_PRIMARY_RHO.items():
        complete = df[
            [
                "locomotor_fog_burden_pct",
                clinical,
            ]
        ].dropna()

        observed = safe_spearman(
            complete["locomotor_fog_burden_pct"].to_numpy(float),
            complete[clinical].to_numpy(float),
        )

        diff = abs(observed - expected)
        passed = diff <= RHO_REPRODUCTION_TOL

        rows.append(
            {
                "clinical_variable": clinical,
                "N": len(complete),
                "observed_rho": observed,
                "expected_rho": expected,
                "absolute_difference": diff,
                "rho_tolerance": RHO_REPRODUCTION_TOL,
                "pass": int(passed),
            }
        )

        print(
            f"locomotor_fog_burden_pct vs {clinical:<12s} "
            f"rho={observed:.6f} "
            f"expected≈{expected:.3f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    audit = pd.DataFrame(rows)

    if not audit["pass"].all():
        raise RuntimeError(
            "Step-06 primary Spearman reproduction gate failed."
        )

    return audit


# =============================================================================
# LOAD STEP-06 PARTICIPANT-LEVEL TABLE
# =============================================================================

header("1. LOAD FROZEN STEP-06 PARTICIPANT-LEVEL TABLE")

df = pd.read_csv(INPUT_PATH)

required_columns = {
    "subjectID",
    "locomotor_observation_minutes",
    "locomotor_fog_burden_pct",
    "fog_q",
    "updrs_iii",
}

missing = required_columns - set(df.columns)

if missing:
    raise RuntimeError(
        "Required columns are missing from the Step-06 merged table:\n"
        f"{sorted(missing)}"
    )

if len(df) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} participants, found {len(df)}."
    )

if df["subjectID"].duplicated().any():
    duplicated = (
        df.loc[
            df["subjectID"].duplicated(keep=False),
            "subjectID",
        ]
        .tolist()
    )

    raise RuntimeError(
        f"Duplicate participant IDs found: {duplicated}"
    )

for col in [
    "locomotor_observation_minutes",
    "locomotor_fog_burden_pct",
    "fog_q",
    "updrs_iii",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

analysis_df = (
    df[
        [
            "subjectID",
            "locomotor_observation_minutes",
            "locomotor_fog_burden_pct",
            "fog_q",
            "updrs_iii",
        ]
    ]
    .copy()
    .sort_values("subjectID")
    .reset_index(drop=True)
)

if analysis_df[
    [
        "locomotor_observation_minutes",
        "locomotor_fog_burden_pct",
        "fog_q",
        "updrs_iii",
    ]
].isna().any().any():
    raise RuntimeError(
        "Missing values were found in variables required for this sensitivity "
        "analysis. Step 06 indicates that FoG-Q and MDS-UPDRS III should be "
        "complete for all 22 participants."
    )

if not analysis_df["locomotor_fog_burden_pct"].between(0, 100).all():
    raise RuntimeError(
        "locomotor_fog_burden_pct contains values outside [0, 100]."
    )

if not (analysis_df["locomotor_observation_minutes"] > 0).all():
    raise RuntimeError(
        "locomotor_observation_minutes must be positive for all participants."
    )

print(f"Input: {INPUT_PATH}")
print(f"Participants: {len(analysis_df)}")
print("Participant-level Step-06 input audit: PASS")


# =============================================================================
# STEP-06 REPRODUCTION GATES
# =============================================================================

header("2. STEP-06 REPRODUCTION GATES")

exposure_audit, exposure_summary = exposure_reproduction_gate(
    analysis_df["locomotor_observation_minutes"]
)

rho_audit = primary_rho_reproduction_gate(
    analysis_df
)

exposure_audit.to_csv(
    REPORT_DIR
    / "01_exposure_reproduction_audit.csv",
    index=False,
)

rho_audit.to_csv(
    REPORT_DIR
    / "02_primary_rho_reproduction_audit.csv",
    index=False,
)

print()
print("STEP-06 REPRODUCTION GATES: PASS")


# =============================================================================
# POST HOC EXPOSURE SENSITIVITY
# =============================================================================

header("3. POST HOC LOCOMOTOR-EXPOSURE SENSITIVITY")

burden = analysis_df[
    "locomotor_fog_burden_pct"
].to_numpy(float)

exposure = analysis_df[
    "locomotor_observation_minutes"
].to_numpy(float)

fogq = analysis_df[
    "fog_q"
].to_numpy(float)

updrs = analysis_df[
    "updrs_iii"
].to_numpy(float)


# 3A. Is burden itself associated with exposure duration?
rho_burden_exposure = safe_spearman(
    burden,
    exposure,
)

(
    ci_burden_exposure_low,
    ci_burden_exposure_high,
    boot_burden_exposure,
    attempts_burden_exposure,
) = bootstrap_ci(
    burden,
    exposure,
    statistic=safe_spearman,
    seed=(
        BOOTSTRAP_SEED
        + BOOTSTRAP_SEED_OFFSETS["burden_vs_exposure"]
    ),
)


# 3B. Burden vs FoG-Q controlling for locomotor exposure.
partial_rho_fogq = partial_spearman_one_covariate(
    burden,
    fogq,
    exposure,
)

(
    ci_partial_fogq_low,
    ci_partial_fogq_high,
    boot_partial_fogq,
    attempts_partial_fogq,
) = bootstrap_ci(
    burden,
    fogq,
    z=exposure,
    statistic=partial_spearman_one_covariate,
    seed=(
        BOOTSTRAP_SEED
        + BOOTSTRAP_SEED_OFFSETS["partial_fogq"]
    ),
)


# 3C. Burden vs MDS-UPDRS III controlling for locomotor exposure.
partial_rho_updrs = partial_spearman_one_covariate(
    burden,
    updrs,
    exposure,
)

(
    ci_partial_updrs_low,
    ci_partial_updrs_high,
    boot_partial_updrs,
    attempts_partial_updrs,
) = bootstrap_ci(
    burden,
    updrs,
    z=exposure,
    statistic=partial_spearman_one_covariate,
    seed=(
        BOOTSTRAP_SEED
        + BOOTSTRAP_SEED_OFFSETS["partial_updrs_iii"]
    ),
)


results = pd.DataFrame(
    [
        {
            "analysis":
                "Locomotor FoG burden vs locomotor exposure duration",
            "method":
                "Spearman",
            "N":
                EXPECTED_N,
            "rho":
                rho_burden_exposure,
            "bootstrap_CI_low":
                ci_burden_exposure_low,
            "bootstrap_CI_high":
                ci_burden_exposure_high,
            "bootstrap_resamples":
                N_BOOTSTRAP,
            "role":
                "additional exposure-adjusted sensitivity",
        },
        {
            "analysis":
                "Locomotor FoG burden vs FoG-Q controlling for locomotor exposure",
            "method":
                "Partial Spearman",
            "N":
                EXPECTED_N,
            "rho":
                partial_rho_fogq,
            "bootstrap_CI_low":
                ci_partial_fogq_low,
            "bootstrap_CI_high":
                ci_partial_fogq_high,
            "bootstrap_resamples":
                N_BOOTSTRAP,
            "role":
                "additional exposure-adjusted sensitivity",
        },
        {
            "analysis":
                "Locomotor FoG burden vs MDS-UPDRS III controlling for locomotor exposure",
            "method":
                "Partial Spearman",
            "N":
                EXPECTED_N,
            "rho":
                partial_rho_updrs,
            "bootstrap_CI_low":
                ci_partial_updrs_low,
            "bootstrap_CI_high":
                ci_partial_updrs_high,
            "bootstrap_resamples":
                N_BOOTSTRAP,
            "role":
                "additional exposure-adjusted sensitivity",
        },
    ]
)

results.to_csv(
    REPORT_DIR
    / "03_exposure_adjusted_sensitivity_results.csv",
    index=False,
)


# =============================================================================
# SAVE PARTICIPANT-LEVEL AUDIT + BOOTSTRAP DISTRIBUTIONS
# =============================================================================

analysis_df.to_csv(
    PROCESSED_DIR
    / "step07_participant_level_analysis_input.csv",
    index=False,
)

bootstrap_df = pd.DataFrame(
    {
        "bootstrap_id":
            np.arange(1, N_BOOTSTRAP + 1),
        "rho_burden_vs_exposure":
            boot_burden_exposure,
        "partial_rho_burden_fogq_given_exposure":
            boot_partial_fogq,
        "partial_rho_burden_updrs_iii_given_exposure":
            boot_partial_updrs,
    }
)

bootstrap_df.to_csv(
    PROCESSED_DIR
    / "step07_bootstrap_distributions.csv",
    index=False,
)


# =============================================================================
# CONFIG / METHOD LOCK
# =============================================================================

method_lock = {
    "analysis_role":
        "additional exposure-adjusted sensitivity",
    "input":
        str(INPUT_PATH),
    "input_is_frozen_step06_participant_level_table":
        True,
    "N":
        EXPECTED_N,
    "exposure_variable":
        "locomotor_observation_minutes",
    "burden_variable":
        "locomotor_fog_burden_pct",
    "clinical_variables":
        [
            "fog_q",
            "updrs_iii",
        ],
    "partial_spearman_definition":
        (
            "Average-rank transform burden, clinical variable, and locomotor "
            "exposure; separately residualize ranked burden and ranked clinical "
            "variable on ranked exposure using OLS with intercept; Pearson "
            "correlate the two residual vectors."
        ),
    "bootstrap":
        {
            "unit":
                "participant",
            "resamples":
                N_BOOTSTRAP,
            "CI":
                "95% percentile",
            "base_seed":
                BOOTSTRAP_SEED,
            "seed_offsets":
                BOOTSTRAP_SEED_OFFSETS,
            "full_statistic_recomputed_within_each_resample":
                True,
        },
    "p_values":
        False,
    "primary_step06_associations_replaced":
        False,
    "burden_definition_changed":
        False,
}

with open(
    REPORT_DIR
    / "04_exposure_sensitivity_method_lock.json",
    "w",
) as f:
    json.dump(
        method_lock,
        f,
        indent=2,
    )


# =============================================================================
# FINAL CONSOLE OUTPUT
# =============================================================================

header("4. FINAL RESULTS")

print(
    "Locomotor exposure: "
    f"median {exposure_summary['median']:.3f} min "
    f"[IQR {exposure_summary['q25']:.3f}–"
    f"{exposure_summary['q75']:.3f}], "
    f"range {exposure_summary['min']:.3f}–"
    f"{exposure_summary['max']:.3f} min."
)

print()

print(
    "Burden vs locomotor exposure duration: "
    f"rho={rho_burden_exposure:.3f}, "
    f"95% bootstrap CI "
    f"[{ci_burden_exposure_low:.3f}, "
    f"{ci_burden_exposure_high:.3f}]."
)

print(
    "Burden vs FoG-Q controlling for locomotor exposure: "
    f"partial rho={partial_rho_fogq:.3f}, "
    f"95% bootstrap CI "
    f"[{ci_partial_fogq_low:.3f}, "
    f"{ci_partial_fogq_high:.3f}]."
)

print(
    "Burden vs MDS-UPDRS III controlling for locomotor exposure: "
    f"partial rho={partial_rho_updrs:.3f}, "
    f"95% bootstrap CI "
    f"[{ci_partial_updrs_low:.3f}, "
    f"{ci_partial_updrs_high:.3f}]."
)

print()
print(
    "Primary unadjusted associations:"
)
print(
    "  burden vs FoG-Q: rho=0.536"
)
print(
    "  burden vs MDS-UPDRS III: rho=0.785"
)

print()
print("Step 07 sensitivity analysis: PASS")
