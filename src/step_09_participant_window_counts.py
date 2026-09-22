#!/usr/bin/env python3
"""
Step 09 — Participant-level window counts.

Summarizes frozen deterministic reference-window counts by participant, stage, and class.
"""

from pathlib import Path
import pandas as pd

LOCOMOTOR_ACTIVITY_CODES = {1, 6, 7}

EXPECTED = {
    "stage1": (3542, 22, 2556, 986),   # total, subjects, non-FoG, FoG
    "stage2": (1004, 16, 290, 714),    # total, subjects, kinetic, akinetic
    "stage3": (290, 15, 83, 207),      # total, subjects, shuffling, trembling
}


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


def header(text):
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def pivot_counts(df, labels):
    return (
        df.groupby(["subjectID", "class_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
        .astype(int)
    )


ROOT = find_project_root()
REFERENCE_PATH = (
    ROOT / "data" / "processed" / "pipeline"
    / "step_01_segments_windows_splits" / "step01_reference_chunks_1s.csv"
)
REPORT_DIR = (
    ROOT / "reports" / "pipeline"
    / "step_09_participant_window_counts"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

header("1. LOAD FROZEN REFERENCE WINDOWS")
reference = pd.read_csv(REFERENCE_PATH)

required = {"chunk_id", "subjectID", "activity", "pool_state"}
missing = required - set(reference.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {sorted(missing)}")

reference["subjectID"] = reference["subjectID"].astype(int)

if reference["chunk_id"].duplicated().any():
    raise RuntimeError("Duplicate chunk_id values found in frozen reference manifest.")

print("Reference file :", REFERENCE_PATH)
print("Rows           :", len(reference))
print("Unique chunks  :", reference["chunk_id"].nunique())
print("Participants   :", reference["subjectID"].nunique())

header("2. APPLY FROZEN STAGE ELIGIBILITY")

stage1 = reference[
    reference["activity"].isin(LOCOMOTOR_ACTIVITY_CODES)
].copy()
stage1["class_label"] = stage1["pool_state"].map(
    lambda s: "S1_FoG" if s != "non_fog_locomotor" else "S1_non_FoG"
)

stage2 = reference[
    reference["pool_state"].isin(["shuffling", "trembling", "akinesia"])
].copy()
stage2["class_label"] = stage2["pool_state"].map(
    lambda s: "S2_akinetic" if s == "akinesia" else "S2_kinetic"
)

stage3 = reference[
    reference["pool_state"].isin(["shuffling", "trembling"])
].copy()
stage3["class_label"] = stage3["pool_state"].map(
    {"shuffling": "S3_shuffling", "trembling": "S3_trembling"}
)

header("3. REPRODUCTION GATES")
audit_rows = []

def gate(name, observed, expected):
    passed = observed == expected
    audit_rows.append(
        {"check": name, "observed": observed, "expected": expected, "pass": int(passed)}
    )
    print(
        f"{name:<40s} observed={str(observed):<8s} "
        f"expected={str(expected):<8s} {'PASS' if passed else 'FAIL'}"
    )
    return passed

ok = True

s1 = stage1["class_label"].value_counts().to_dict()
s2 = stage2["class_label"].value_counts().to_dict()
s3 = stage3["class_label"].value_counts().to_dict()

observed_sets = {
    "stage1": (
        len(stage1), stage1["subjectID"].nunique(),
        int(s1.get("S1_non_FoG", 0)), int(s1.get("S1_FoG", 0))
    ),
    "stage2": (
        len(stage2), stage2["subjectID"].nunique(),
        int(s2.get("S2_kinetic", 0)), int(s2.get("S2_akinetic", 0))
    ),
    "stage3": (
        len(stage3), stage3["subjectID"].nunique(),
        int(s3.get("S3_shuffling", 0)), int(s3.get("S3_trembling", 0))
    ),
}

labels = {
    "stage1": ["total windows", "participants", "non-FoG", "FoG"],
    "stage2": ["total windows", "participants", "kinetic", "akinetic"],
    "stage3": ["total windows", "participants", "shuffling", "trembling"],
}

for stage_key in ["stage1", "stage2", "stage3"]:
    for lab, obs, exp in zip(labels[stage_key], observed_sets[stage_key], EXPECTED[stage_key]):
        ok &= gate(f"{stage_key} {lab}", int(obs), int(exp))

header("4. PARTICIPANT-BY-CLASS COUNTS")

subjects = sorted(stage1["subjectID"].unique())

p1 = pivot_counts(stage1, ["S1_non_FoG", "S1_FoG"])
p2 = pivot_counts(stage2, ["S2_kinetic", "S2_akinetic"])
p3 = pivot_counts(stage3, ["S3_shuffling", "S3_trembling"])

table = pd.DataFrame(index=subjects)
table.index.name = "Participant"
table = (
    table.join(p1, how="left")
    .join(p2, how="left")
    .join(p3, how="left")
    .fillna(0)
    .astype(int)
    .reset_index()
)

kinetic_identity = (
    table["S2_kinetic"]
    == table["S3_shuffling"] + table["S3_trembling"]
).all()
ok &= gate("Per-subject kinetic identity", bool(kinetic_identity), True)

stage1_subset = (
    table["S1_FoG"]
    <= table["S2_kinetic"] + table["S2_akinetic"]
).all()
ok &= gate("Per-subject Stage-1 FoG subset", bool(stage1_subset), True)

totals = {
    "Participant": "Total",
    "S1_non_FoG": int(table["S1_non_FoG"].sum()),
    "S1_FoG": int(table["S1_FoG"].sum()),
    "S2_kinetic": int(table["S2_kinetic"].sum()),
    "S2_akinetic": int(table["S2_akinetic"].sum()),
    "S3_shuffling": int(table["S3_shuffling"].sum()),
    "S3_trembling": int(table["S3_trembling"].sum()),
}
table_out = pd.concat([table, pd.DataFrame([totals])], ignore_index=True)
print(table_out.to_string(index=False))

header("5. SAVE OUTPUTS")

csv_path = REPORT_DIR / "01_table_s4_participant_window_counts.csv"
table_out.to_csv(csv_path, index=False)

latex_rows = []
for _, r in table.iterrows():
    latex_rows.append(
        f"{int(r['Participant'])} & {int(r['S1_non_FoG'])} & {int(r['S1_FoG'])} & "
        f"{int(r['S2_kinetic'])} & {int(r['S2_akinetic'])} & "
        f"{int(r['S3_shuffling'])} & {int(r['S3_trembling'])} \\\\"
    )

latex_rows.append("\\midrule")
latex_rows.append(
    f"Total & {totals['S1_non_FoG']} & {totals['S1_FoG']} & "
    f"{totals['S2_kinetic']} & {totals['S2_akinetic']} & "
    f"{totals['S3_shuffling']} & {totals['S3_trembling']} \\\\"
)

latex_path = REPORT_DIR / "02_table_s4_latex_rows.txt"
latex_path.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")

audit_path = REPORT_DIR / "03_step09_reproduction_audit.csv"
pd.DataFrame(audit_rows).to_csv(audit_path, index=False)


header("6. FINAL STATUS")
if not ok:
    raise RuntimeError("Step 09 reproduction gate failed.")
print("STEP 09 REPRODUCTION GATE: PASS")
