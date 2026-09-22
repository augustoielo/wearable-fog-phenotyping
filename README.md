# Wearable inertial phenotyping of freezing of gait

Minimal reproducibility repository for the analysis supporting:

**Wearable inertial phenotyping of freezing of gait in Parkinson's disease: biomechanical signatures, subject-independent classification, and clinical associations**

This repository contains the numerical analysis pipeline, frozen analysis configuration, and selected non-sensitive reference outputs. It intentionally does **not** redistribute the FoG-STAR dataset, manuscript files, figures, or plotting-only code.

## Data

The analyses use the public **FoG-STAR** dataset:

- Zenodo DOI: `10.5281/zenodo.17838806`
- License: Creative Commons Attribution 4.0 International

Download the dataset separately and place the two files expected by the analysis scripts at:

```text
data/raw/sensor_data.csv
data/raw/clinical_data.csv
```

The raw dataset is not included in this repository.

## Environment

The manuscript analyses were implemented in Python 3.12. The required Python packages are listed in `requirements.txt`.

A minimal setup is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch will use CUDA, Apple MPS, or CPU when available according to the logic in the CNN script.

## Frozen configuration

`configs/frozen_analysis_config.json` records the main fixed analysis choices, including:

- right-ankle input channels and 60 Hz sampling;
- 1-s deterministic reference windows;
- five participant-independent outer test folds;
- inner-validation participant sets;
- the nine interpretable biomechanical features;
- traditional machine-learning hyperparameter grids;
- CNN architecture, optimizer, early stopping, seeds 42--46, and seed-ensemble rule;
- mixed-subject nested LOSO eligibility;
- clinical bootstrap settings;
- exposure-adjusted and activity-context sensitivity specifications.

The scripts also contain reproduction sentinels that check key frozen counts and numerical outputs.

## Analysis scripts

Run scripts from the repository root in the following order:

```bash
python src/step_01_segments_windows_splits.py
python src/step_02_biomechanical_features.py
python src/step_03_traditional_ml_benchmark.py
python src/step_04_raw_temporal_cnn_benchmark.py
python src/step_05_mixed_subject_nested_loso.py
python src/step_06_clinical_digital_phenotype.py
python src/step_07_exposure_adjusted_sensitivity.py
python src/step_08_stage1_activity_context_sensitivity.py
python src/step_09_participant_window_counts.py
```

Scripts are numbered `01`--`09` in execution order within this repository. Plotting-only and document-synthesis utilities are not included because they are not required to reproduce the numerical analyses.

### Step 01 — windows and participant-independent splits

Reconstructs homogeneous signal segments, deterministic non-overlapping 1-s reference windows, complete right-ankle training blocks, and the frozen outer/inner participant partitions.

### Step 02 — interpretable biomechanical features

Computes the fixed nine-feature representation and the participant-level biomechanical summaries and within-participant phenotype contrasts.

### Step 03 — traditional machine-learning benchmark

Runs Logistic Regression, RBF-SVM, and Random Forest using the fixed feature representation and frozen participant-independent partitions. Hyperparameter selection uses only the inner-validation participants.

### Step 04 — raw temporal CNN benchmark

Runs the compact 1D CNN on raw 60 x 6 right-ankle windows. The script contains the full training specification, subject-balanced training-only normalization, dynamic training sampler, five random seeds (`42`--`46`), early stopping, and probability-level seed ensembling.

### Step 05 — mixed-subject nested LOSO sensitivity analysis

Runs the nested leave-one-subject-out Logistic Regression sensitivity analyses in participants exhibiting both target phenotypes.

### Step 06 — clinical digital phenotype analysis

Computes protocol-observed digital FoG measures and participant-level Spearman associations with 10,000 participant bootstrap resamples.

### Step 07 — exposure-adjusted sensitivity analysis

Quantifies the association between locomotor exposure and protocol-observed FoG burden and computes exposure-adjusted partial Spearman correlations.

### Step 08 — Stage-1 activity-context sensitivity analysis

Re-evaluates the frozen outer-test predictions within walking, right-turning, and left-turning strata. Activity labels are used only for post-prediction stratification.

### Step 09 — participant-level window counts

Produces participant-by-class deterministic reference-window counts for the three classification stages.

## Reference outputs

`reference_outputs/` contains a compact set of frozen numerical outputs from the analysis used for the manuscript. These files allow a reproduced run to be compared against the reported benchmark without distributing the full set of intermediate files.

The aggregate frozen reference sets are:

| Stage | Participants | Class 0 | Class 1 | Total windows |
|---|---:|---:|---:|---:|
| Stage 1: locomotor non-FoG vs FoG | 22 | 2556 | 986 | 3542 |
| Stage 2: kinetic vs akinetic FoG | 16 | 290 | 714 | 1004 |
| Stage 3: shuffling vs trembling | 15 | 83 | 207 | 290 |

## CNN specification

The included CNN uses:

```text
Conv1d(6 -> 32, kernel=5, padding=2)
ReLU
Conv1d(32 -> 64, kernel=5, padding=2)
ReLU
Adaptive global average pooling
Dropout(p=0.20)
Linear(64 -> 1)
```

Total trainable parameters: **11,361**.

Training uses AdamW (`lr=1e-3`, `weight_decay=1e-4`), `BCEWithLogitsLoss`, batch size 64, 1,024 dynamically sampled windows per epoch, a maximum of 40 epochs, and early-stopping patience 7. Five independently trained models use seeds 42--46. Outer-test probabilities are averaged across these five models within each outer fold and converted to hard predictions at a fixed threshold of 0.5.

## What is intentionally excluded

To keep this repository focused on reproducibility of the numerical analyses, it does not include:

- the FoG-STAR raw dataset;
- generated figures or plotting routines;
- manuscript or supplementary-document source files;
- manuscript-drafting or synthesis utilities;
- temporary files, local logs, caches, or operating-system metadata;
- intermediate analyses not used in the reported study.

## Citation

Citation metadata are provided in `CITATION.cff`.

## Contact

Correspondence: **Lilla Bonanno**, IRCCS Centro Neurolesi Bonino-Pulejo, Messina, Italy.
