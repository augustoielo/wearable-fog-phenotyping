# Frozen reference outputs

This directory contains selected non-sensitive numerical outputs used for reproduction checks. File names identify the pipeline step that produced each result.

- `step01_stage_reference_eligibility.csv`: stage-level participant and class counts.
- `step03_traditional_ml_performance_summary.csv`: five-fold traditional ML performance summary.
- `step04_cnn_seed_fold_metrics.csv`: seed-specific CNN fold metrics.
- `step04_cnn_seed_ensemble_outer_fold_metrics.csv`: CNN metrics after probability-level seed ensembling within each outer fold.
- `step04_cnn_performance_summary.csv`: five-fold CNN performance summary.
- `step05_mixed_subject_nested_loso_summary.csv`: subject-equal nested LOSO sensitivity summary.
- `step06_primary_clinical_associations.csv`: primary participant-level clinical associations.
- `step07_exposure_adjusted_sensitivity.csv`: locomotor-exposure and partial-Spearman sensitivity results.
- `step08_stage1_within_activity_metrics.csv`: Stage-1 metrics stratified by locomotor activity.
- `step08_stage1_activity_equal_summary.csv`: activity-equal descriptive Stage-1 summary.
- `step09_participant_window_counts.csv`: participant-by-class window counts for Stages 1--3.

These files are reference values only and are not inputs to model fitting.
