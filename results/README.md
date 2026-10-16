One folder per variant, written by the training notebooks:

- `test_results.json` - mean and std of each metric, overall and per group (AD/MCI/CN/HCP)
- `test_results_per_subject.json` - every metric for each of the 43 test subjects
- `bidirectional_check_results.json`, `tractography_results.json`, `diagnostic_validity_results.json` - the three validation checks
- `history.csv` - training and validation loss per epoch

`V3_adversarial_baseline_pipeline/` was produced with the earlier voxel-size harmonization and is not directly comparable with V1, V2 and V4.
`alignment_qc.csv` - T1/DTI brain overlap and tensor-frame checks for the 32-subject QC sample.
