# Progress

## What Works

- Read the paper and upstream code release.
- Implemented the paper's Bispectral OT pipeline: polar angular sampling, per-radius cyclic bispectrum, per-image feature normalization, Greenkhorn entropic OT, and coupling-based class-preservation scoring.
- Added portable MNIST and KMNIST IDX downloads under `scripts/download.sh`; the files are already present in `data/downloads/` for the submitted job.
- Added a method-independent evaluator in `eval/` that reads saved coupling artifacts and writes `scoring/scores.json`.
- `python validate.py --reference-only` and `python validate.py --compare` pass.
- The evaluator now merges multiple saved manifests into one `scores.json`, so multi-dataset jobs can be scored together.
- Added Fashion-MNIST IDX download support and method/baseline dataset support, extending the reusable benchmark to a third Table 1 dataset.
- Added EMNIST letters download extraction and dataset support; labels are converted from the official 1..26 encoding to 0..25 for evaluation.
- Expanded `scoring/reference.json` from L1-only Table 1 coverage to the paper's full raw/BOT distance-metric matrix: L1, L2, squared L2, and cosine for each rotated and unrotated condition.
- Updated the evaluator so separate manifests can contribute disjoint method results to the same experiment; evaluation still reads saved coupling artifacts and does not import `method/`.

## Results

- Job `3da96518-891` completed successfully on CPU.
- Scaled MNIST rotated matching, 20 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.305.
  - `bispectral_ot_l1`: class-preservation accuracy 0.605.
  - Paper Table 1 reports 0.3297 for raw L1 and 0.8405 for BOT L1 at full scale.
- Scaled MNIST unrotated control, 20 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.815.
  - `bispectral_ot_l1`: class-preservation accuracy 0.635.
  - Paper Table 1 reports 0.9725 for raw L1 and 0.8603 for BOT L1 at full scale.
- Job `1d3643cc-26f` completed successfully on CPU in about 60 seconds and merged MNIST+KMNIST scores.
- Scaled KMNIST rotated matching, 20 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.210.
  - `bispectral_ot_l1`: class-preservation accuracy 0.435.
  - Paper Table 1 reports 0.2420 for raw L1 and 0.7815 for BOT L1 at full scale.
- Scaled KMNIST unrotated control, 20 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.690.
  - `bispectral_ot_l1`: class-preservation accuracy 0.435.
  - Paper Table 1 reports 0.9724 for raw L1 and 0.8143 for BOT L1 at full scale.
- Job `55a1ecdd-4af` completed successfully on CPU in about 60 seconds at 50 examples per class.
- Scaled MNIST rotated matching, 50 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.310.
  - `bispectral_ot_l1`: class-preservation accuracy 0.674.
- Scaled MNIST unrotated control, 50 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.814.
  - `bispectral_ot_l1`: class-preservation accuracy 0.652.
- Scaled KMNIST rotated matching, 50 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.218.
  - `bispectral_ot_l1`: class-preservation accuracy 0.492.
- Scaled KMNIST unrotated control, 50 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.800.
  - `bispectral_ot_l1`: class-preservation accuracy 0.514.
- Job `2e3b5935-44d` completed successfully on CPU in about 4 minutes at 100 examples per class.
- Scaled MNIST rotated matching, 100 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.297.
  - `bispectral_ot_l1`: class-preservation accuracy 0.694.
- Scaled MNIST unrotated control, 100 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.869.
  - `bispectral_ot_l1`: class-preservation accuracy 0.699.
- Scaled KMNIST rotated matching, 100 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.203.
  - `bispectral_ot_l1`: class-preservation accuracy 0.564.
- Scaled KMNIST unrotated control, 100 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.851.
  - `bispectral_ot_l1`: class-preservation accuracy 0.561.
- Job `48c562eb-931` completed successfully on CPU and added Fashion-MNIST at 100 examples per class.
- Scaled Fashion-MNIST rotated matching, 100 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.262.
  - `bispectral_ot_l1`: class-preservation accuracy 0.709.
  - Paper Table 1 reports 0.3003 for raw L1 and 0.7617 for BOT L1 at full scale.
- Scaled Fashion-MNIST unrotated control, 100 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.792.
  - `bispectral_ot_l1`: class-preservation accuracy 0.697.
  - Paper Table 1 reports 0.8726 for raw L1 and 0.7982 for BOT L1 at full scale.
- Job `23fc5045-020` completed successfully on CPU and added EMNIST letters at 50 examples per class.
- Scaled EMNIST rotated matching, 50 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.1415.
  - `bispectral_ot_l1`: class-preservation accuracy 0.3792.
  - Paper Table 1 reports 0.1969 for raw L1 and 0.5983 for BOT L1 at full scale.
- Scaled EMNIST unrotated control, 50 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.6169.
  - `bispectral_ot_l1`: class-preservation accuracy 0.3862.
  - Paper Table 1 reports 0.8754 for raw L1 and 0.6416 for BOT L1 at full scale.
- Job `b1483a5b-018` completed successfully on CPU and scaled EMNIST letters to 100 examples per class.
- Scaled EMNIST rotated matching, 100 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.1692.
  - `bispectral_ot_l1`: class-preservation accuracy 0.4188.
  - Paper Table 1 reports 0.1969 for raw L1 and 0.5983 for BOT L1 at full scale.
- Scaled EMNIST unrotated control, 100 examples per class:
  - `raw_ot_l1`: class-preservation accuracy 0.6912.
  - `bispectral_ot_l1`: class-preservation accuracy 0.4346.
  - Paper Table 1 reports 0.8754 for raw L1 and 0.6416 for BOT L1 at full scale.

## Remaining

- Full-scale Table 1 evaluation on complete dataset halves remains open.
- Reproduced L2, squared L2, and cosine distance variants are queued for the next compute job.
- Appendix confusion matrices remain open.

## Issues

- `/home/user/shared/datasets` is read-only in this session, so the downloaded public IDX files are stored in workspace-local `data/downloads/`. `scripts/download.sh` is self-contained and can repopulate them.
- A direct login-node import smoke test for Fashion-MNIST failed because the host Python lacks NumPy; this is covered by `environment/setup.sh` and will be tested inside the compute container.
- The job log paths reported by the scheduler were not visible from this shell, but the jobs updated `outputs/` and `scoring/scores.json`, and `python validate.py --compare` passed after each run.
- A direct login-node smoke test of the expanded method path cannot run because the host Python lacks NumPy. This is expected for this workspace; `environment/setup.sh` installs NumPy, SciPy, and POT in the compute container overlay.

## Deviations from Paper

- Scale: the paper runs OT on full training halves of each dataset. The completed jobs use up to 100 examples per class to validate the exact algorithm under scheduler control. This scales resources, not procedure.
- EMNIST was first run at 50 examples per class because it has 26 classes, then scaled to 100 examples per class after the smaller job completed quickly.
- Libraries: rotation uses `scipy.ndimage.rotate` with bilinear interpolation instead of torchvision because this container does not need PyTorch for the scaled reproduction. The rotation distribution and no-expand geometry are preserved.
