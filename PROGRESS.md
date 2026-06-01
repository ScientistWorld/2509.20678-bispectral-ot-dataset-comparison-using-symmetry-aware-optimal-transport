# Progress

## What Works

- Read the paper and upstream code release.
- Implemented the paper's Bispectral OT pipeline: polar angular sampling, per-radius cyclic bispectrum, per-image feature normalization, Greenkhorn entropic OT, and coupling-based class-preservation scoring.
- Added portable MNIST and KMNIST IDX downloads under `scripts/download.sh`; the files are already present in `data/downloads/` for the submitted job.
- Added a method-independent evaluator in `eval/` that reads saved coupling artifacts and writes `scoring/scores.json`.
- `python validate.py --reference-only` and `python validate.py --compare` pass.
- The evaluator now merges multiple saved manifests into one `scores.json`, so multi-dataset jobs can be scored together.
- Added Fashion-MNIST IDX download support and method/baseline dataset support, extending the reusable benchmark to a third Table 1 dataset.

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

## Remaining

- Run the expanded MNIST/KMNIST/Fashion-MNIST job and update `scoring/scores.json` with all six configured experiments.
- Full-scale Table 1 evaluation on complete MNIST/KMNIST training halves remains open.
- EMNIST remains open.

## Issues

- `/home/user/shared/datasets` is read-only in this session, so the downloaded public IDX files are stored in workspace-local `data/downloads/`. `scripts/download.sh` is self-contained and can repopulate them.
- A direct login-node import smoke test for Fashion-MNIST failed because the host Python lacks NumPy; this is covered by `environment/setup.sh` and will be tested inside the compute container.

## Deviations from Paper

- Scale: the paper runs OT on full training halves of each dataset. The completed jobs use up to 100 examples per class to validate the exact algorithm under scheduler control. This scales resources, not procedure.
- Libraries: rotation uses `scipy.ndimage.rotate` with bilinear interpolation instead of torchvision because this container does not need PyTorch for the scaled reproduction. The rotation distribution and no-expand geometry are preserved.
