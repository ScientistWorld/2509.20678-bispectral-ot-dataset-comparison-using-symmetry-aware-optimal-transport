# Progress

## What Works

- Read the paper and upstream code release.
- Implemented the paper's Bispectral OT pipeline: polar angular sampling, per-radius cyclic bispectrum, per-image feature normalization, Greenkhorn entropic OT, and coupling-based class-preservation scoring.
- Added portable MNIST and KMNIST IDX downloads under `scripts/download.sh`; the files are already present in `data/downloads/` for the submitted job.
- Added a method-independent evaluator in `eval/` that reads saved coupling artifacts and writes `scoring/scores.json`.
- `python validate.py --reference-only` and `python validate.py --compare` pass.
- The evaluator now merges multiple saved manifests into one `scores.json`, so multi-dataset jobs can be scored together.

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

## Remaining

- Increase the MNIST/KMNIST sample scale to 50 examples per class.
- Add more Table 1 metrics or datasets if budget permits.

## Issues

- `/home/user/shared/datasets` is read-only in this session, so the downloaded public IDX files are stored in workspace-local `data/downloads/`. `scripts/download.sh` is self-contained and can repopulate them.

## Deviations from Paper

- Scale: the paper runs OT on full training halves of each dataset. The completed jobs use 20 examples per class to validate the exact algorithm under scheduler control. This scales resources, not procedure.
- Libraries: rotation uses `scipy.ndimage.rotate` with bilinear interpolation instead of torchvision because this container does not need PyTorch for the scaled reproduction. The rotation distribution and no-expand geometry are preserved.
