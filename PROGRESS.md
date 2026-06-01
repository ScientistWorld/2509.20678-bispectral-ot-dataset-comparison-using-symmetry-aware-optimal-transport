# Progress

## What Works

- Read the paper and upstream code release.
- Implemented the paper's Bispectral OT pipeline: polar angular sampling, per-radius cyclic bispectrum, per-image feature normalization, Greenkhorn entropic OT, and coupling-based class-preservation scoring.
- Added portable MNIST and KMNIST IDX downloads under `scripts/download.sh`; the files are already present in `data/downloads/` for the submitted job.
- Added a method-independent evaluator in `eval/` that reads saved coupling artifacts and writes `scoring/scores.json`.
- `python validate.py --reference-only` passes.

## Results

- No reproduced scores yet. The first scheduled job will run a scaled MNIST smoke test with 20 examples per class and compare against the Table 1 MNIST references.

## Remaining

- Run the first scheduled job to verify the method end-to-end and produce `scoring/scores.json`.
- If MNIST supports the core claim, expand to larger MNIST samples and KMNIST.
- Add more Table 1 metrics or datasets if budget permits.

## Issues

- `/home/user/shared/datasets` is read-only in this session, so the downloaded public IDX files are stored in workspace-local `data/downloads/`. `scripts/download.sh` is self-contained and can repopulate them.

## Deviations from Paper

- Scale: the paper runs OT on full training halves of each dataset. The first job uses 20 examples per class to smoke-test the exact algorithm under scheduler control. This scales resources, not procedure.
- Libraries: rotation uses `scipy.ndimage.rotate` with bilinear interpolation instead of torchvision because this container does not need PyTorch for the scaled reproduction. The rotation distribution and no-expand geometry are preserved.
