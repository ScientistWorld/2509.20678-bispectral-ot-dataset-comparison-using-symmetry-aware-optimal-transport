# Experiment Overview

## rotated_alignment

Tests whether an alignment method can recover semantic correspondences when source images have been randomly rotated and target images remain unrotated. This is the main experiment behind the paper's headline claim: the aligned mass should follow class identity rather than nuisance orientation.

## unrotated_baseline

Tests whether the same alignment method remains meaningful when both source and target datasets are unrotated. This condition guards against methods that improve the shifted setting by discarding too much information or producing alignments that no longer work under the ordinary same-domain comparison.

## raw_pixel_ot_l1

Records the raw-feature optimal transport reference point used in the paper. This experiment gives future scientists a fixed baseline for judging whether their improvements exceed a simple geometry-based alignment under the same data splits and scoring rule.
