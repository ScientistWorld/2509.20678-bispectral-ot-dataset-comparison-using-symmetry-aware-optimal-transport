# Problem Definition

This benchmark asks whether two finite image datasets can be aligned so that corresponding examples share semantic class labels even when one dataset has been transformed by nuisance rotations. A solution outputs a transport plan between a source set and a target set, with mass spread uniformly over each set's images. Success is measured by how much transported mass from each source image lands on target images of the same class.

## Research Question

Can a dataset comparison method produce semantic correspondences between transformed and untransformed image datasets without being misled by rotations that change pixel geometry but should not change class identity?

## Why It Matters

Dataset alignment is used in domain adaptation, transfer learning, distribution comparison, and dataset diagnostics. In those settings, a useful correspondence should reflect stable semantic content rather than accidental acquisition conditions such as camera angle, object orientation, or other nuisance transformations.

## Success Criteria

A successful solution achieves high class-preservation accuracy when matching randomly rotated images to unrotated images from the same dataset. The same solution must also match two unrotated sets well, so that robustness to rotation is not obtained by destroying ordinary within-domain matching quality. The solver is never told which of the two situations it faces, and it never sees labels.
