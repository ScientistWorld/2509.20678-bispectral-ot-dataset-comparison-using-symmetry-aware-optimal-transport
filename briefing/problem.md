# Problem Definition

This benchmark asks whether two finite labeled image datasets can be aligned so that corresponding examples share semantic class labels even when one dataset has been transformed by nuisance rotations. A solution outputs a transport or matching relation between a source set and a target set. Success is measured by how much transported mass from each source class lands on the same target class.

## Research Question

Can a dataset comparison method produce semantic correspondences between transformed and untransformed image datasets without being misled by rotations that change pixel geometry but should not change class identity?

## Why It Matters

Dataset alignment is used in domain adaptation, transfer learning, distribution comparison, and dataset diagnostics. In those settings, a useful correspondence should reflect stable semantic content rather than accidental acquisition conditions such as camera angle, object orientation, or other nuisance transformations.

## Success Criteria

A successful solution should achieve high class-preservation accuracy when matching randomly rotated images to unrotated images from the same dataset. It should also be evaluated on an unrotated-to-unrotated condition so that improved robustness to rotation is not obtained by destroying ordinary within-domain matching quality.
