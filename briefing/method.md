# Paper's Method

## Key Contribution

The paper introduces Bispectral Optimal Transport, a symmetry-aware variant of discrete optimal transport that computes the OT cost matrix after representing each image with rotation-invariant bispectral features.

## Approach

Each image is converted from a Cartesian pixel grid into a discretized polar representation. For every radial slice, the method computes a one-dimensional discrete Fourier transform over angular bins, then forms bispectrum entries from triples of Fourier coefficients. These per-radius bispectra are concatenated into a global representation that is invariant to discrete rotations of the angular coordinate.

The method then solves an entropically regularized optimal transport problem between two empirical datasets using pairwise distances between these representations as the ground cost. The resulting coupling is converted into class predictions by aggregating transported mass over target labels and choosing the label receiving the most mass for each source example.

## Main Claims

- **Primary claim:** Transport plans computed in the symmetry-aware feature space preserve semantic class labels much better than raw-feature OT when one dataset is randomly rotated and the other is unrotated.
- **Constraint claim:** The method should remain a meaningful dataset comparison procedure, so it is also evaluated in an unrotated-to-unrotated setting rather than only under the transformed condition.
- **Secondary evidence:** Visualizations and per-class matching statistics suggest that the feature space clusters examples by semantic class while reducing sensitivity to rotation.
