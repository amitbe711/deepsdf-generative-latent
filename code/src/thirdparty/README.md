# Third-party code

This folder is intentionally empty of vendored source code.

Per the assignment's Implementation grading criterion ("I will only assess the
code you write ... you must make a clear distinction between any code taken from
online sources and your own"), the following is the full accounting:

- **All algorithmic code** (DeepSDF decoder, latent codes, auto-decoder training,
  Gaussian prior, latent DDPM, SDF sampling, marching-cubes wrapper, metrics,
  grid runner, figures) is the author's own implementation, living under `src/`.
- **External dependencies used as libraries** (not vendored code): PyTorch,
  NumPy, SciPy, scikit-image (`marching_cubes`), trimesh (mesh I/O, surface
  sampling, and the `signed_distance` proximity query), Matplotlib, PyYAML.
- **Conceptual references** (algorithms re-implemented from the papers, not
  copied): DeepSDF (Park et al., 2019), DDPM (Ho et al., 2020), the cosine noise
  schedule (Nichol & Dhariwal, 2021), and the generation-metric protocol
  (Achlioptas et al., 2018).
