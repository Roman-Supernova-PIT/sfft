# TODO.md - sfft work items

Open work for this repository. Decisions and their reasons are in `DECISIONS.md`.

## Branch `51-an-engine-update-for-space-flow`

- [x] Approve the draft D1 in `DECISIONS.md` (LNA, 2026-11-25).
- [x] Step 1: the wrapper `SpaceSFFT`, `sfft.SpaceSFFTFlow`, the old paths and top-level names with their warnings, and their tests
      (built, not committed yet).
- [ ] Step 2: `SpaceL2_Prep` and `SpaceL2_Flow` take arrays (D1 P3).
- [ ] Step 3: the examples (D1 P4).
- [ ] Step 4: `setup.py` and `README.rst`.
- [ ] `var_poisson` (D1 O8): the branch's reader reads it from the file (`sfft/spaceflow/prep/reader.py`, `Level2._load`) but nothing
      uses it. LNA asked Lei on 2026-11-25 whether the engine should use it or drop it; waiting for the answer. Until then the design
      accepts it as an optional keyword with a warning when missing (D1 decision 10).
- [ ] Wrapper `SpaceSFFT` with `engine='legacy' | 'romanL2'` (D1 P2).
- [ ] Master import paths: `sfft.SpaceSFFTFlow.SpaceSFFT_Flow` (the wrapper) and `sfft.utils.StampGenerator.Stamp_Generator` (D1 P1).
- [ ] `SpaceL2_Prep` and `SpaceL2_Flow` take arrays and keywords, not ASDF paths; remove `roman_datamodels` from the package (D1 P3).
- [ ] `examples/read_roman_l2.py` and `examples/sfft_spaceflow_demo.ipynb` without outputs (D1 P4).
- [ ] Tests on synthetic arrays (D1 P5).
- [ ] Dependencies in `setup.py`: `stpsf` (needed for the `'stpsf'` PSF source) is not declared; `roman_datamodels` and `asdf` are
      needed only by the examples.
- [ ] Update `README.rst` for the wrapper and the array inputs.
- [ ] Ask Lei about `sfft/legacy/utils/pyAstroMatic/__init__ copy.py`: it looks like a stray copy committed on the branch.

## phrosty and sidecar (other repositories; each change confirmed with the maintainer first)

- [ ] sidecar on `romanL2` (D1 findings 1 and 2, O12). LNA asked Lei and MWV on 2026-11-25 what they would like to do; waiting for
      the answer.
      - sidecar's default path (`cross_convolve=False` in `subtraction.py:599`, `pipeline.py:121` and `--cross-convolve`) never calls
        `cross_convolve()`, but always calls `create_score_image()`, which raises `RuntimeError` on `romanL2`.
      - The spline option (`resample_method='spline3'/'spline5'`, `_spline_resample_template` at `subtraction.py:719`) calls
        `op.Resampling`, which `romanL2` lacks, and overwrites `PixA_resamp_object` / `BlankMask`, which `romanL2` ignores.
      - Options: change sidecar, make `romanL2` support sidecar's calling pattern, or reconsider the default engine.
- [ ] phrosty on `romanL2`: bad pixels as NaN, PSF sampling, the decorrelation it expects (D1 finding 3).
- [ ] Both: add `drizzle>=3.0`, raise the minimum `sfft-romansnpit` version, and expose the engine choice in their configs.
