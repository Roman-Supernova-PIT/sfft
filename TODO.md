# TODO.md - sfft work items

Open work for this repository. Decisions and their reasons are in `DECISIONS.md`.

## Branch `51-an-engine-update-for-space-flow`

Done (design D1, approved by LNA on 2026-11-25; all pushed):

- [x] Step 1: the wrapper `SpaceSFFT` (`engine='legacy' | 'romanL2'`), `sfft.SpaceSFFTFlow.SpaceSFFT_Flow` as the wrapper, the
      release's old import paths and top-level names with their `DeprecationWarning`, and their tests (commit `a478761`).
- [x] Step 2: `SpaceL2_Prep` and `SpaceL2_Flow` take arrays and keywords instead of ASDF files, `roman_datamodels` removed from the
      package, `examples/read_roman_l2.py` and `examples/sfft_spaceflow_demo.ipynb` (without outputs), and tests on synthetic arrays,
      including the ones that run SExtractor; the example checked on two real WFI11 F106 files (commit `5998474`).
- [x] Step 3: merged into step 2 (decision 41).
- [x] Step 4: `setup.py` (required `numpy` and `packaging`, the `stpsf` and `examples` extras, the README as reStructuredText, version
      `1.6.4.dev21`), `README.rst` and `test/README.rst` (commit `8bab091`).

Open:

- [ ] `var_poisson` (D1 O8): accepted as `PixA_<side>_VARPOISSON`, with a warning when missing, but not used. LNA asked Lei on
      2026-11-25 whether the engine should use it or drop it; waiting for the answer.
- [ ] Ask Lei about `sfft/legacy/utils/pyAstroMatic/__init__ copy.py`: it looks like a stray copy committed on the branch.
- [ ] Ask Lei which contact address sfft should give: `README.rst` gives `astroleihu@gmail.com`, `setup.py` `leihu@sas.upenn.edu`.
- [ ] Optional, for Lei: the real-data run shows warnings from older code: unclosed files (`ResourceWarning`) in
      `sfft/spaceflow/prep/astromatic/bundled.py`, and "the 'warn' method is deprecated" messages.

## phrosty and sidecar (other repositories; each change confirmed with the maintainer first)

- [ ] sidecar on `romanL2` (D1 findings 1 and 2, O12). LNA asked Lei and MWV on 2026-11-25 what they would like to do; waiting for
      the answer.
      - sidecar's default path (`cross_convolve=False` in `subtraction.py:599`, `pipeline.py:121` and `--cross-convolve`) never calls
        `cross_convolve()`, but always calls `create_score_image()`, which raises `RuntimeError` on `romanL2`.
      - The spline option (`resample_method='spline3'/'spline5'`, `_spline_resample_template` at `subtraction.py:719`) calls
        `op.Resampling`, which `romanL2` lacks, and overwrites `PixA_resamp_object` / `BlankMask`, which `romanL2` ignores.
      - Options: change sidecar, make `romanL2` support sidecar's calling pattern, or reconsider the default engine.
- [ ] phrosty on `romanL2`: bad pixels as NaN, PSF sampling, the decorrelation it expects (D1 finding 3).
- [ ] Both: raise the minimum `sfft-romansnpit` version to `1.6.4.dev21` (which brings `drizzle>=3.0`), expose the engine choice in
      their configs, and move their imports from the old paths (`sfft.utils.StampGenerator`) to `sfft.legacy`.
