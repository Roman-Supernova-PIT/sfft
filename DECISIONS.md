# DECISIONS.md - sfft design decisions

Design and implementation decisions for this repository: the decision, the options considered and why one was chosen. Each entry is
dated. **Decided** marks a decision that has been made; **Proposed** is a draft waiting for approval; **[OPEN]** is a point not
decided yet. Work items go in `TODO.md`, not here.

**Always record who made each decision.** Every **Decided** heading or entry names the person responsible for it (initials or name)
with the date, for example "Decided (LNA, 2026-11-25)". A decision recorded without a name is not complete. When a decision is
changed, the new entry names the person who changed it.

People named in this file:

- **LNA:** responsible for the decisions of D1 dated 2026-11-25.
- **Claude:** the AI assistant working with LNA; made the decisions C1-C48 of D1 while building (2026-11-25), where the
  earlier decisions did not address them.

---

## D1. One engine wrapper, array-only inputs (branch `51-an-engine-update-for-space-flow`)

**Status:** Approved (LNA, 2026-11-25), built in small steps (`TODO.md`). Still open: O8 and the sidecar part of O12, both waiting
on others.

### Context

The branch adds a new engine for Roman Space Telescope Level 2 exposures (`sfft.spaceflow`) and moves the SFFT 1.6.4 code into
`sfft.legacy`. It left nothing at the old import paths, so phrosty and sidecar stop at import:

- `phrosty/pipeline.py`, `sidecar/subtraction.py`: `from sfft.SpaceSFFTFlow import SpaceSFFT_Flow`
- `phrosty/imagesubtraction.py`: `from sfft.utils.StampGenerator import Stamp_Generator`

The new engine has three layers. `SpaceL2_Sub` (the subtraction) already takes arrays, with the constructor signature of master's
`SpaceSFFT_Flow` plus three keywords. `SpaceL2_Prep` (preparation) and `SpaceL2_Flow` (prep, subtraction and products) take two
ASDF file paths and read them with `roman_datamodels` (`sfft/spaceflow/prep/reader.py`).

### Decided (LNA, 2026-11-25)

1. **Compatibility scope.** Everything phrosty or sidecar calls directly from sfft keeps working with its master import path, names,
   arguments, methods and attributes. The `sfft/legacy/` folder stays.
2. **One wrapper, two engines.** A class `SpaceSFFT` takes `engine='legacy'` or `engine='romanL2'`; the default is `'romanL2'`. It is a
   class with methods, as master's `SpaceSFFT_Flow` is, not a factory function.
3. **The master name is the wrapper.** `sfft.SpaceSFFTFlow.SpaceSFFT_Flow` becomes the wrapper, so phrosty and sidecar, unchanged, run
   on the new engine by default.
4. **The wrapper wraps the subtraction step only.** That is the one step both engines share (`sfft.legacy.SpaceSFFTFlow.SpaceSFFT_Flow`
   and `sfft.spaceflow.subtract.SpaceL2_Sub`). Preparation (`SpaceL2_Prep`), the fit mask and the full flow (`SpaceL2_Flow`) exist only
   for `romanL2`. With `engine='legacy'`, the keywords only the new engine knows are refused.
5. **sfft never reads an ASDF file.** Every input is a numpy array, with what the file held passed as keywords. snappl reads the files
   for phrosty and sidecar. An example script and the demo notebook in `examples/` show the reading.
6. **WCS:** either a FITS header or a gWCS object, per exposure.
7. **Masks:** the caller passes each kind of mask (unusable, saturated, cosmic-ray jump) as its own array; sfft warns
   (`warnings.warn`) when one is not passed.
8. **Sky and units:** preparation removes the sky and converts to MJy/sr by default, from keywords (flux conversion, nominal pixel
   area, whether the data is a surface brightness), with a switch to turn each off when the caller has done it.
9. **PSF:** either the caller's PSF arrays or `detector` and `filter` keywords, from which sfft evaluates STPSF (both as on the branch).
10. **Optional arrays** `dumo` and `var_poisson`: accepted; a warning (`warnings.warn`) when they are not passed. Whether
    `var_poisson` is needed at all is waiting on Lei's answer (O8, `TODO.md`).
11. **Keyword style:** flat keywords in master's style, with `_target` / `_object` names (`PixA_target`, `PixA_object`, ...).
12. **Records:** decisions in this file, work items in `TODO.md`.
13. **Examples:** the demo notebook is committed as `examples/sfft_spaceflow_demo.ipynb`, reading ASDF itself (not through snappl),
    saved without cell outputs. The notebook at the repository root stays untracked.
14. **Git:** commits go to `51-an-engine-update-for-space-flow` itself (the branch owner agreed), each commit and push asked first.
15. **Wrapper file (O1).** The class `SpaceSFFT` lives in a new file, `sfft/SpaceSFFT.py`. `sfft/SpaceSFFTFlow.py` only makes the
    master name `SpaceSFFT_Flow` point to it.
16. **Old paths (O2).** Every `sfft.*` import path of master keeps working for every outside user, not only the ones phrosty and
    sidecar use (the list is in P1).
17. **Engine names (O3).** `engine` accepts `'legacy'` and `'romanL2'` in any letter case (`'ROMANL2'`, `'romanl2'`, ...).
18. **Engine attribute (O4).** The wrapper keeps its engine object in the attribute `sfftengine`.
19. **Keyword names (O6).** The keywords of P3 are adopted. Where master's `SpaceSFFT_Flow` already has a name for an input, that name
    is used (`PixA_target`, `PixA_targetVar`, `PSF_target`, ...); new inputs follow the same pattern. The WCS is the exception
    (decision 26).
20. **Noise input (O7).** The caller passes the **variance** (`PixA_targetVar`, `PixA_objectVar`), as master's `SpaceSFFT_Flow`
    takes it, not the error.
21. **Missing optional inputs (O9).** The behavior in the table "Missing inputs" of P3 is adopted as written there.
22. **Switch names (O10).** `subtract_sky=True` and `convert_units=True`.
23. **Two PSFs (O11).** Each image has its own PSF: two PSFs in total, `PSF_target` for the target and `PSF_object` for the object.
    `SpaceL2_Prep` and `SpaceL2_Flow` take them as two keywords.
24. **Results and dependency (O12, phrosty part).** That phrosty's results change on `romanL2` (finding 3) is accepted, and so is the
    new dependency `drizzle>=3.0` (finding 4). The sidecar part of O12 is waiting on Lei and MWV (`TODO.md`).
25. **Attribute forwarding (O5).** The wrapper forwards attributes both ways, as written in P2: reading an attribute it does not have
    returns the engine's, and setting one sets it on the engine. On `romanL2`, setting an attribute the engine does not read back warns.
    Reason: sidecar's spline option sets `PixA_resamp_object` and `BlankMask` (`sidecar/subtraction.py:733-734`); with read-only
    forwarding, the legacy engine would subtract with the original template while sidecar read back its own, a silently wrong result.
26. **WCS keyword names (O13).** `SpaceL2_Prep` and `SpaceL2_Flow` take the WCS as `wcs_target` and `wcs_object` (a FITS header or a
    gWCS, decision 6), not master's `hdr_target` / `hdr_object`, since the value need not be a header. The wrapper `SpaceSFFT` keeps
    master's `hdr_target` / `hdr_object` (decision 1).
27. **Old paths by one import hook (O14).** One import hook in `sfft/__init__.py` maps each `sfft.<old path>` of P1.1 to
    `sfft.legacy.<old path>`; there are no stub files at the old locations.
28. **Deprecation warning (O15).** An import through an old path of P1.1 gives a `DeprecationWarning` that names the `sfft.legacy` path
    to use. `sfft.SpaceSFFTFlow` does not warn, since it holds the wrapper.
29. **No shared PSF stamp (O16).** `psf=dict(source='array', stamp=...)`, one stamp for both exposures, is removed. A caller's PSFs come
    only as `PSF_target` and `PSF_object` (decision 23). `psf` stays for the STPSF settings.
30. **Top-level names warn too (O17).** Master's top-level names of P1.2 (`from sfft import Customized_Packet`, ...) also give a
    `DeprecationWarning` naming the `sfft.legacy` name to use.
31. **Approval.** LNA approved D1 on 2026-11-25, to be built in small steps, each confirmed before the next.
32. **Code style.** New and changed code follows sfft's own local style (as in `sfft/spaceflow` and `sfft/legacy`), not the SNPIT
    workspace's `CODE_STYLE.md`; no repository-wide `ruff` or `mypy` configuration is added.
33. **Test environment.** Tests run in the `phot` conda environment against this working tree (`python -m pytest` from the repository
    root), with `drizzle>=3.0` added to `phot`. The sfft installed in `phot` (1.6.4.dev16, used by phrosty there) is left as it is.
34. **`SpaceSFFT` imported up front.** `sfft/__init__.py` imports `SpaceSFFT` itself, not when first used (P1). Measured in `phot`:
    loading `sfft/SpaceSFFT.py` takes about 0.14 ms and 6 KiB, and it avoids `from sfft import SpaceSFFT` returning the module of the
    same name instead of the class.
35. **Examples stay current.** Every change to the package's API updates `examples/` in the same step, so that the examples always run
    against the package as it is.
36. **Shared outlier rejection needs detector names.** A pixel rejected as an outlier in one exposure is rejected in the other only
    when `detector_target` and `detector_object` are both given and equal; without them, nothing is shared.
37. **Positional inputs.** `SpaceL2_Prep` and `SpaceL2_Flow` take the six required inputs positionally, in master's order
    (`PixA_target, PixA_object, PixA_targetVar, PixA_objectVar, wcs_target, wcs_object`); every other input is keyword-only.
38. **Input provenance.** `record.json` records each input array's shape and SHA-256 (about 0.1 s per 4088 x 4088 array).
39. **Caller PSFs and the WCS position type (for now).** When a PSF is passed (`PSF_target` / `PSF_object`) instead of being evaluated
    from STPSF, sfft warns (`warnings.warn`) that it does not correct that PSF for where the WCS places a star, so the caller must do
    that correction outside sfft. sfft's correction (`wcs_position_type` other than `'chief-ray'`) is applied only to STPSF PSFs. Reason:
    the offset between a PSF model's centre and where a WCS fitted to star centroids places a star depends on how the PSF was made,
    which only its maker knows.
40. **4-D caller PSFs with `crop_px` (option a).** Accepted. Their stamps must be given at the nodes of the target grid as cropped; sfft
    checks the number of nodes, and the docstring says where the nodes are.
41. **Examples built in step 2.** `examples/` is built in step 2, together with the array inputs it depends on (decision 35).
42. **Real data and its results never go into git.** Roman Level 2 files used to check the code (such as the two WFI11 F106 `_cal.asdf`
    files of 2026-11-25) and every product, log or cache made from them are never committed to git or pushed to GitHub. They are kept
    outside the repository (products in the workspace's `scratch/`, STPSF's cache in `~/.cache/sfft_romanl2`), and tests use synthetic
    data only. As a safeguard, `.gitignore` ignores `*.asdf` and `spaceflow_output/` (the example's default output folder).
43. **The SExtractor build stays local.** The program built by `sfft/spaceflow/prep/_sextractor/build.py` into its `build/` folder is
    git-ignored (each installation builds its own, as `setup.py` says).

### Decided by Claude while building (2026-11-25)

Claude (the AI assistant working with LNA) made these decisions while building steps 1 and 2, because the decisions above did not
address them. LNA was shown the full list on 2026-11-25. Any of them can be changed by a new decision that names who made it.

**Step 1: the wrapper and the old paths**

- **C1. romanL2-only keywords default to `None`.** `PSF_OVERSAMPLING`, `PSF_PIXEL_INTEGRATED` and `instrument` default to `None` in
  `SpaceSFFT`, meaning "romanL2's own default". With `engine='legacy'`, any value other than `None` raises `TypeError`.
- **C2. Engine names ignore surrounding spaces** (`" LEGACY "` is accepted); any other name raises `ValueError` listing the two.
- **C3. `engine` and `sfftengine` cannot be reassigned** on a built `SpaceSFFT` (`AttributeError`).
- **C4. `fit_mask`.** `sfft_subtract(fit_mask=...)` with `engine='legacy'` raises `TypeError`; without a fit mask, the engine's
  `sfft_subtract()` is called with no argument, as phrosty and sidecar call it.
- **C5. The romanL2 set-attribute warning is a `UserWarning`**, limited to `PixA_resamp_object` and `BlankMask`: reading
  `SpaceL2Sub.py` shows romanL2 writes them during resampling and never reads them back.
- **C6. Wrapper methods return what the engine's method returns**, and the wrapper has a `__repr__`.
- **C7. `SpaceSFFT_Flow` is the same class object as `SpaceSFFT`** (an alias), not a subclass.
- **C8. The import hook comes first in `sys.meta_path`, once**, so that an old module inside an old package is not loaded a second time
  from its file.
- **C9. The old paths are listed explicitly** (42 paths, packages included). `sfft.utils.meta.TimeoutKit` gets the branch's rewritten
  `sfft.legacy` version.
- **C10. A top-level name warns once per process:** after its first lookup it is kept on the package (`from sfft import` looks a name
  up twice).
- **C11. `sfft/__init__.py` has `__all__ = ["SpaceSFFT"]`**, and its docstring describes the two engines and the old paths.
- **C12. Two warnings for phrosty's one import line are accepted:** `from sfft.utils.StampGenerator import ...` warns for
  `sfft.utils` and for `sfft.utils.StampGenerator`, both old paths.
- **C13. Step 1 tests use 256 x 256 synthetic images** on the CPU, with `GKerHW=4` to keep them quick.
- **C14. Tests of the two legacy modules that import `cupy`** (`utils.CupyWCSTransform`, `utils.ResampKits`) are skipped when `cupy`
  is not installed; sidecar's default sequence on romanL2 is a strict expected failure (O12), so it is reported once it passes.

**Step 2: array inputs and examples**

- **C15. Each exposure's PSF is chosen on its own:** one may be the caller's and the other STPSF's. An exposure without `PSF_<side>`
  needs `detector_<side>` and `filter_<side>`, or `ValueError`.
- **C16. A caller's PSF is copied to float64 and must be 2-D or 4-D;** a 4-D one must have `nodes x nodes` stamps (decision 40). The
  odd-side check stays in `SpaceL2_Sub`.
- **C17. The caller-PSF warning (decision 39) is given for every caller PSF**, with `'chief-ray'` too; with another `wcs_position_type`
  it adds that the setting is applied only to STPSF's PSFs. `wcs_position_type` is checked even when no exposure uses STPSF.
- **C18. STPSF is set up only when an exposure needs it.** `psf['source']` must be `'stpsf'`; the error points to
  `PSF_target` / `PSF_object`.
- **C19. The filter check applies only when both filters are given.**
- **C20. Default exposure names** are `target` and `object`.
- **C21. The missing-input warnings follow the table literally:** `DUMO` and `JUMPMASK` warn when missing even with `add_dumo=False`.
  The flux-conversion warning is given only with `convert_units=True`, since the conversion is not used otherwise.
- **C22. One warning per missing input per exposure,** starting with the exposure's name and pointing at the code that built the
  preparation.
- **C23. A FITS header must give `NAXIS1` and `NAXIS2`** equal to the image's shape; without them, `ValueError`.
- **C24. A passed number must be valid:** `conversion_megajanskys_<side>` and `pixel_area_sr_<side>`, when given, must be positive and
  finite, or `ValueError`.
- **C25. The error is the square root of the variance** where the code needs it (outlier pixels, the noise dumo adds); a negative
  variance makes that pixel unusable.
- **C26. `var_poisson` is checked for shape, then not used** (O8).
- **C27. `subtract_sky=False`** records a sky level of 0, but still measures the source mask (the detection mask) and the sky noise.
- **C28. `convert_units=False` applies no scaling at all**, the pixel-area correction included; a conversion passed anyway is ignored,
  and the unit is "input unit".
- **C29. Unconverted images are labelled "input unit"** (`reader.INPUT_UNIT`), not "file unit", also in `record.json`.
- **C30. `Level2` keeps a small `meta` dict** (`filename`, `detector`, `optical_element`), so `resample.py` and `psfsource.py` need no
  change; its attribute `native_wcs` (the WCS as passed) replaces `gwcs`.
- **C31. The WCS offset vector (STPSF only) is measured after the exposures are prepared;** STPSF's grids are still set up first, so an
  STPSF problem shows before the preparation.
- **C32. Kept from the branch:** the reference is prepared before the science exposure, and the bad-pixel mask applies after dumo is
  added.
- **C33. Removed from the package:** `KEEP_FLAGS` and the DQ-flag decoding (moved to the example), `read_meta`, `Scan`, and the
  options `keep_flags` and `pixel_area`.
- **C34. The TAN-SIP accuracy warning says "the exposures' gWCS"**, not "the files' WCS".
- **C35. `SpaceL2_Flow` checks its keyword options** against `SpaceL2_Prep`'s keyword-only inputs.
- **C36. `record.json` hashes each array in its own type** (the SHA-256 of its type, shape and bytes), not converted to float64 as
  `psfsource.array_digest` does, to avoid a full-size copy of each mask. A FITS header is recorded by the SHA-256 of its text, a gWCS
  by its kind only.
- **C37. `record.json` records each exposure's PSF source** (`caller` or `stpsf`).
- **C38. `SCIFILE` / `REFFILE`** in the products' headers hold `name_target` / `name_object`.
- **C39. The example's flux conversion is 1.0 when `cal_step.flux` is `COMPLETE`** (the data is in MJy/sr), else the file's
  `photometry.conversion_megajanskys`.
- **C40. The example takes a file as a surface brightness when it names a flat** (the branch's former `'auto'` rule), and keeps the
  branch's `KEEP_FLAGS`.
- **C41. The example leaves out `dumo` / `var_poisson` when the file has none**, so their warnings show; an empty detector or filter
  becomes `None`.
- **C42. The example leaves the datamodel open**, as the branch's reader did.
- **C43. The example's command-line options** are `--output-dir` (default `spaceflow_output`), `--cache-dir`, `--backend` and
  `--crop-px`.
- **C44. The notebook imports the example script,** so it is run from `examples/`; its intro says so.
- **C45. The notebook is saved without outputs, execution counts or run timestamps** (the cells' `execution` metadata); its
  `parameters` tag stays.
- **C46. Lei's docstrings describe array inputs:** `spaceflow/__init__.py`, `prep/__init__.py`, `instruments.py`, `SpaceL2Sub.py` and
  `SpaceL2Flow.py` no longer describe reading files.
- **C47. Lines made over 158 characters are wrapped;** the two existing `F401` re-export warnings in Lei's `__init__` files are left.
- **C48. Step 2's `Level2` unit tests skip SExtractor and use a constant sky;** a synthetic gWCS checks the FITS-header path against
  the gWCS path.

### Design (approved by LNA, decision 31)

#### P1. Package layout

```text
sfft/
├── __init__.py            # __version__; SpaceSFFT; master's top-level names, imported only when first used (P1.2)
├── SpaceSFFT.py           # the wrapper class SpaceSFFT (decision 15)
├── SpaceSFFTFlow.py       # master path: SpaceSFFT_Flow = SpaceSFFT (decision 3)
├── legacy/                # unchanged
└── spaceflow/             # the romanL2 engine; prep/ and SpaceL2Flow.py take arrays (P3)
examples/
├── read_roman_l2.py       # reads a _cal.asdf with roman_datamodels, builds the inputs, runs SpaceL2_Flow
└── sfft_spaceflow_demo.ipynb
```

**Built in step 1: `SpaceSFFT` is imported by `sfft/__init__.py` itself, not when first used.** The class and its module share the
name `SpaceSFFT`. Imported on first use, `from sfft import SpaceSFFT` would return the module instead of the class whenever
`import sfft.SpaceSFFT` had run first. `sfft/SpaceSFFT.py` imports only `warnings` (each engine is imported when it is built), so
`import sfft` stays light; a test checks that it imports neither engine. Confirmed (LNA, 2026-11-25, decision 34).

##### P1.1. The old paths (decision 16)

Each old module path is the same module as its `sfft.legacy` twin, not a copy: `sfft.utils.StampGenerator is
sfft.legacy.utils.StampGenerator`. A class or a module setting therefore behaves the same through either path. One import hook in
`sfft/__init__.py` provides them (decision 27), and an import through one of them gives a `DeprecationWarning` (decision 28).

- `sfft.AutoCrowdedPrep`, `AutoSparsePrep`, `BSplineSFFT`, `CustomizedPacket`, `EasyCrowdedPacket`, `EasySparsePacket`,
  `MultiEasyCrowdedPacket`, `MultiEasySparsePacket`, `PureCupyCustomizedPacket`
- `sfft.sfftcore` with `SFFTConfigure` and `SFFTSubtract`
- `sfft.utils`, the names of its `__init__`, and its modules: `CombineHeader`, `ConvKernelConvertion`, `CupyWCSTransform`,
  `DeCorrelationCalculator`, `DeCorrelationCalculatorDep`, `HoughDetection`, `HoughMorphClassifier`, `ImageZoomRotate`,
  `NeighboringPixelCovariance`, `NumpyFFTKits`, `NumpyResampKits`, `PatternRotationCalculator`, `PureCupyDeCorrelationCalculator`,
  `PureCupyFFTKits`, `ReadWCS`, `ResampKits`, `SExSkySubtract`, `SFFTSolutionReader`, `SkyLevelEstimator`, `StampGenerator`,
  `SymmetricMatch`, `WeightedQuantile`
- `sfft.utils.meta` with `MultiProc` and `TimeoutKit` (the branch rewrote `TimeoutKit` in `sfft.legacy`; the old path gets that one)
- `sfft.utils.pyAstroMatic` with `AMConfigMaker`, `PYSEx`, `PYSWarp`
- `sfft.SpaceSFFTFlow` is the exception: its `SpaceSFFT_Flow` is the wrapper (decision 3), not the legacy class.

##### P1.2. Master's top-level names

`from sfft import Customized_Packet` (and `PureCupy_Customized_Packet`, `Auto_SparsePrep`, `Auto_CrowdedPrep`, `Easy_SparsePacket`,
`Easy_CrowdedPacket`, `MultiEasy_SparsePacket`, `MultiEasy_CrowdedPacket`, `BSpline_Packet`, `BSpline_MatchingKernel`,
`BSpline_DeCorrelation`, `BSpline_GridConvolve`) works as on master. Each name is imported from `sfft.legacy` the first time it is
used (a module-level `__getattr__`), so a bare `import sfft` stays light, as the branch's `sfft/__init__.py` promises. Each gives a
`DeprecationWarning` naming the `sfft.legacy` name to use (decision 30).

#### P2. The wrapper `SpaceSFFT`

- **Constructor:** master's `SpaceSFFT_Flow` arguments in master's positional order (sidecar passes the first twelve positionally),
  then `engine='romanL2'`, then the keywords only `romanL2` takes (`PSF_OVERSAMPLING`, `PSF_PIXEL_INTEGRATED`, `instrument`).
  `engine` is matched without regard to case (decision 17); any other value raises `ValueError` naming the two engines.
- **It builds one engine object** and keeps it in `self.sfftengine` (decision 18): `sfft.legacy.SpaceSFFTFlow.SpaceSFFT_Flow` or
  `sfft.spaceflow.subtract.SpaceL2_Sub`, imported only when chosen.
- **Methods** (explicit, each calls the engine's method of the same name): `resample_image_mask_psf()`, `cross_convolve()`,
  `sfft_subtract(fit_mask=None)`, `find_decorrelation()`, `apply_decorrelation(img)`, `create_score_image()`,
  `create_variance_image()`. `fit_mask` is refused with `engine='legacy'`.
- **Attributes** (decision 25): reading an attribute the wrapper does not have (`PixA_DIFF`, `op`, `hdr_target`, ...) returns the
  engine's, and setting one sets it on the engine. On `romanL2`, setting an attribute the engine does not read back
  (`PixA_resamp_object`, `BlankMask`) also gives a `warnings.warn` that the value is not used. The wrapper's own attributes,
  `engine` and `sfftengine`, stay on the wrapper.
- `engine` holds the engine's name as given in decision 2 (`'legacy'` or `'romanL2'`).

#### P3. Array inputs of `SpaceL2_Prep` and `SpaceL2_Flow`

`science_file` and `reference_file` are replaced by these keywords (decisions 19 and 26), each given for the target (science) and the object
(reference). The table shows the target's name; the object's is the same with `object` in place of `target`.

| Keyword | Type | Required | Replaces in the file |
| :--- | :--- | :--- | :--- |
| `PixA_target` | 2-D float | yes | `data` |
| `PixA_targetVar` | 2-D float, the variance | yes | `err` squared (decision 20) |
| `wcs_target` | FITS header (TAN-SIP, with `NAXIS1/2`) or gWCS | yes | `meta.wcs` |
| `PixA_target_BADMASK` | 2-D bool | no, warns | `dq` outside `KEEP_FLAGS` |
| `PixA_target_SATMASK` | 2-D bool | no, warns | `dq & SATURATED` |
| `PixA_target_JUMPMASK` | 2-D bool | no, warns | `dq & JUMP_DET` |
| `PixA_target_DUMO` | 2-D float | no, warns | `dumo` |
| `PixA_target_VARPOISSON` | 2-D float | no, warns | `var_poisson` (O8) |
| `PSF_target` | 2-D, or 4-D at the PSF nodes | yes, unless `detector_target` and `filter_target` are given | the STPSF model |
| `detector_target`, `filter_target` | str | when `PSF_target` is not given | `meta.instrument.detector`, `.optical_element` |
| `conversion_megajanskys_target` | float | no | `meta.photometry.conversion_megajanskys` |
| `pixel_area_sr_target` | float | no | `meta.photometry.pixel_area` |
| `is_surface_brightness_target` | bool | no | the "a flat was applied" test on `meta.ref_file.flat` |
| `name_target` | str | no | `meta.filename` (used in messages) |

`PSF_OVERSAMPLING` and `PSF_PIXEL_INTEGRATED` (one value or a pair `(target, object)`, as in `SpaceL2_Sub`) say how `PSF_target` and
`PSF_object` are sampled. The two exposures' filters must match: given as `filter_target` and `filter_object`, a mismatch raises
`ValueError`, as the branch does for two files.

##### Missing inputs (decision 21)

When an optional input is not passed, this is what sfft does. Each warning is one `warnings.warn` per exposure, naming the exposure
(`name_target` / `name_object`, or "target" / "object").

| Input not passed | Warns? | What sfft does instead |
| :--- | :--- | :--- |
| `PixA_target_BADMASK` | yes | Only pixels whose value or variance is not finite (NaN, inf) are unusable. |
| `PixA_target_SATMASK` | yes | No saturation guard: the fit mask does not keep saturated stars out (`saturation_map` is `None`). |
| `PixA_target_JUMPMASK`, with `PixA_target_DUMO` given | yes | `dumo` is added at every pixel where it is finite, cosmic-ray pixels included. |
| `PixA_target_JUMPMASK`, without `PixA_target_DUMO` | yes | Nothing else changes: the jump mask is only used to place `dumo`. |
| `PixA_target_DUMO` | yes | Nothing is added to the data, and no `dumo` noise is added to the variance. |
| `PixA_target_VARPOISSON` | yes | Nothing (the branch does not use it; O8). |
| `conversion_megajanskys_target` | yes | The images stay in the input unit, and the products are labeled "input unit", not MJy/sr. |
| `pixel_area_sr_target` | no | The median of the pixel-area map computed from the WCS. |
| `is_surface_brightness_target` | no | `True`: the data is taken to be a surface brightness already (a flat-fielded Level 2). |
| `name_target` | no | The exposure is called "target" (or "object") in messages. |

A required input that is missing (`PixA_target`, `PixA_targetVar`, `wcs_target`, and a PSF: `PSF_target` or both `detector_target` and
`filter_target`) raises `ValueError` naming it; nothing is computed.

##### Other details

- **Switches (decision 22):** `subtract_sky=True` and `convert_units=True`; `False` skips that step for both exposures, for a caller
  that has done it already.
- **Removed:** `keep_flags` (the caller builds `PixA_target_BADMASK`) and the `'auto'` value of `pixel_area`.
- **Unchanged:** the other options of `SpaceL2_Prep` (`crop_px`, `crop_centre`, `sky_model`, `mesh_px`, `add_dumo`,
  `reject_outliers`, `wcs_position_type`, `nodes`, `cache_dir`, `instrument`), and `psf` for the STPSF settings only: its
  `source='array'` form is removed (decision 29).
- **WCS:** a gWCS is turned into a TAN-SIP header of degree 5 (`to_fits_sip`, as now), and the pixel areas come from the gWCS. A FITS
  header is used as given, with the pixel areas from its astropy WCS (`geometry.pixel_area_map` already takes either).
- **`SpaceL2_Flow`** takes the same keywords and passes them on. Its `record.json` records each input array's shape and SHA-256
  (`psfsource.array_digest`) in place of the file's path and hash.
- **`reader.py`:** `Level2` takes the arrays instead of a path. `Scan`, `_load` and `read_meta` and every `roman_datamodels` import
  leave sfft. `roman_datamodels` stays in the versions that `record.json` lists only if it is installed.

##### Built in step 2

The details settled while building are recorded under "Decided by Claude while building" (C15-C48).

#### P4. Examples

- **`examples/read_roman_l2.py`:** opens two `_cal.asdf` files with `roman_datamodels`, builds every keyword of P3 (masks from `dq`
  with the branch's `KEEP_FLAGS`), and runs `SpaceL2_Flow`.
- **`examples/sfft_spaceflow_demo.ipynb`:** the root notebook with its reading cell replaced by the example's functions, saved without
  outputs.

#### P5. Tests

`test/` gains pytest tests on small synthetic arrays (no files, CPU backend):

- every master import path of P1.1 and P1.2 imports, and each old module path is the same object as its `sfft.legacy` twin;
- an import through an old path of P1.1 gives a `DeprecationWarning` naming the new path, and `sfft.SpaceSFFTFlow` gives none;
- setting `PixA_resamp_object` through the wrapper reaches the legacy engine, and warns on `romanL2`;
- `SpaceSFFT` with each engine runs the method sequences phrosty and sidecar run, and exposes the attributes they read;
- `engine` is accepted in any letter case, and an unknown name raises;
- `engine='legacy'` refuses the new-only keywords;
- each row of "Missing inputs" warns (or not) and behaves as written;
- a FITS header and a gWCS give the same prepared inputs.

### Findings on decision 3 (phrosty and sidecar, unchanged, on `romanL2`)

Read from the code, not run.

1. **sidecar's default path stops.** sidecar's defaults are `cross_convolve=False` and `resample_method='bilinear'`, and it always
   calls `create_score_image()`. `SpaceL2_Sub.create_score_image()` raises `RuntimeError` without `cross_convolve()` first.
2. **sidecar's spline resampling breaks.** `sidecar/subtraction.py:719` calls `sfftifier.op.Resampling(...)`, which the new engine's
   `op` does not have. It then replaces `PixA_resamp_object` and `BlankMask`, which the new engine does not read back: it works from
   its own internal stacks, so the replacement would be ignored silently.
3. **phrosty's results change** without an error (accepted, decision 24):
   - phrosty marks bad pixels with interpolated values, not NaN;
   - the new engine ignores `target_skyrms` and `object_skyrms`;
   - decorrelation becomes an identity filter (`FKDECO`);
   - `PSF_OVERSAMPLING` defaults to 1, and the sampling of the PSFs snappl gives has not been checked.
4. **`drizzle>=3.0` becomes a runtime need** of both packages through the default engine (accepted, decision 24).

### [OPEN] points

- **O8.** `var_poisson` is read from the file on the branch but used nowhere. Accept it anyway (decision 10), or drop it? LNA has
  asked Lei (2026-11-25); tracked in `TODO.md`.
- **O12, sidecar part.** Findings 1 and 2: change sidecar, make `romanL2` handle sidecar's calling pattern, or reconsider the default?
  LNA has asked Lei and MWV (2026-11-25); tracked in `TODO.md`.
