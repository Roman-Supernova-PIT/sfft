# DECISIONS.md - sfft design decisions

Design and implementation decisions for this repository: the decision, the options considered and why one was chosen. Each entry is
dated. **Decided** marks a decision that has been made; **Proposed** is a draft waiting for approval; **[OPEN]** is a point not
decided yet. Work items go in `TODO.md`, not here.

**Always record who made each decision.** Every **Decided** heading or entry names the person responsible for it (initials or name)
with the date, for example "Decided (LNA, 2026-11-25)". A decision recorded without a name is not complete. When a decision is
changed, the new entry names the person who changed it.

People named in this file:

- **LNA:** responsible for the decisions of D1 dated 2026-11-25.

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

### Proposed (needs approval)

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
`import sfft` stays light; a test checks that it imports neither engine. To be confirmed by LNA.

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
