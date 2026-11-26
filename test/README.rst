Testing sfft
============

The tests use small synthetic images only: no data files are needed, and none may be added (real data and
results made from it never go into git). Run them from the repository root, so that the package in this
working tree is the one tested::

    python -m pytest test

Before the first run, build the SExtractor program that the ``romanL2`` preparation uses (it needs a C
compiler)::

    python sfft/spaceflow/prep/_sextractor/build.py

The tests need ``drizzle>=3.0`` and ``gwcs`` besides sfft's own dependencies. Some are skipped when
something they need is missing:

- the tests that run SExtractor, when its program is not built;
- the tests of ``sfft.utils.CupyWCSTransform`` and ``sfft.utils.ResampKits``, when ``cupy`` is not
  installed (both modules import it when they load).

One test is an expected failure: sidecar's default calling sequence (``cross_convolve`` skipped, then
``create_score_image``) on ``engine='romanL2'``, which that engine does not support yet.

What the files test:

- ``test_legacy_paths.py``: the SFFT 1.6.4 release's import paths (``sfft.utils.StampGenerator``,
  ``from sfft import Customized_Packet``, ...) give the ``sfft.legacy`` modules and classes, with a
  ``DeprecationWarning``, and ``import sfft`` imports neither engine;
- ``test_spacesfft.py``: ``SpaceSFFT`` with each engine, the keywords each takes, attribute forwarding,
  and the calling sequences of phrosty and sidecar;
- ``test_prep_arrays.py``: the ``romanL2`` preparation from arrays (a FITS header or a gWCS, the masks and
  optional inputs and their warnings, the sky and unit switches, the PSF inputs), and ``SpaceL2_Prep`` and
  ``SpaceL2_Flow`` end to end.
