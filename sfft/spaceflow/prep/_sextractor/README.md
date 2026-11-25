# SExtractor 2.28.0 as betasfft's separate program

This folder holds a modified version of SExtractor, the source extractor by Emmanuel Bertin (IAP, CNRS, Sorbonne Université, CFHT),
distributed under the GNU General Public License version 3 or later (`LICENSE`, `COPYRIGHT`, `AUTHORS`, `NOTICE`). betasfft builds it into a
program of its own, `sextractor228`, and runs that program in a separate process for the SExtractor calls of its single-exposure flow: the
source masks of the two exposures and SFFT's sparse fit mask (`sfft.spaceflow.prep.fitmask`, through `sfft.spaceflow.prep.astromatic.bundled`). The rest of
betasfft is licensed under the MIT License (`LICENSE` at the top of the repository). This folder and the program built from it are shipped
with betasfft as an aggregate, not combined with it into one program: the GPL covers them, not the rest of betasfft.

## What is here

| File | What it is |
|---|---|
| `src/` | the sources of the 2.28.0 release (github.com/astromatic/sextractor, tag 2.28.0), unchanged: the 37 files of `src/` its build compiles without model fitting, all headers, and the `fits/` and `wcs/` libraries |
| `patches.py` | the edits applied to a copy of `src/` at build time, each an exact replacement that must match once (no computation of the program is changed); also the source hash the build embeds |
| `sexlib.c` | the runtime the edits route into: memory files, errors and warnings, a copy of glibc's random generator (its notices: `NOTICE`) |
| `sexprog.c` | the program's `main`: SExtractor's own command line, or `--frames` (input files from standard input, output files to standard output) |
| `build.py` | builds the program into `build/`, with pinned compiler flags and checks |
| `config.h`, `SOURCES` | the fixed configuration the sources read (no configure script runs) and the files compiled |

## Modifications

Modified on 2026-11-21 and 2026-11-22. The edits in `patches.py` route six things of the program into
`sexlib.c`; the arithmetic is the program's:

- `main()` becomes `sexlib_main()`, run by `sexlib_run()`, which `sexprog.c`'s `main` calls;
- `error()` returns to `sexlib_run()` with its message instead of ending the process;
- `warning()` collects its messages during the run;
- `rand()` in the deblender (and in `fitswcs.c`) uses a copy of glibc's generator, reseeded with 1 at every run: the stream a new process of
  the executable starts from under glibc, so the labels are the same on every platform (another stream changes them);
- the five places that open files call `sexlib_fopen()`: names that start with `sexlib_mem_` are memory streams, other names real files;
- `reendcheck()` may hand the SEGMENTATION check image to a caller's buffer (the program gives none, so the image is written as usual).

`build.py` pins the compiler flags that would change the numbers: no fused multiply-add contraction, no fast-math, and on x86-64
conda-forge's `-march=nocona`.

## Building and running

Installing betasfft with pip builds the program (`setup.py` runs `build.py`); by hand: `python build.py` (a C compiler is needed: gcc on
Linux, the command-line tools on macOS). betasfft looks for it in `build/` and refuses it when its embedded source hash differs from this
folder's, so build again after changing anything here.

    build/sextractor228 image.fits -c default.sex        SExtractor's own command line, files on disk
    build/sextractor228 --sexlib-info                    version, sources' hash, compiler flags

betasfft runs it as `sextractor228 --frames <image> -c <configuration>`: the files whose names start with `sexlib_mem_` travel through
standard input and output, in SExtractor's own formats (FITS images and catalogues, configuration and parameter text, the filter); the
comment at the top of `sexprog.c` gives the framing.

## Checks

Made on 2026-11-22 against the conda-forge executable 2.28.0, on three pairs of full-frame Roman WFI exposures: the program gave the
executable's catalogue (every column, row for row, byte for byte) and segmentation (every pixel) for both SExtractor calls of the masks,
and betasfft's single-exposure flow gave identical products with either (source masks, fit mask, matched pair, difference, score,
variance). Until 2026-11-23 a test repeated the comparison on a synthetic image and checked the program's own command line, its errors
and the source hash; it was removed on 2026-11-23, when betasfft's tests moved to the upstream repository (`test/README.rst`).
