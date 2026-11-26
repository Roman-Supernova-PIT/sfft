Roman SNPIT fork of SFFT
------------------------

Please look at the PyPi package ``sfft`` for the original.  This
package has customizations created for the Roman SNPIT. It is
listed on PyPi as ``sfft-romansnpit``.

.. image:: https://raw.githubusercontent.com/thomasvrussell/sfft/a24fd31258c816337e68c16ff4a10295794b8386/docs/sfft_logo_gwbkg.png
   :alt: SFFT

*SFFT: Saccadic Fast Fourier Transform for image subtraction*
-------------------------------------------------------------
.. image:: https://img.shields.io/pypi/v/sfft.svg
    :target: https://pypi.python.org/pypi/sfft
    :alt: Latest Version

.. image:: https://static.pepy.tech/personalized-badge/sfft?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
    :target: https://pepy.tech/project/sfft

.. image:: https://img.shields.io/badge/python-3.7-green.svg
    :target: https://www.python.org/downloads/release/python-370/

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.6463000.svg
    :target: https://doi.org/10.5281/zenodo.6463000
    :alt: 1.0.6

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

|

Saccadic Fast Fourier Transform (SFFT) is an algorithm for fast & accurate image subtraction in Fourier space. 
SFFT brings about a remarkable improvement of computational performance of around an order of magnitude compared to other published image subtraction codes. 

SFFT method is the transient detection engine for several ongoing time-domain programs, including the `DESIRT <https://ui.adsabs.harvard.edu/abs/2022TNSAN.107....1P/abstract>`_ survey based on DECam & DESI, the DECam GW-MMADS Survey for GW Follow-ups and the JWST Cycle 3 Archival program `AR 5965 <https://www.stsci.edu/jwst/science-execution/program-information?id=5965>`_. SFFT is also the core engine for the differential photometry pipeline of the `Roman Supernova PIT <https://github.com/Roman-Supernova-PIT>`_.

Roman SN PIT: engines and inputs
--------------------------------

**This is the Roman SN PIT fork of SFFT.** The original SFFT is at https://github.com/thomasvrussell/sfft, and Lei Hu
is the primary contact person for SFFT.

This package subtracts images with either of two engines, chosen with the ``engine`` keyword of
``SpaceSFFT``:

- ``engine='romanL2'`` (the default): the engine for Roman Space Telescope Level 2 exposures
  (``sfft.spaceflow``);
- ``engine='legacy'``: the subtraction of the SFFT 1.6.4 release (``sfft.legacy``).

Both take the same arguments and run the same steps::

    from sfft import SpaceSFFT

    sub = SpaceSFFT(hdr_target, hdr_object, target_skyrms, object_skyrms,
                    PixA_target, PixA_object, PixA_targetVar, PixA_objectVar,
                    PixA_target_DMASK, PixA_object_DMASK, PSF_target, PSF_object,
                    BACKEND_4SUBTRACT="Cupy", engine="romanL2")
    sub.resample_image_mask_psf()
    sub.cross_convolve()
    sub.sfft_subtract()
    sub.find_decorrelation()
    decorr_diff = sub.apply_decorrelation(sub.PixA_DIFF)

With ``engine='romanL2'``, ``create_score_image()`` needs ``cross_convolve()`` first.

``sfft.SpaceSFFTFlow.SpaceSFFT_Flow`` is the same class under the release's name. The release's other
import paths (``sfft.utils.StampGenerator``, ``from sfft import Customized_Packet``, ...) still work: they
give the ``sfft.legacy`` modules and classes, with a ``DeprecationWarning`` naming the path to use instead.

**Inputs are arrays, never files.** sfft reads no file. ``sfft.spaceflow.SpaceL2_Flow`` and
``SpaceL2_Prep`` take each exposure's data, variance and WCS (a gWCS or a FITS header), and, as keywords,
what its Level 2 file holds: the bad-pixel, saturation and cosmic-ray masks, ``dumo``, the detector,
filter and flux calibration, and a PSF of your own. sfft warns for each optional input left out.
``examples/read_roman_l2.py`` shows how to read two ``_cal.asdf`` files into these inputs, and
``examples/sfft_spaceflow_demo.ipynb`` runs it.

Installation options::

    pip install sfft-romansnpit                 # both engines, with PSFs of your own
    pip install 'sfft-romansnpit[stpsf]'        # also STPSF's Roman WFI PSFs, evaluated by sfft
    pip install 'sfft-romansnpit[examples]'     # also roman_datamodels, to run examples/

The ``romanL2`` preparation runs a bundled copy of SExtractor as a separate program, built when the
package is installed. Without a C compiler the installation goes on without it; in a clone of this
repository, build it later with::

    python sfft/spaceflow/prep/_sextractor/build.py

A GPU (``cupy``) is optional for both engines: ``BACKEND_4SUBTRACT="Numpy"`` (or ``backend="cpu"`` for
``SpaceL2_Flow``) runs on the CPU.

Get started
---------------------

**NOTE:** The below links apply to the original SFFT repo, located at https://github.com/thomasvrussell/sfft. 

- **Documentation:** https://thomasvrussell.github.io/sfft-doc/ **[recommended]**
- **Installation:** https://thomasvrussell.github.io/sfft-doc/installation/
- **Tutorials:** https://thomasvrussell.github.io/sfft-doc/tutorials/
- **Contact the author:** astroleihu@gmail.com


Citing
--------

*Image Subtraction in Fourier Space, Lei Hu et al. 2022, The Astrophysical Journal, 936, 157* 

See ADS Link: https://ui.adsabs.harvard.edu/abs/2022ApJ...936..157H/abstract

Publications using SFFT method
--------------------------------

See ADS Library: https://ui.adsabs.harvard.edu/public-libraries/lc4tiTR_T--92f9k0YrRQg

Roman SNPIT Variant Notes
-------------------------

(These notes are for members of the Roman SNPIT who need to upload a new
version of this package.)

* Make sure that ``NAME`` in ``setup.py`` is ``sfft-romansnpit``.

* Make sure that VERSION in ``setup.py`` is of the form
  ``<MAJOR>.<MINOR>.<PATCH>.dev<n>`` where ``<MAJOR>``,
  ``<MINOR>``, AND ``<PATCH>`` match the upstream that has most recently been
  merged in, and ``<n>`` is an integer that can start at 0 for a given
  ``MAJOR.MINOR.PATCH`` and that needs to increment each time we want to
  push a new package.

* Build in the main directory with::

    python -m build --sdist --outdir dist

  That will create a file in ``dist`` named
  ``sfft_romansnpit-<version>.tar.gz`` where ``<version>`` is what you
  put in the ``VERSION`` variable.

* You can then upload it to the ``sfft-romansnpit`` package on PyPi with::

    twine upload dist/sfft_romansnpit-<version>.tar.gz

License
-------

This particular version uses the MIT License of SFFT and keeps its copyright notice
(``LICENSE``). It ships SExtractor 2.28.0 in ``sfft/spaceflow/prep/_sextractor/``,
under the GNU General Public License version 3 or later, and builds it into a
separate program that the single-exposure flow runs in its own process; that
folder holds the program's licence and notices, and the GPL covers the folder
and the program, not the rest of betasfft.