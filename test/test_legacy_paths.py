"""The SFFT 1.6.4 release's import paths: each old path is its sfft.legacy module (or name) itself and gives a DeprecationWarning, and
sfft.SpaceSFFTFlow, which holds the wrapper, gives none. Run from the repository root: python -m pytest test"""
import importlib
import importlib.util
import os
import subprocess
import sys

import pytest

import sfft

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDS_CUPY = ("utils.CupyWCSTransform", "utils.ResampKits")       # these two import cupy when they load, on the release as here
HAS_CUPY = importlib.util.find_spec("cupy") is not None


def skip_without_cupy(name):
    if name in NEEDS_CUPY and not HAS_CUPY:
        pytest.skip(f"sfft.legacy.{name} imports cupy, which is not installed here")


def run_python(code):
    """Runs code in a new interpreter from the repository root (so that a warning is the first one of its import), and returns it."""
    return subprocess.run([sys.executable, "-c", code], cwd=REPOSITORY, capture_output=True, text=True)


@pytest.mark.parametrize("name", sfft.LEGACY_MODULES)
def test_old_module_path_is_the_legacy_module(name):
    skip_without_cupy(name)
    assert importlib.import_module("sfft." + name) is importlib.import_module("sfft.legacy." + name)


@pytest.mark.parametrize("name", sfft.LEGACY_MODULES)
def test_old_module_path_warns(name):
    skip_without_cupy(name)
    code = ("import warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            f"    import sfft.{name}\n"
            f"wanted = 'sfft.{name} is deprecated: import sfft.legacy.{name} instead'\n"
            "found = [w for w in caught if w.category is DeprecationWarning and str(w.message) == wanted]\n"
            "assert len(found) == 1, [str(w.message) for w in caught]\n"
            "assert found[0].filename == '<string>', found[0].filename\n")
    result = run_python(code)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name, module", sorted(sfft.LEGACY_NAMES.items()))
def test_old_top_level_name_is_the_legacy_name_and_warns(name, module):
    code = ("import importlib, warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            f"    from sfft import {name}\n"
            f"assert {name} is getattr(importlib.import_module('sfft.legacy.{module}'), '{name}')\n"
            f"wanted = 'sfft.{name} is deprecated: use sfft.legacy.{name} (from sfft.legacy import {name})'\n"
            "found = [w for w in caught if w.category is DeprecationWarning and str(w.message) == wanted]\n"
            "assert len(found) == 1, [str(w.message) for w in caught]\n")
    result = run_python(code)
    assert result.returncode == 0, result.stderr


def test_space_sfft_flow_path_gives_the_wrapper_without_warning():
    code = ("import warnings\n"
            "warnings.simplefilter('error', DeprecationWarning)\n"
            "from sfft.SpaceSFFTFlow import SpaceSFFT_Flow\n"
            "from sfft import SpaceSFFT\n"
            "assert SpaceSFFT_Flow is SpaceSFFT\n")
    result = run_python(code)
    assert result.returncode == 0, result.stderr


def test_import_sfft_imports_neither_engine():
    code = ("import sys\n"
            "import sfft\n"
            "loaded = sorted(m for m in sys.modules if m.startswith(('sfft.legacy', 'sfft.spaceflow')))\n"
            "assert not loaded, loaded\n")
    result = run_python(code)
    assert result.returncode == 0, result.stderr


def test_unknown_top_level_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'NotAName'"):
        sfft.NotAName
