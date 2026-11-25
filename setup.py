import os
import subprocess
import sys

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

HERE = os.path.dirname(os.path.abspath(__file__))
DESCRIPTION = "Image Subtraction in Fourier Space"
LONG_DESCRIPTION = open('README.rst').read()
NAME = "sfft-romansnpit"
AUTHOR = "Lei Hu"
AUTHOR_EMAIL = "leihu@sas.upenn.edu"
MAINTAINER = "Lei Hu"
MAINTAINER_EMAIL = "leihu@sas.upenn.edu"
DOWNLOAD_URL = 'https://github.com/thomasvrussell/sfft'

# betasfft: the MIT License (LICENSE). The package also ships SExtractor 2.28.0 in sfft/spaceflow/prep/_sextractor (GNU GPL v3 or later),
# built into a separate program that sfft.spaceflow runs in its own process; that folder holds its licence and notices.
LICENSE = 'MIT AND GPL-3.0-or-later'
LICENSE_FILES = ['LICENSE',
                 'sfft/spaceflow/prep/_sextractor/LICENSE', 'sfft/spaceflow/prep/_sextractor/COPYRIGHT',
                 'sfft/spaceflow/prep/_sextractor/AUTHORS', 'sfft/spaceflow/prep/_sextractor/NOTICE',
                 'sfft/spaceflow/prep/_sextractor/src/wcs/LICENSE']
VERSION = '1.6.4.dev20'

# SExtractor's sources, licence files and build script travel with the package: the program is built from them, and the GPL asks for
# them beside it. The folder's build/ (a program built in the source tree) does not: each installation builds its own.
SEXTRACTOR_DIR = ('sfft', 'spaceflow', 'prep', '_sextractor')      # the folder, in the package sfft.spaceflow.prep
SEXTRACTOR_DATA = ['_sextractor/' + p for p in ('*.py', '*.c', '*.h', 'SOURCES', 'LICENSE', 'COPYRIGHT', 'AUTHORS', 'NOTICE', 'README.md',
                                                  'src/*', 'src/fits/*', 'src/wcs/*')]


class BuildPyWithSExtractor(build_py):
    """build_py, then betasfft's SExtractor program built into the package (sfft/spaceflow/prep/_sextractor/build/sextractor228) by
    sfft/spaceflow/prep/_sextractor/build.py, run as a separate process; an editable install builds it in the source tree. Without a C
    compiler the package installs without it, and the single-exposure flow stops with the command that builds it."""

    def run(self):
        super().run()
        script = os.path.join(HERE, *SEXTRACTOR_DIR, 'build.py')
        cmd = [sys.executable, script] + ([] if getattr(self, 'editable_mode', False) else
                                          ['--out', os.path.join(self.build_lib, *SEXTRACTOR_DIR, 'build')])
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            sys.stderr.write("WARNING: betasfft's SExtractor program was not built; build it later with "
                             f"`python {script}`:\n{p.stdout}{p.stderr}\n")
        else:
            sys.stdout.write(p.stdout)


class BinaryDistribution(Distribution):
    """The package holds a compiled program, so its wheels are tagged for their platform."""

    def has_ext_modules(self):
        return True


install_reqs = ['scipy>=1.5.2',
                'astropy>=3.2.3',
                'scikit-image>=0.25.2',
                'fastremap>=1.7.0',
                'sep>=1.0.3',
                'numba>=0.53.1',
                'llvmlite>=0.36.0',
                'pyfftw>=0.12.0',
                'drizzle>=3.0.0']

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      setup_requires=['numpy'],
      install_requires=install_reqs,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      license_files=LICENSE_FILES,
      packages=find_packages(),
      include_package_data=True,
      package_data={'sfft.spaceflow.prep': SEXTRACTOR_DATA},
      cmdclass={'build_py': BuildPyWithSExtractor},
      distclass=BinaryDistribution,
      zip_safe=False,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Astronomy'],
     )
