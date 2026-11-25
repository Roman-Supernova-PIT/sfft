/* config.h for SExtractor 2.28.0 built as betasfft's separate program (build.py, no configure script).

   The sources read only these macros from it. Their values are those SExtractor's own configure script set on Perlmutter for
   "--disable-model-fitting --disable-cfitsio" (2026-11-21); a build with that configure script gave the conda-forge executable's catalogue and
   segmentation byte for byte. Model fitting (USE_MODEL), CFITSIO (HAVE_CFITSIO) and threads (USE_THREADS) stay undefined, as there.
   DATE is the date the conda-forge executable 2.28.0 reports; it only appears in banners and catalogue headers, never in measurements. */
#define VERSION "2.28.0"
#define DATE "2024-06-24"
#define THREADS_NMAX 1024
#define XSL_URL "file:///usr/local/share/sextractor/sextractor.xsl"
#define PACKAGE "sextractor"
#define PACKAGE_NAME "SExtractor"
#define PACKAGE_VERSION "2.28.0"
#define PACKAGE_STRING "SExtractor 2.28.0"
#define PACKAGE_TARNAME "sextractor"
#define PACKAGE_BUGREPORT "astromatic@astromatic.iap.fr"
#define HAVE_LONG_LONG_INT 1
#define HAVE_UNSIGNED_LONG_LONG_INT 1
#define HAVE_SYS_MMAN_H 1
#define HAVE_GETENV 1
