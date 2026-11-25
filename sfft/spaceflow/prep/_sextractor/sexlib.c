/*
 * sexlib.c: the runtime of SExtractor 2.28.0 as betasfft's separate program (sexprog.c; run by sfft.utils.astromatic.bundled).
 *
 * Part of a modified version of SExtractor, distributed under the GNU General Public License version 3 or later, as SExtractor itself
 * (see LICENSE, COPYRIGHT and NOTICE in this folder). Written in 2026-11, used by betasfft since 2026-11-22; the program's own sources in src/ are those of the 2.28.0
 * release, changed only by the edits in patches.py, which route the calls below into this file.
 *
 * What this file adds to the program:
 *   - sexlib_run(): runs SExtractor's own main() (renamed sexlib_main by patches.py) with the SSE control word (x86-64) and the C locale
 *     pinned for the call, and with error() returning here through longjmp instead of ending the process;
 *   - memory files: names that start with "sexlib_mem_" open in-memory streams (glibc fopencookie, macOS funopen) instead of files, so the
 *     image, weight map, configuration, parameter list and filter come from the caller's buffers and the catalogue and check images go to
 *     buffers the caller reads back; every other name goes to fopen();
 *   - a private copy of glibc's random() (TYPE_3) for the deblender, reseeded with 1 at every call: the stream a new process of the
 *     executable starts from under glibc, on every platform;
 *   - warnings collected instead of printed; the SEGMENTATION check image copied straight into the caller's buffer.
 *
 * SExtractor keeps static state that does not survive a second run in one process (a second run crashes), so the program
 * makes one run per process.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <locale.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <xlocale.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define SEXLIB_X86 1
#endif

#define SEXLIB_API __attribute__((visibility("default")))
#ifndef SEXLIB_SOURCE_HASH
#define SEXLIB_SOURCE_HASH "unknown"
#endif
#ifndef SEXLIB_BUILD_FLAGS
#define SEXLIB_BUILD_FLAGS "unknown"
#endif
#define SEXLIB_PREFIX "sexlib_mem_"
#define SEXLIB_NFILES 32

int sexlib_main(int argc, char *argv[]);        /* SExtractor's main() (src/main.c), renamed by patches.py */

/*---------------------------------------------------------------- glibc's random() and rand(), TYPE_3 (degree 31, separation 3) */
/* sexlib_srand() and sexlib_rand() follow __srandom_r() and __random_r() of the GNU C Library (stdlib/random_r.c), whose notices are:

   Copyright (C) 1995-2026 Free Software Foundation, Inc.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.

   Copyright (C) 1983 Regents of the University of California.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   4. Neither the name of the University nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
   OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
   SUCH DAMAGE.

   This is derived from the Berkeley source:
   	@(#)random.c	5.5 (Berkeley) 7/6/88
   It was reworked for the GNU C Library by Roland McGrath.
   Rewritten to be reentrant by Ulrich Drepper, 1995

   Here the code is used under the GNU General Public License version 3 or later, with the rest of this program, as section 3 of the GNU
   Lesser General Public License version 2.1 allows. NOTICE in this folder repeats these notices. */
static int32_t sl_state[31];
static int sl_f = 3, sl_r = 0;

int sexlib_rand(void)
  {
  uint32_t val = (uint32_t)sl_state[sl_f] + (uint32_t)sl_state[sl_r];
  int result;

  sl_state[sl_f] = (int32_t)val;
  result = (int)(val >> 1);
  if (++sl_f >= 31)
    {
    sl_f = 0;
    ++sl_r;
    }
  else if (++sl_r >= 31)
    sl_r = 0;
  return result;
  }

void sexlib_srand(unsigned int seed)
  {
  long int i, hi, lo;
  int32_t word;

  if (seed == 0)
    seed = 1;
  sl_state[0] = (int32_t)seed;
  word = (int32_t)seed;
  for (i=1; i<31; i++)
    {
    hi = word / 127773;
    lo = word % 127773;
    word = 16807 * lo - 2836 * hi;
    if (word < 0)
      word += 2147483647;
    sl_state[i] = word;
    }
  sl_f = 3;
  sl_r = 0;
  for (i=0; i<310; i++)
    (void)sexlib_rand();
  }

/*---------------------------------------------------------------- memory files */
typedef struct
  {
  char name[96];
  char *data;          /* inputs: the caller's buffer (not owned); outputs: owned, grown on write */
  size_t size, cap;
  int output;
  } memfile;

typedef struct
  {
  memfile *f;
  size_t pos;
  FILE *fp;
  } memcookie;

static memfile sl_files[SEXLIB_NFILES];
static int sl_nfiles = 0;
/* The memory streams that are open. A stream SExtractor leaves open (when error() returns here in the middle of a run) is closed by
   sexlib_run before it returns: glibc would otherwise flush it at the end of the process through functions of a library already unloaded. */
#define SEXLIB_NOPEN 64
static memcookie *sl_open[SEXLIB_NOPEN];
static int sl_nopen = 0;

static memfile *find_file(const char *name, int output)
  {
  int i;

  for (i=0; i<sl_nfiles; i++)
    if (sl_files[i].output == output && !strcmp(sl_files[i].name, name))
      return sl_files + i;
  return NULL;
  }

static memfile *new_file(const char *name, int output)
  {
  memfile *f;

  if (sl_nfiles >= SEXLIB_NFILES || strlen(name) >= sizeof(sl_files[0].name))
    return NULL;
  f = sl_files + sl_nfiles++;
  memset(f, 0, sizeof(*f));
  strcpy(f->name, name);
  f->output = output;
  return f;
  }

static ssize_t mf_read(void *cookie, char *buf, size_t n)
  {
  memcookie *c = (memcookie *)cookie;
  size_t left;

  if (c->pos >= c->f->size)
    return 0;
  left = c->f->size - c->pos;
  if (n > left)
    n = left;
  memcpy(buf, c->f->data + c->pos, n);
  c->pos += n;
  return (ssize_t)n;
  }

static ssize_t mf_write(void *cookie, const char *buf, size_t n)
  {
  memcookie *c = (memcookie *)cookie;
  memfile *f = c->f;
  size_t end = c->pos + n, cap;
  char *p;

  if (!f->output)
    {
    errno = EBADF;
    return -1;
    }
  if (end > f->cap)
    {
    cap = f->cap ? f->cap : 65536;
    while (cap < end)
      cap *= 2;
    if (!(p = (char *)realloc(f->data, cap)))
      {
      errno = ENOMEM;
      return -1;
      }
    f->data = p;
    f->cap = cap;
    }
  if (c->pos > f->size)
    memset(f->data + f->size, 0, c->pos - f->size);
  memcpy(f->data + c->pos, buf, n);
  c->pos = end;
  if (end > f->size)
    f->size = end;
  return (ssize_t)n;
  }

static int mf_seekto(memcookie *c, long long offset, int whence, long long *result)
  {
  long long base;

  switch (whence)
    {
    case SEEK_SET:
      base = 0;
      break;
    case SEEK_CUR:
      base = (long long)c->pos;
      break;
    case SEEK_END:
      base = (long long)c->f->size;
      break;
    default:
      errno = EINVAL;
      return -1;
    }
  if (base + offset < 0)
    {
    errno = EINVAL;
    return -1;
    }
  c->pos = (size_t)(base + offset);
  *result = (long long)c->pos;
  return 0;
  }

static int mf_close(void *cookie)
  {
  int i;

  for (i=0; i<sl_nopen; i++)
    if (sl_open[i] == (memcookie *)cookie)
      {
      sl_open[i] = sl_open[--sl_nopen];
      break;
      }
  free(cookie);
  return 0;
  }

static void close_open_streams(void)
  {
  while (sl_nopen > 0)
    fclose(sl_open[sl_nopen - 1]->fp);           /* mf_close takes it off the list */
  }

#ifdef __APPLE__
static int mfa_read(void *cookie, char *buf, int n)
  {
  return (int)mf_read(cookie, buf, (size_t)n);
  }

static int mfa_write(void *cookie, const char *buf, int n)
  {
  return (int)mf_write(cookie, buf, (size_t)n);
  }

static fpos_t mfa_seek(void *cookie, fpos_t offset, int whence)
  {
  long long result;

  if (mf_seekto((memcookie *)cookie, (long long)offset, whence, &result))
    return (fpos_t)-1;
  return (fpos_t)result;
  }
#else
static int mf_seek(void *cookie, off64_t *offset, int whence)
  {
  long long result;

  if (mf_seekto((memcookie *)cookie, (long long)*offset, whence, &result))
    return -1;
  *offset = (off64_t)result;
  return 0;
  }
#endif

/* fopen() for SExtractor: names starting with SEXLIB_PREFIX are memory files, anything else a real file. A memory name opened for reading
   must have been given by the caller (else NULL with ENOENT, as for a missing file); opened for writing it becomes an output, emptied first. */
FILE *sexlib_fopen(const char *name, const char *mode)
  {
  memfile *f;
  memcookie *c;
  int output;

  if (strncmp(name, SEXLIB_PREFIX, strlen(SEXLIB_PREFIX)))
    return fopen(name, mode);
  output = (mode[0] != 'r');
  if (!(f = find_file(name, output)))
    {
    if (!output || !(f = new_file(name, 1)))
      {
      errno = ENOENT;
      return NULL;
      }
    }
  if (output)
    f->size = 0;
  if (sl_nopen >= SEXLIB_NOPEN || !(c = (memcookie *)malloc(sizeof(memcookie))))
    {
    errno = EMFILE;
    return NULL;
    }
  c->f = f;
  c->pos = 0;
#ifdef __APPLE__
  c->fp = funopen(c, output ? NULL : mfa_read, output ? mfa_write : NULL, mfa_seek, mf_close);
#else
  {
  cookie_io_functions_t io;

  io.read = output ? NULL : mf_read;
  io.write = output ? mf_write : NULL;
  io.seek = mf_seek;
  io.close = mf_close;
  c->fp = fopencookie(c, output ? "wb" : "rb", io);
  }
#endif
  if (!c->fp)
    {
    free(c);
    return NULL;
    }
  sl_open[sl_nopen++] = c;
  return c->fp;
  }

/*---------------------------------------------------------------- errors and warnings */
static jmp_buf sl_jmp;
static int sl_active = 0, sl_errcode = 0, sl_nruns = 0;
static char sl_errmsg[1024] = "";
static char *sl_warnings = NULL;
static size_t sl_nwarn = 0, sl_capwarn = 0;

/* Called first thing in error() (src/fits/fitsmisc.c): during sexlib_run, keep the message and return to sexlib_run; otherwise return, and
   error() goes on as in the program (prints and exits). */
void sexlib_error_hook(int num, const char *msg1, const char *msg2)
  {
  if (!sl_active)
    return;
  snprintf(sl_errmsg, sizeof(sl_errmsg), "%s%s", msg1 ? msg1 : "", msg2 ? msg2 : "");
  sl_errcode = num ? num : 1;
  sl_active = 0;
  longjmp(sl_jmp, 1);
  }

/* Called first thing in warning(): during sexlib_run, append "msg1msg2\n" to the warnings and tell warning() to print nothing. */
int sexlib_warning_hook(const char *msg1, const char *msg2)
  {
  size_t n;
  char *p;

  if (!sl_active)
    return 0;
  n = strlen(msg1 ? msg1 : "") + strlen(msg2 ? msg2 : "") + 2;
  if (sl_nwarn + n > sl_capwarn)
    {
    size_t cap = sl_capwarn ? 2 * sl_capwarn : 4096;
    while (cap < sl_nwarn + n)
      cap *= 2;
    if (!(p = (char *)realloc(sl_warnings, cap)))
      return 1;
    sl_warnings = p;
    sl_capwarn = cap;
    }
  sl_nwarn += (size_t)sprintf(sl_warnings + sl_nwarn, "%s%s\n", msg1 ? msg1 : "", msg2 ? msg2 : "");
  return 1;
  }

/*---------------------------------------------------------------- the segmentation image */
static int32_t *sl_seg = NULL;
static size_t sl_segnpix = 0;
static int sl_segdone = 0;

/* Called by reendcheck() (src/check.c) for the SEGMENTATION check image: copy it into the caller's buffer and return 1 (the image is then
   not written to its file), or return 0 when the caller gave no buffer of that size. */
int sexlib_segmentation(const unsigned int *pix, size_t npix)
  {
  size_t i;

  if (!sl_seg || npix != sl_segnpix)
    return 0;
  for (i=0; i<npix; i++)
    sl_seg[i] = (int32_t)pix[i];
  sl_segdone = 1;
  return 1;
  }

/*---------------------------------------------------------------- the interface */
SEXLIB_API int sexlib_runs(void)
  {
  return sl_nruns;
  }

SEXLIB_API const char *sexlib_source_hash(void)
  {
  return SEXLIB_SOURCE_HASH;
  }

SEXLIB_API const char *sexlib_build_flags(void)
  {
  return SEXLIB_BUILD_FLAGS;
  }

SEXLIB_API const char *sexlib_version(void)
  {
  return "2.28.0";
  }

SEXLIB_API const char *sexlib_error(void)
  {
  return sl_errmsg;
  }

SEXLIB_API const char *sexlib_warnings(void)
  {
  return sl_warnings ? sl_warnings : "";
  }

/* An input memory file: the caller keeps the buffer alive until sexlib_run returns. Returns 0, or -1 when the table is full. */
SEXLIB_API int sexlib_add_input(const char *name, const void *data, size_t size)
  {
  memfile *f;

  if (!(f = find_file(name, 0)) && !(f = new_file(name, 0)))
    return -1;
  f->data = (char *)data;
  f->size = size;
  f->cap = size;
  return 0;
  }

/* Where the SEGMENTATION check image goes: npix int32 values, (y, x) order as in the FITS file. */
SEXLIB_API void sexlib_set_segmentation(int32_t *buf, size_t npix)
  {
  sl_seg = buf;
  sl_segnpix = npix;
  sl_segdone = 0;
  }

SEXLIB_API int sexlib_segmentation_done(void)
  {
  return sl_segdone;
  }

SEXLIB_API int sexlib_output_count(void)
  {
  int i, n = 0;

  for (i=0; i<sl_nfiles; i++)
    n += sl_files[i].output;
  return n;
  }

static memfile *output_at(int k)
  {
  int i;

  for (i=0; i<sl_nfiles; i++)
    if (sl_files[i].output && !k--)
      return sl_files + i;
  return NULL;
  }

SEXLIB_API const char *sexlib_output_name(int k)
  {
  memfile *f = output_at(k);
  return f ? f->name : NULL;
  }

SEXLIB_API const void *sexlib_output_data(int k)
  {
  memfile *f = output_at(k);
  return f ? f->data : NULL;
  }

SEXLIB_API size_t sexlib_output_size(int k)
  {
  memfile *f = output_at(k);
  return f ? f->size : 0;
  }

/* Frees what the library holds for the caller (outputs, warnings); call before closing the library. */
SEXLIB_API void sexlib_release(void)
  {
  int i;

  for (i=0; i<sl_nfiles; i++)
    if (sl_files[i].output)
      free(sl_files[i].data);
  sl_nfiles = 0;
  free(sl_warnings);
  sl_warnings = NULL;
  sl_nwarn = sl_capwarn = 0;
  sl_seg = NULL;
  sl_segnpix = 0;
  }

/* For the tests of the generator copy (L0). */
SEXLIB_API void sexlib_test_srand(unsigned int seed)
  {
  sexlib_srand(seed);
  }

SEXLIB_API int sexlib_test_rand(void)
  {
  return sexlib_rand();
  }

/* Runs SExtractor's main() on argv (argv[0] a name, then the image name, -c and the configuration's name, as for the executable). Returns
   main()'s value (0), or error()'s code (>= 1) with the message in sexlib_error(). */
SEXLIB_API int sexlib_run(int argc, char **argv)
  {
  volatile int rc;
#ifdef SEXLIB_X86
  unsigned int csr;
#endif
  locale_t cloc, oldloc;

  sl_nruns++;
  sl_errcode = 0;
  sl_errmsg[0] = '\0';
#ifdef SEXLIB_X86
  csr = _mm_getcsr();
  _mm_setcsr(0x1F80);
#endif
  cloc = newlocale(LC_ALL_MASK, "C", (locale_t)0);
  oldloc = cloc ? uselocale(cloc) : (locale_t)0;
  sexlib_srand(1);
  if (setjmp(sl_jmp))
    rc = sl_errcode ? sl_errcode : 1;
  else
    {
    sl_active = 1;
    rc = sexlib_main(argc, argv);
    sl_active = 0;
    }
  sl_active = 0;
  close_open_streams();
  fflush(stdout);
  fflush(stderr);
  if (cloc)
    {
    uselocale(oldloc);
    freelocale(cloc);
    }
#ifdef SEXLIB_X86
  _mm_setcsr(csr);
#endif
  return rc;
  }
