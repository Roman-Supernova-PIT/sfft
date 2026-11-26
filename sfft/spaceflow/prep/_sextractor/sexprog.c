/*
 * sexprog.c: SExtractor 2.28.0 as betasfft's separate program, sextractor228 (built by build.py, run by sfft.utils.astromatic.bundled).
 *
 * Part of a modified version of SExtractor, distributed under the GNU General Public License version 3 or later, as SExtractor itself (see
 * LICENSE, COPYRIGHT and NOTICE in this folder). Written for betasfft on 2026-11-22.
 *
 * The program is SExtractor: its main() (renamed sexlib_main by patches.py) runs through sexlib_run() in sexlib.c, which pins the C locale and
 * the SSE control word, starts the deblender's random stream from seed 1 and returns SExtractor's error code instead of exiting. One run per
 * process. Three ways to call it:
 *
 *   sextractor228 <SExtractor's arguments>            SExtractor's own program: the same command line, files on disk;
 *   sextractor228 --frames <SExtractor's arguments>   the same, but the files whose names start with "sexlib_mem_" are read from standard
 *                                                     input and written to standard output instead of the disk;
 *   sextractor228 --sexlib-info                       prints the version, the hash of the sources it was built from, and its compiler flags.
 *
 * Frames: standard input holds the line "SEXFRAMES 1", then for each input file a line "<name> <size>" followed by its size bytes, then the
 * line "END". After a successful run standard output holds "SEXFRAMES 1", then a line "<name> <size>" and the bytes of each file SExtractor
 * wrote, then the warnings as one more frame named "sexlib_warnings" (text, one per line), then "END". The files are SExtractor's own formats
 * (FITS images, configuration and parameter text, filter files, FITS catalogues). Whatever SExtractor prints goes to standard error. Exit
 * status 0 on success; on failure SExtractor's error code (1 when it has none) with its message on standard error, and no frames.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int sexlib_run(int argc, char **argv);
int sexlib_add_input(const char *name, const void *data, size_t size);
int sexlib_output_count(void);
const char *sexlib_output_name(int k);
const void *sexlib_output_data(int k);
size_t sexlib_output_size(int k);
const char *sexlib_error(void);
const char *sexlib_warnings(void);
const char *sexlib_source_hash(void);
const char *sexlib_build_flags(void);
const char *sexlib_version(void);

#define MAGIC "SEXFRAMES 1\n"

/* All of standard input; NULL on a read error or when memory runs out. */
static char *read_all(FILE *in, size_t *n)
  {
  size_t cap = (size_t)1 << 22, size = 0, got;
  char *buf = (char *)malloc(cap), *p;

  if (!buf)
    return NULL;
  while ((got = fread(buf + size, 1, cap - size, in)) > 0)
    {
    size += got;
    if (size == cap)
      {
      if (!(p = (char *)realloc(buf, 2 * cap)))
        {
        free(buf);
        return NULL;
        }
      buf = p;
      cap *= 2;
      }
    }
  if (ferror(in))
    {
    free(buf);
    return NULL;
    }
  *n = size;
  return buf;
  }

/* The line starting at p, without its newline, into line (of len bytes); the position after it, or NULL. */
static char *next_line(char *p, char *end, char *line, size_t len)
  {
  char *nl = (char *)memchr(p, '\n', (size_t)(end - p));

  if (!nl || (size_t)(nl - p) >= len)
    return NULL;
  memcpy(line, p, (size_t)(nl - p));
  line[nl - p] = '\0';
  return nl + 1;
  }

/* Reads the input frames and hands each file to sexlib.c, which reads it from this buffer during the run (*keep: free after the run). */
static int read_frames(char **keep)
  {
  size_t n, m = strlen(MAGIC);
  char *buf, *p, *end, line[256], name[128];
  unsigned long long size;

  if (!(buf = read_all(stdin, &n)))
    {
    fprintf(stderr, "sextractor228: cannot read the frames on standard input\n");
    return -1;
    }
  *keep = buf;
  end = buf + n;
  if (n < m || memcmp(buf, MAGIC, m))
    {
    fprintf(stderr, "sextractor228: standard input does not start with \"SEXFRAMES 1\"\n");
    return -1;
    }
  p = buf + m;
  for (;;)
    {
    if (!(p = next_line(p, end, line, sizeof(line))))
      {
      fprintf(stderr, "sextractor228: the frames on standard input end without \"END\"\n");
      return -1;
      }
    if (!strcmp(line, "END"))
      return 0;
    if (sscanf(line, "%127s %llu", name, &size) != 2 || size > (unsigned long long)(end - p))
      {
      fprintf(stderr, "sextractor228: bad frame header \"%s\"\n", line);
      return -1;
      }
    if (sexlib_add_input(name, p, (size_t)size))
      {
      fprintf(stderr, "sextractor228: cannot take the input file \"%s\" (too many files, or the name is too long)\n", name);
      return -1;
      }
    p += size;
    }
  }

static int write_frame(FILE *out, const char *name, const void *data, size_t size)
  {
  fprintf(out, "%s %llu\n", name, (unsigned long long)size);
  return (size && fwrite(data, 1, size, out) != size) ? -1 : 0;
  }

static int write_frames(FILE *out)
  {
  int k, n = sexlib_output_count();
  const char *w = sexlib_warnings();

  fputs(MAGIC, out);
  for (k=0; k<n; k++)
    if (write_frame(out, sexlib_output_name(k), sexlib_output_data(k), sexlib_output_size(k)))
      return -1;
  if (write_frame(out, "sexlib_warnings", w, strlen(w)))
    return -1;
  fputs("END\n", out);
  return fflush(out) ? -1 : 0;
  }

/* The warnings collected during the run, printed as SExtractor prints them (plain runs). */
static void print_warnings(void)
  {
  const char *w = sexlib_warnings(), *q;

  while (*w)
    {
    if (!(q = strchr(w, '\n')))
      q = w + strlen(w);
    fprintf(stderr, "\n> WARNING: %.*s\n\n", (int)(q - w), w);
    w = *q ? q + 1 : q;
    }
  }

/* The exit status for SExtractor's return code, with its message on standard error. */
static int status(int rc)
  {
  if (!rc)
    return 0;
  fprintf(stderr, "\n> %s\n\n", sexlib_error());      /* as SExtractor's error() prints it (its message starts with "*Error*: ") */
  return (rc & 0xFF) ? (rc & 0xFF) : 1;
  }

int main(int argc, char *argv[])
  {
  char *keep = NULL;
  int rc, out_fd, i;
  FILE *out;

  if (argc == 2 && !strcmp(argv[1], "--sexlib-info"))
    {
    printf("SExtractor %s\nsources %s\nflags %s\n", sexlib_version(), sexlib_source_hash(), sexlib_build_flags());
    return 0;
    }
  if (argc < 2 || strcmp(argv[1], "--frames"))
    {
    rc = sexlib_run(argc, argv);
    print_warnings();
    return status(rc);
    }
  if (read_frames(&keep))
    return 2;
  /* standard output carries only the frames: what SExtractor prints there goes to standard error */
  fflush(stdout);
  if ((out_fd = dup(STDOUT_FILENO)) < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0 || !(out = fdopen(out_fd, "wb")))
    {
    fprintf(stderr, "sextractor228: cannot set up standard output\n");
    return 2;
    }
  for (i=1; i<argc-1; i++)                /* argv[0], then SExtractor's arguments without "--frames" */
    argv[i] = argv[i+1];
  argv[argc-1] = NULL;
  rc = sexlib_run(argc - 1, argv);
  if (!rc && write_frames(out))
    {
    fprintf(stderr, "sextractor228: cannot write the frames on standard output\n");
    return 3;
    }
  free(keep);
  return status(rc);
  }
