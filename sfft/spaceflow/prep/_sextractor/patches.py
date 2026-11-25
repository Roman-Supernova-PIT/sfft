"""The edits that turn SExtractor 2.28.0 (src/, as released) into betasfft's SExtractor program (sexlib.c, sexprog.c; build.py builds it).

    python patches.py apply <this folder> <build folder>     copies src/, config.h, sexlib.c and sexprog.c into the build folder and applies the edits
    python patches.py hash <this folder>                     prints the SHA-1 the build embeds in the program (sfft.utils.astromatic.bundled checks it)

Each edit replaces one exact piece of a source file and must match exactly once, else the build stops. The edits only route calls into
sexlib.c; no computation of the program is changed:
  main         main() becomes sexlib_main(), which sexlib_run() calls;
  error        error() first calls sexlib_error_hook(), which returns to sexlib_run() during a call (the process does not exit);
  warning      warning() first calls sexlib_warning_hook(), which collects the message during a call;
  rand         the deblender's rand() (refine.c) and the one in fitswcs.c call sexlib_rand(), a copy of glibc's generator reseeded with 1 at
               every call (a new process of the executable under glibc starts from that seed);
  fopen        the five places that open files (images, catalogue and check images, configuration, parameter list, filter, .head files)
               call sexlib_fopen(), which opens memory files for names that start with "sexlib_mem_" and real files otherwise;
  segmentation reendcheck() hands the SEGMENTATION check image to sexlib_segmentation(), which copies it into a caller's buffer when one is given (the program gives none: the image is written as SExtractor writes it).
Part of a modified version of SExtractor (GPL v3 or later), see README.md.
"""
import hashlib
import os
import shutil
import sys

FOPEN = "FILE *sexlib_fopen(const char *name, const char *mode);\n"

EDITS = [
    ("main", "src/main.c",
     "int\tmain(int argc, char *argv[])\n",
     "int\tsexlib_main(int argc, char *argv[])\n"),
    ("error", "src/fits/fitsmisc.c",
     "static void\t(*errorfunc)(const char *msg1, const char *msg2) = NULL;\n",
     "static void\t(*errorfunc)(const char *msg1, const char *msg2) = NULL;\n"
     "void\tsexlib_error_hook(int num, const char *msg1, const char *msg2);\n"
     "int\tsexlib_warning_hook(const char *msg1, const char *msg2);\n"),
    ("error", "src/fits/fitsmisc.c",
     "  {\n  fprintf(stderr, \"\\n> %s%s\\n\\n\",msg1,msg2);\n",
     "  {\n  sexlib_error_hook(num, msg1, msg2);\n  fprintf(stderr, \"\\n> %s%s\\n\\n\",msg1,msg2);\n"),
    ("warning", "src/fits/fitsmisc.c",
     "void    warning(char *msg1, char *msg2)\n  {\n   time_t\twarntime;\n   struct tm\t*tm;\n\n",
     "void    warning(char *msg1, char *msg2)\n  {\n   time_t\twarntime;\n   struct tm\t*tm;\n\n"
     "  if (sexlib_warning_hook(msg1, msg2))\n    return;\n"),
    ("rand", "src/refine.c",
     "static objliststruct\t*objlist;\n",
     "int\tsexlib_rand(void);\nstatic objliststruct\t*objlist;\n"),
    ("rand", "src/refine.c",
     "        drand = p[nobj-1]*rand()/RAND_MAX;",
     "        drand = p[nobj-1]*sexlib_rand()/RAND_MAX;"),
    ("rand", "src/fitswcs.c",
     "/******* create_wcs ***********************************************************",
     "int\tsexlib_rand(void);\n/******* create_wcs ***********************************************************"),
    ("rand", "src/fitswcs.c",
     "(0.5-(double)rand()/RAND_MAX)",
     "(0.5-(double)sexlib_rand()/RAND_MAX)"),
    ("fopen", "src/fits/fitscat.c",
     "\nint\topen_cat(catstruct *cat, access_type_t at)\n",
     "\n" + FOPEN + "int\topen_cat(catstruct *cat, access_type_t at)\n"),
    ("fopen", "src/fits/fitscat.c",
     "    if ((cat->file = fopen(cat->filename, at==WRITE_ONLY?\"wb\":\"rb\")) == NULL)",
     "    if ((cat->file = sexlib_fopen(cat->filename, at==WRITE_ONLY?\"wb\":\"rb\")) == NULL)"),
    ("fopen", "src/prefs.c",
     "\nvoid    readprefs(char *filename, char **argkey, char **argval, int narg)\n",
     "\n" + FOPEN + "void    readprefs(char *filename, char **argkey, char **argval, int narg)\n"),
    ("fopen", "src/prefs.c",
     "  if ((infile = fopen(filename,\"r\")) == NULL)",
     "  if ((infile = sexlib_fopen(filename,\"r\")) == NULL)"),
    ("fopen", "src/catout.c",
     "\nvoid\treadcatparams(char *filename)\n",
     "\n" + FOPEN + "void\treadcatparams(char *filename)\n"),
    ("fopen", "src/catout.c",
     "  if ((infile = fopen(filename,\"r\")) == NULL)",
     "  if ((infile = sexlib_fopen(filename,\"r\")) == NULL)"),
    ("fopen", "src/filter.c",
     "\nint\tgetconv(char *filename)\n",
     "\n" + FOPEN + "int\tgetconv(char *filename)\n"),
    ("fopen", "src/filter.c",
     "  if (!(file = fopen(filename, \"r\")))",
     "  if (!(file = sexlib_fopen(filename, \"r\")))"),
    ("fopen", "src/header.c",
     "\nint     read_aschead(char *filename, int frameno, tabstruct *tab)\n",
     "\n" + FOPEN + "int     read_aschead(char *filename, int frameno, tabstruct *tab)\n"),
    ("fopen", "src/header.c",
     "  if ((file=fopen(filename, \"r\")))",
     "  if ((file=sexlib_fopen(filename, \"r\")))"),
    ("segmentation", "src/check.c",
     "\nvoid\treendcheck(picstruct *field, checkstruct *check)\n",
     "\nint\tsexlib_segmentation(const unsigned int *pix, size_t npix);\nvoid\treendcheck(picstruct *field, checkstruct *check)\n"),
    ("segmentation", "src/check.c",
     "    case CHECK_SEGMENTATION:\n      write_ibody(cat->tab, check->pix, check->npix);\n",
     "    case CHECK_SEGMENTATION:\n      if (!sexlib_segmentation((const unsigned int *)check->pix, (size_t)check->npix))\n"
     "        write_ibody(cat->tab, check->pix, check->npix);\n"),
]

HASH_SKIP = ("README", "COPYRIGHT", "LICENSE", "AUTHORS", "NOTICE")


def source_files(folder):
    """The files the program is built from, as sorted paths relative to folder: src/ (the release's sources), patches.py, sexlib.c, sexprog.c,
    build.py, config.h and SOURCES. Documentation, licence and notice files and the build/ folder (where build.py puts the program) are left
    out, so editing them, or building, does not make a built program stale."""
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", "build"))
        for name in sorted(filenames):
            if name.startswith(".") or name.endswith((".pyc", ".pyo")) or name.startswith(HASH_SKIP):
                continue
            out.append(os.path.relpath(os.path.join(dirpath, name), folder).replace(os.sep, "/"))
    return sorted(out)


def source_hash(folder):
    """SHA-1 of the names and contents of source_files(folder)."""
    h = hashlib.sha1()
    for rel in source_files(folder):
        h.update(rel.encode() + b"\0")
        with open(os.path.join(folder, rel), "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return h.hexdigest()


def apply(folder, build):
    """Copy src/, config.h, sexlib.c, sexprog.c and SOURCES into build (which must not exist yet or be empty) and apply EDITS there."""
    if os.path.exists(build) and os.listdir(build):
        sys.exit(f"{build} is not empty")
    os.makedirs(build, exist_ok=True)
    shutil.copytree(os.path.join(folder, "src"), os.path.join(build, "src"))
    for name in ("config.h", "sexlib.c", "sexprog.c", "SOURCES"):
        shutil.copy2(os.path.join(folder, name), os.path.join(build, name))
    for tag, rel, old, new in EDITS:
        path = os.path.join(build, rel)
        with open(path) as f:
            text = f.read()
        n = text.count(old)
        if n != 1:
            sys.exit(f"patch {tag}: the piece to replace occurs {n} times in {rel}, not once:\n{old!r}")
        with open(path, "w") as f:
            f.write(text.replace(old, new))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "apply":
        apply(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "hash":
        print(source_hash(sys.argv[2]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
