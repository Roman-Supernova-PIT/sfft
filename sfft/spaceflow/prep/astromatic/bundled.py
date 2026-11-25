"""SExtractor 2.28.0 as a separate program bundled with betasfft: the conda-forge executable's catalogue and segmentation, byte for byte, with
nothing written to disk.

betasfft ships SExtractor's own C code in sfft/spaceflow/prep/_sextractor (the 2.28.0 release's sources, edits that route its files and
errors, a main of its own; GNU GPL v3 or later, see that folder's README.md). It is built into a program,
sextractor228, when betasfft is installed, or by hand with `python <sfft>/spaceflow/prep/_sextractor/build.py`. This module (MIT, as
the rest of betasfft) runs that program once per call, as SFFT's wrapper runs the executable: the inputs (the image as the very FITS bytes
the wrapper writes to its temporary file, the weight map, the configuration, the parameter list, the filter) go to it through a pipe, and its
outputs (the FITS catalogue and the check images) come back through a pipe, all in SExtractor's own formats. Each call is a new process, so
SExtractor's static state starts clean and its deblender's random stream starts from seed 1, as in a new process of the executable under
glibc, on every platform.

    tool = BundledSExtractor()          # BundledSExtractorError when the program is not built or was built from other sources
    res = tool.run(image=img, config=SExConfig(...), columns=(...), checkimages=("SEGMENTATION",), weight=w)
    res.catalog, res.checkimages["SEGMENTATION"], res.warnings, res.timing

run() takes the arguments of sfft.spaceflow.prep.astromatic.sextractor.SExtractor.run and returns its SExResult, plus warnings and timing.
Arrays read from ASDF files go through np.asarray like any other.
"""
import hashlib
import io
import os
import subprocess
import sys
import threading
import time

import numpy as np

VERSION = "2.28.0"
PROGRAM_NAME = "sextractor228"
PREFIX = "sexlib_mem_"                                  # names the program reads from and writes to the pipes
SOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_sextractor")      # sfft/spaceflow/prep/_sextractor
PROGRAM = os.path.join(SOURCE_DIR, "build", PROGRAM_NAME)
HASH_SKIP = ("README", "COPYRIGHT", "LICENSE", "AUTHORS", "NOTICE")      # as _sextractor/patches.py
_MAGIC = b"SEXFRAMES 1\n"


class BundledSExtractorError(RuntimeError):
    """betasfft's SExtractor program cannot be used: not built, built from other sources than the package's, or it broke off its answer."""


class SExtractorError(RuntimeError):
    """An error SExtractor itself reported (the message of its error(): an unreadable image, a configuration it refuses, ...)."""


def build_command():
    return f"{sys.executable} {os.path.join(SOURCE_DIR, 'build.py')}"


def build(out=None):
    """Builds the program (_sextractor/build.py, run as a separate process); returns its output."""
    cmd = [sys.executable, os.path.join(SOURCE_DIR, "build.py")] + (["--out", out] if out else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise BundledSExtractorError(f"building the SExtractor program failed:\n{p.stdout}{p.stderr}")
    return p.stdout


def source_hash(folder=SOURCE_DIR):
    """SHA-1 of the sources the program is built from: the files and the digest of `python patches.py hash` in that folder, which build.py
    embeds in the program (documentation, licence and notice files, caches and the build/ folder are left out)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", "build"))
        for name in sorted(filenames):
            if name.startswith(".") or name.endswith((".pyc", ".pyo")) or name.startswith(HASH_SKIP):
                continue
            files.append(os.path.relpath(os.path.join(dirpath, name), folder).replace(os.sep, "/"))
    h = hashlib.sha1()
    for rel in sorted(files):
        h.update(rel.encode() + b"\0")
        with open(os.path.join(folder, rel), "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return h.hexdigest()


def program_info(program=PROGRAM):
    """What the program reports about itself: its version ('SExtractor'), its sources' hash ('sources') and compiler flags ('flags')."""
    p = subprocess.run([program, "--sexlib-info"], capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise BundledSExtractorError(f"{program} --sexlib-info failed:\n{p.stdout}{p.stderr}")
    return dict(line.split(" ", 1) for line in p.stdout.splitlines() if " " in line)


def _fits_bytes(data, header=None):
    """The bytes SFFT's wrapper writes for an array (fits.PrimaryHDU(data, header).writeto), here into memory."""
    from astropy.io import fits
    buf = io.BytesIO()
    fits.PrimaryHDU(data=data, header=header).writeto(buf)
    return buf.getvalue()


def _parse_frames(out):
    """The program's answer on standard output: {name: bytes}."""
    if not out.startswith(_MAGIC):
        raise BundledSExtractorError("the SExtractor program wrote no answer on standard output")
    view, pos, files = memoryview(out), len(_MAGIC), {}
    while True:
        nl = out.find(b"\n", pos)
        if nl < 0:
            raise BundledSExtractorError("the SExtractor program's answer breaks off")
        line = out[pos:nl].decode()
        pos = nl + 1
        if line == "END":
            return files
        name, size = line.rsplit(" ", 1)
        size = int(size)
        if pos + size > len(out):
            raise BundledSExtractorError(f"the SExtractor program's answer breaks off in {name}")
        files[name] = bytes(view[pos:pos + size])
        pos += size


class BoundExtractor:
    """A tool for sfft.spaceflow.prep.sparse.catalog.image_products that runs `tool` on one in-memory image whatever path it is given.
    image_products opens neither the path nor any file when it also gets image= and identity= and no cache folder: the path then only names
    the products. Bind the array the executable path writes to its temporary FITS file (fitmask.sparse_products: float32, (y, x), missing
    pixels 0)."""

    def __init__(self, tool, image):
        self.tool, self.image = tool, image

    def run(self, path=None, ext=0, **kw):
        return self.tool.run(image=self.image, **kw)


class BundledSExtractor:
    """SExtractor 2.28.0 as betasfft's separate program, run once per call with its files passed through pipes. program: the program to run
    (default: the one built in _sextractor/build); timeout: seconds one call may take."""
    CANDIDATES = ()
    NAME = "SExtractor (bundled program)"

    def __init__(self, program=None, timeout=900):
        self.program = os.path.abspath(program or PROGRAM)
        if not os.path.exists(self.program):
            raise BundledSExtractorError(f"betasfft's SExtractor program is not built (looked for {self.program}). Build it with:\n"
                                         f"    {build_command()}")
        info = program_info(self.program)
        self.source_hash = source_hash()
        if info.get("sources") != self.source_hash:
            raise BundledSExtractorError(f"{self.program} was built from other sources (hash {info.get('sources')}, the package's "
                                         f"{self.source_hash}). Build it again with:\n    {build_command()}")
        self.version = info.get("SExtractor", VERSION)
        self.build_flags = info.get("flags")
        self.binary = self.program                      # the attribute SFFT's wrapper has, for logs and records
        self.timeout = timeout

    def info(self):
        return dict(engine="bundled program", version=self.version, program=self.program, source_hash=self.source_hash,
                    build_flags=self.build_flags)

    def _call(self, argv, inputs):
        """Runs the program with the input files as frames on standard input: (standard output, standard error, exit status)."""
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if sys.platform.startswith("linux"):
            import fcntl
            for stream in (proc.stdin, proc.stdout):
                try:
                    fcntl.fcntl(stream.fileno(), 1031, 1 << 20)          # F_SETPIPE_SZ: 1 MiB pipes, fewer switches for the large images
                except OSError:
                    pass
        got = {}

        def write():
            try:
                proc.stdin.write(_MAGIC)
                for name, data in inputs.items():
                    proc.stdin.write(f"{name} {len(data)}\n".encode())
                    proc.stdin.write(data)
                proc.stdin.write(b"END\n")
            except (BrokenPipeError, OSError):
                pass                                    # the program stopped early: its exit status and standard error say why
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        def read(stream, key):
            got[key] = stream.read()

        threads = [threading.Thread(target=write, daemon=True), threading.Thread(target=read, args=(proc.stdout, "out"), daemon=True),
                   threading.Thread(target=read, args=(proc.stderr, "err"), daemon=True)]
        for t in threads:
            t.start()
        try:
            rc = proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise SExtractorError(f"SExtractor did not finish within {self.timeout} s")
        finally:
            for t in threads:
                t.join(timeout=60)
        return got.get("out", b""), got.get("err", b"").decode(errors="replace"), rc

    def run(self, path=None, image=None, header=None, ext=0, config=None, columns=None, checkimages=(), keep_workspace=False, weight=None):
        """SExtractor on an in-memory image (or a FITS path): the arguments and the result of SFFT 2.0's SExtractor.run. keep_workspace is
        ignored (there is none). The result also carries warnings (SExtractor's warning messages, one per line) and timing [s]."""
        from astropy.io import fits
        from astropy.table import Table
        from .sextractor import CONV_DEFAULT, DEFAULT_COLUMNS, SExConfig, SExResult
        config = SExConfig() if config is None else config
        columns = DEFAULT_COLUMNS if columns is None else columns
        if (path is None) == (image is None):
            raise ValueError("give exactly one of path= or image=")
        map_types = ("MAP_VAR", "MAP_RMS", "MAP_WEIGHT")
        if config.weight_type in map_types and weight is None:
            raise ValueError(f"weight_type={config.weight_type!r} needs run(weight=...)")
        if weight is not None and config.weight_type not in map_types:
            raise ValueError(f"weight= given but config.weight_type is {config.weight_type!r} -- set it to one of {map_types}")
        columns = tuple(dict.fromkeys(tuple(columns) + ("X_IMAGE", "Y_IMAGE")))
        t0 = time.time()
        inputs = {}
        if image is not None:
            inputs[PREFIX + "image"] = _fits_bytes(np.asarray(image), header)
            target = PREFIX + "image"
        else:
            target = os.fspath(path) if ext == 0 else "%s[%d]" % (os.fspath(path), ext)
        lines = ["%-18s %s" % kv for kv in config.config_pairs()]
        if weight is not None:
            if isinstance(weight, (str, os.PathLike)):
                wname = os.fspath(weight)
            else:
                w = np.asarray(weight)
                if w.dtype == bool:
                    w = w.astype(np.float32)
                wname = PREFIX + "weight"
                inputs[wname] = _fits_bytes(w)          # the caller's dtype: the wrapper's float64 holds the same values once read as float
            lines.append("%-18s %s" % ("WEIGHT_IMAGE", wname))
        inputs[PREFIX + "param"] = "\n".join(columns).encode()
        lines.append("%-18s %s" % ("PARAMETERS_NAME", PREFIX + "param"))
        if config.use_filter:
            inputs[PREFIX + "conv"] = CONV_DEFAULT.encode()
            lines.append("%-18s %s" % ("FILTER_NAME", PREFIX + "conv"))
        lines.append("%-18s %s" % ("CATALOG_NAME", PREFIX + "catalog"))
        lines.append("%-18s %s" % ("CATALOG_TYPE", "FITS_LDAC"))
        names = {c: PREFIX + "check_" + c.lower() for c in checkimages}
        if checkimages:
            lines.append("%-18s %s" % ("CHECKIMAGE_TYPE", ",".join(checkimages)))
            lines.append("%-18s %s" % ("CHECKIMAGE_NAME", ",".join(names[c] for c in checkimages)))
        else:
            lines.append("%-18s %s" % ("CHECKIMAGE_TYPE", "NONE"))
        inputs[PREFIX + "config"] = ("\n".join(lines) + "\n").encode()
        argv = [self.program, "--frames", target, "-c", PREFIX + "config"]
        t_prep = time.time()
        out, err, rc = self._call(argv, inputs)
        t_run = time.time()
        if rc != 0:
            tail = "\n".join(err.strip().splitlines()[-8:])
            raise SExtractorError(f"SExtractor stopped with status {rc}:\n{tail}" if tail else f"SExtractor stopped with status {rc}")
        outputs = _parse_frames(out)
        cat_bytes = outputs.get(PREFIX + "catalog")
        if not cat_bytes:
            raise SExtractorError("SExtractor wrote no catalogue")
        catalog = Table.read(io.BytesIO(cat_bytes), format="fits", hdu=2)
        checks = {}
        for c in checkimages:
            if not outputs.get(names[c]):
                raise SExtractorError(f"SExtractor wrote no {c} check image")
            checks[c] = fits.getdata(io.BytesIO(outputs[names[c]]), ext=0)
        t_done = time.time()
        res = SExResult(catalog=catalog, checkimages=checks, config=config, columns=columns,
                        command=" ".join(argv[:1] + argv[2:]), runtime=t_done - t0)
        res.warnings = outputs.get("sexlib_warnings", b"").decode(errors="replace")
        # where the time went [s]: building the inputs, the program's run (process start, pipes, SExtractor), reading the answer back
        res.timing = dict(prepare=t_prep - t0, program=t_run - t_prep, read_back=t_done - t_run)
        return res
