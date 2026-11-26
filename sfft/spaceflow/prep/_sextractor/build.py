"""Builds betasfft's SExtractor program, sextractor228: SExtractor 2.28.0 (src/, as released) with the edits in patches.py, the runtime sexlib.c
and the main sexprog.c. Part of a modified version of SExtractor (GNU GPL v3 or later), see README.md.

    python build.py [--out <folder>] [--cc <compiler>]        default folder: build/ in this folder, where betasfft looks for the program

It needs a C compiler: on Linux gcc, on macOS the command-line tools (xcode-select --install); no configure script, CFITSIO, FFTW or BLAS.
The compiler is --cc, else $CC, else the first of gcc, cc and clang found; on a Cray system use the native gcc rather than the cc wrapper,
which adds a -march of its own. The flags are pinned for SExtractor's exact numbers: no fused multiply-add contraction, no fast-math, and
on x86-64 conda-forge's -march=nocona -mtune=haswell. The program embeds the SHA-1 of this folder's sources (python patches.py hash <folder>);
betasfft refuses a program whose hash differs from its sources, so build again after an update that changes this folder. The compiler runs inside
a temporary folder on relative paths, so the program's source-file names hold no folder of the machine that built it.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "sextractor228"
X86 = platform.machine().lower() in ("x86_64", "amd64")


def compiler(cc=None):
    cc = cc or os.environ.get("CC") or next((c for c in ("gcc", "cc", "clang") if shutil.which(c)), None)
    if not cc:
        sys.exit("no C compiler: on macOS run 'xcode-select --install', on Linux install gcc; then run this script again")
    return shutil.which(cc) or cc          # an absolute path: the compiler runs inside the build folder


def flags():
    return ["-O2", "-fPIC", "-fvisibility=hidden", "-ffp-contract=off", "-fno-fast-math"] + (["-march=nocona", "-mtune=haswell"] if X86 else [])


def run(cmd, cwd=None):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if p.returncode:
        raise RuntimeError(f"failed: {' '.join(cmd)}\n{p.stdout}{p.stderr}")
    return p.stdout


def check(exe, source_hash):
    """No fused multiply-add instructions on x86-64 (where objdump is installed); the program runs and reports its sources' hash."""
    if X86 and sys.platform.startswith("linux") and shutil.which("objdump"):
        dis = subprocess.run(["objdump", "-d", exe], capture_output=True, text=True).stdout
        n = sum(dis.count(op) for op in ("vfmadd", "vfmsub", "vfnmadd", "vfnmsub"))
        if n:
            raise RuntimeError(f"the program holds {n} fused multiply-add instructions")
    info = run([exe, "--sexlib-info"])
    if f"sources {source_hash}" not in info:
        raise RuntimeError(f"the program built does not report its sources' hash {source_hash}:\n{info}")


def build(out=None, cc=None):
    out = out or os.path.join(HERE, "build")
    cc, fl = compiler(cc), flags()
    patches = os.path.join(HERE, "patches.py")
    source_hash = run([sys.executable, patches, "hash", HERE]).strip()
    with tempfile.TemporaryDirectory(prefix="sextractor228_") as work:
        # Every path below is relative to work, where the compiler runs: __FILE__ and the symbols get "build/src/...", not this machine's folder.
        run([sys.executable, patches, "apply", HERE, os.path.join(work, "build")])
        os.makedirs(os.path.join(work, "obj"))
        inc = ["-Ibuild", "-Ibuild/src"]
        jobs = []
        for rel in open(os.path.join(work, "build", "SOURCES")).read().split():
            src = f"build/src/{rel}"
            jobs.append([cc, *fl, "-DHAVE_CONFIG_H", f"-I{os.path.dirname(src)}", *inc, "-c", src,
                         "-o", f"obj/{rel.replace('/', '_')[:-2]}.o"])
        jobs.append([cc, *fl, "-DHAVE_CONFIG_H", f'-DSEXLIB_SOURCE_HASH="{source_hash}"', f'-DSEXLIB_BUILD_FLAGS="{" ".join(fl)}"', *inc,
                     "-c", "build/sexlib.c", "-o", "obj/sexlib.o"])
        jobs.append([cc, *fl, "-c", "build/sexprog.c", "-o", "obj/sexprog.o"])
        print(f"=== compiling SExtractor 2.28.0 (program) with {cc}: {' '.join(fl)}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, min(16, os.cpu_count() or 1))) as ex:
            list(ex.map(lambda cmd: run(cmd, cwd=work), jobs))
        exe = os.path.join(work, NAME)
        run([cc, "-o", NAME, *sorted(f"obj/{o}" for o in os.listdir(os.path.join(work, "obj"))), "-lm"], cwd=work)
        check(exe, source_hash)
        os.makedirs(out, exist_ok=True)
        dst = os.path.join(out, NAME)
        shutil.copy2(exe, dst + ".part")
        os.replace(dst + ".part", dst)
    print(f"built {dst} (SExtractor 2.28.0, sources {source_hash})", flush=True)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="folder for the program (default: build/ in this folder)")
    ap.add_argument("--cc", help="the C compiler (default: $CC, else gcc, cc or clang)")
    a = ap.parse_args()
    try:
        build(a.out, a.cc)
    except RuntimeError as e:
        sys.exit(str(e))
