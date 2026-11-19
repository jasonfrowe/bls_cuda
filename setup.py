"""Build the optional C/OpenMP BLS kernel (pytfit5._cbls).

Project metadata lives in pyproject.toml. The extension is optional: if it
fails to compile, pytfit5 still installs and bls() uses the Numba kernels.
Set PYTFIT5_NO_CBLS=1 to skip building it.
"""
import os
import sys

from setuptools import Extension, setup

ext_modules = []
if not os.environ.get("PYTFIT5_NO_CBLS"):
    if sys.platform == "win32":
        cflags, ldflags = ["/O2", "/openmp"], []
    elif sys.platform == "darwin":
        cflags, ldflags = ["-O3"], []  # Apple clang has no OpenMP by default
    else:
        cflags, ldflags = ["-O3", "-fopenmp"], ["-fopenmp"]
    ext_modules.append(
        Extension(
            "pytfit5._cbls",
            sources=["pytfit5/_cbls.c"],
            define_macros=[("CBLS_PYMODULE", "1")],
            extra_compile_args=cflags,
            extra_link_args=ldflags,
            optional=True,
        )
    )

setup(ext_modules=ext_modules)
