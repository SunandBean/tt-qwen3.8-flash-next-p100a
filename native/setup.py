"""Builds lib/{v3,v4}/ with build.sh before packaging (git, cmake and a C++ compiler with OpenMP are needed)."""
import pathlib
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = pathlib.Path(__file__).resolve().parent


class BuildNative(build_py):
    def run(self):
        if not (HERE / "tt_qwen38_native" / "lib" / "v3" / "libhostexp.so").exists():
            subprocess.run(["bash", str(HERE / "build.sh")], check=True, cwd=HERE)
        super().run()


setup(cmdclass={"build_py": BuildNative})
