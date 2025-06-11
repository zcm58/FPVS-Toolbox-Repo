from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

this_dir = Path(__file__).resolve().parent

ext_modules = [
    Pybind11Extension(
        "downsample_filter",
        [this_dir / "downsample_filter.cpp"],
    ),
]

setup(
    name="downsample_filter",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
