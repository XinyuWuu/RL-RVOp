from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
from mypy import stubgen
# from Cython.Build import cythonize

# setup(name="PyRVOcalculator",
#       ext_modules=cythonize("PyRVOcalculator.pyx",
#                             language_level=3))
source_list = [
    "pybind11_bind.cpp",
    "funcsdef.cpp",
    "RVOcalculator.cpp",
    "observator.cpp",
    "myreward.cpp"
]
ext_modules = [
    Pybind11Extension(
        "Observator",
        source_list
    ),
    Pybind11Extension(
        "RVOcalculator",
        source_list
    ),
]

setup(name="CppClass", ext_modules=ext_modules)
print("___________________stubgen___________________")
stubgen.generate_stubs(stubgen.parse_options(
    ["-p" "RVOcalculator", "-o", "../stub/CppClass/"]))
stubgen.generate_stubs(stubgen.parse_options(
    ["-p" "Observator", "-o", "../stub/CppClass/"]))
