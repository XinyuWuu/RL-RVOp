# export LD_LIBRARY_PATH="mujoco/lib;glfw/src"
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
from mypy import stubgen
import os
os.environ['CPPFLAGS'] = '-Imujoco/include -Iglfw/include'
os.environ['LDFLAGS'] = '-Lmujoco/lib -Lglfw/src -lmujoco -lglfw'
# from Cython.Build import cythonize

# setup(name="PyRVOcalculator",
#       ext_modules=cythonize("PyRVOcalculator.pyx",
#                             language_level=3))
source_list = [
    "simulator.cpp",
    "environment.cpp",
    "pybind11_bind.cpp",
    "funcsdef.cpp",
    "RVOcalculator.cpp",
    "observator.cpp",
    "myreward.cpp",
    "ctrlConverter.cpp",
]

extra_objects = [
    r'mujoco/lib/libmujoco.so',
    r'glfw/src/libglfw.so',
]

ext_modules = [
    Pybind11Extension(
        "Environment",
        source_list,
        extra_objects=extra_objects,
    ),
    # Pybind11Extension(
    #     "Simulator",
    #     source_list,
    #     extra_objects=extra_objects,
    # ),
    # Pybind11Extension(
    #     "Observator",
    #     source_list,
    #     extra_objects=extra_objects,
    # ),
    # Pybind11Extension(
    #     "RVOcalculator",
    #     source_list,
    #     extra_objects=extra_objects,
    # ),
    # Pybind11Extension(
    #     "CtrlConverter",
    #     source_list,
    #     extra_objects=extra_objects,
    # ),
]

setup(name="CppClass", ext_modules=ext_modules)
print("___________________stubgen___________________")
stubgen.generate_stubs(stubgen.parse_options(
    ["-p" "Simulator", "-o", "../stub/CppClass/"]))
stubgen.generate_stubs(stubgen.parse_options(
    ["-p" "RVOcalculator", "-o", "../stub/CppClass/"]))
stubgen.generate_stubs(stubgen.parse_options(
    ["-p" "Observator", "-o", "../stub/CppClass/"]))
stubgen.generate_stubs(stubgen.parse_options(
    ["-p" "CtrlConverter", "-o", "../stub/CppClass/"]))
stubgen.generate_stubs(stubgen.parse_options(
    ["-p" "Environment", "-o", "../stub/CppClass/"]))
