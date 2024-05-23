import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools import setup

extensions = [Extension("src.sampling.sampling", ["src/sampling/sampling.pyx"])]

setup(
    name="src",
    version="0.0.6",
    url="",
    license="MIT",
    author="Caspar Roebroek",
    author_email="roebroek.jasper@gmail.com",
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[numpy.get_include()],
    setup_requires=["setuptools>=61.0", "numpy", "Cython"],
)
