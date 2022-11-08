import os
from setuptools import setup
from setuptools import find_packages

import numpy as np
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules = ['ibug/parsers/_tree32.pyx',
               'ibug/parsers/_tree64.pyx']

libraries = []
if os.name == 'posix':
    libraries.append('m')

setup(name="ibug",
    version="0.0.7",
    description="Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees",
    author="Jonathan Brophy",
    author_email="jonathanbrophy47@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jjbrophy47/ibug",
    packages=find_packages(),
    include_package_data=True,
    package_dir={"ibug": "ibug"},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"],
    python_requires='>=3.9',
    install_requires=[
        "numpy>=1.22",
        "uncertainty-toolbox>=0.1.0",
        "joblib>=1.1.0",
        "scikit-learn>=1.1.1",
        "scipy>=1.8.1",
        "pandas>=1.4.3",
        "Cython>=0.29.23",
        "xgboost>=1.6.1"
    ],
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': 3}, annotate=True),
    include_dirs=np.get_include(),
    zip_safe=False
)
