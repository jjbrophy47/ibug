import os

import numpy
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    config = Configuration('', parent_name=parent_package, top_path=top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension("_tree32",
                         sources=["_tree32.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])

    config.add_extension("_tree64",
                         sources=["_tree64.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])

    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives={'language_level': 3},
        annotate=True
    )

    return config


if __name__ == "__main__":
    setup(**configuration(top_path='').todict())
