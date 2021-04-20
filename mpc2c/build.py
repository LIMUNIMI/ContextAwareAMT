from glob import glob

from Cython.Build import Cythonize

from . import settings as s


def build():

    if s.BUILD:
        paths = set(glob("mpc2c/**.py"))
        paths -= set(glob("mpc2c/__init__.py"))
        paths -= set(glob("mpc2c/build.py"))
        paths -= set(glob("mpc2c/settings.py"))
        for path in paths:
            Cythonize.main([path, "-3", "--inplace"])
