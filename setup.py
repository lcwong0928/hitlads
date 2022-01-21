"""Setup file for pytest."""
from setuptools import setup
import src

setup(name=src.__name__,
      version=src.__version__,
      packages=['src'],
      test_suites='tests',
      python_requires='==3.7',
      )
