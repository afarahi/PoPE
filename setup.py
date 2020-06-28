import os.path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import warnings


# numpy path is needed
try:
    import numpy
    numpy_includes = [numpy.get_include()]
    HAVE_NUMPY = True
except ImportError:
    # "python setup.py build" will not work and trigger fallback to pure python later on,
    # but "python setup.py clean" will be successful with the first call of setup(...)
    numpy_includes = []
    HAVE_NUMPY = False

classifiers = [
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Astronomy',
    'Topic :: Scientific/Engineering :: Physics'
]

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'readme.md')) as f:
    long_description = f.read()

kwargs = {
    'name': 'PoPE',
    'version': '1.0.2',
    'author': 'Arya Farahi, Daisuke Nagai, Yang Chen',
    'author_email': 'aryaf@umich.edu',
    'packages': ['PoPE'],
    'url': 'https://github.com/afarahi/PoPE',
    'description': 'Population Profile Estimator.',
    'long_description': long_description,
    'long_description_content_type':'text/markdown',
    'url':'https://github.com/afarahi/PoPE',
    'license': 'MIT',
    'keywords': ['linear', 'regression', 'astronomy', 'astrophysics', 'parameter estimation'],
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn', 'theano', 'pymc3', 'KLLR'],
    'packages': find_packages(),
    'test_suite': 'tests',
    'setup_requires': ['pytest-runner'],
    'tests_require': ['pytest'],
    'classifiers': classifiers
}

try:
    setup(**kwargs)
except SystemExit:
    del kwargs['ext_modules']
    reason = 'numpy missing, ' if not HAVE_NUMPY else ''
    warnings.warn(reason+'compilation failed. Installing pure python package')
    setup(**kwargs)