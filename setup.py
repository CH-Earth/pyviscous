#!/usr/bin/env python

"""The setup script."""

try:
    from setuptools import setup
except:
    from distutils.core import setup

from setuptools import find_packages

requirements = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'copulae', 'matplotlib', 'jupyter']
# Important note: if users install the above required packages by themselves, please install copulae via pip, not conda. 
# This is because the conda distribution of copulae does not properly include its full source code/functions. 
# We will remind the developer of copulae to fix this.

test_requirements = [ ]

# read the contents of your README file for distribution on PyPI
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    author="Hongli Liu",
    author_email='hongliliu68@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python codes to implement the VISCOUSm global sensitivity analysis framework",
    entry_points={
        'console_scripts': [
            'pyviscous=pyviscous.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    include_package_data=True,
    keywords='pyviscous',
    name='pyviscous',
    packages=find_packages(include=['pyviscous', 'pyviscous.*', 'plot.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/CH-Earth/pyviscous',
    version='2.2.1',
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown',
)


