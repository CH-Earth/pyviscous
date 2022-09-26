#!/usr/bin/env python

"""The setup script."""

try:
    from setuptools import setup
except:
    from distutils.core import setup
    
from setuptools import find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

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
    description="Python codes you will need to implement the VISCOUS global sensitivity analysis framework",
    entry_points={
        'console_scripts': [
            'pyviscous=pyviscous.cli:main',
        ],
    },
    install_requires=['xarray', 'numpy', 'pandas', 'scipy',
                      'matplotlib', 'sklearn','jupyter'],
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyviscous',
    name='pyviscous',
    packages=find_packages(include=['pyviscous', 'pyviscous.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/h294liu/pyviscous',
    version='0.1.0',
    zip_safe=False,
)


