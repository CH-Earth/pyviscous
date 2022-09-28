#!/usr/bin/env python

"""The setup script."""

try:
    from setuptools import setup
except:
    from distutils.core import setup
    
from setuptools import find_packages

requirements = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'jupyter']

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
    description="Python codes to implement the VISCOUS global sensitivity analysis framework",
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
    packages=find_packages(include=['pyviscous', 'pyviscous.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/h294liu/pyviscous',
    version='0.1.1',
    zip_safe=False,
)


