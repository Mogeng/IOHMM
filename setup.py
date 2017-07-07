#!/usr/bin/env python

# Copyright 2017 Mogeng Yin | https://www.apache.org/licenses/LICENSE-2.0

from setuptools import setup

setup(
    name="IOHMM",
    version="0.0.0",
    description='Input Output Hidden Markov Models',
    author='Mogeng Yin',
    author_email='mogengyin@berkeley.edu',
    packages=['IOHMM'],
    install_requires=[
        'numpy>=1.11.0',
        'pandas>=0.19.0',
        'scikit-learn>=0.18.0',
        'scipy>=0.19.0',
        'statsmodels>=0.8.0'
    ],
    extras_require={
        'tests': [
            'flake8>=2.5.4',
            'mock>=2.0.0',
            'nose>=1.3.4',
            'coveralls>=1.1',
            'pytest',
        ]
    },
    zip_safe=True,
    keywords='sequence learning',
)
