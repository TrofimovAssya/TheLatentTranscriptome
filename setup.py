#!/usr/bin/env python
import os
from setuptools import setup

setup(
    name='factembseq',
    description='Simple research project example.',
    version='0.0.0',
    author='Mila',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'factembseq = factembseq.main:main'
        ]
    }

)
