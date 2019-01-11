#!/usr/bin/env python
from setuptools import setup

setup(
    name='latenttranscriptome',
    description='The Latent Transcriptome project.',
    version='0.0.0',
    author='Mila',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'latenttranscriptome = latenttranscriptome.main:main'
        ]
    }

)
