"""
  Utility package for MLHEP-2019 summer school.
"""

from setuptools import setup

from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(
  name='mlhep2019',

  version='0.0.0',

  description="""Utility package for MLHEP-2019 summer school.""",

  long_description=long_description,

  url='https://github.com/yandexdataschool/mlhep2019/',

  author='Andrey Ustyuzhanin',
  author_email='andrey.u at gmail dot com',

  maintainer='Andrey Ustyuzhanin',
  maintainer_email='andrey.u at gmail dot com',

  license='MIT',

  classifiers=[
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],

  keywords='Machine Learning',

  packages=['mlhep2019'],

  install_requires=[
    'tqdm',
    'matplotlib',
    'numpy',
    'torch',
    'torchvision',
  ]
)
