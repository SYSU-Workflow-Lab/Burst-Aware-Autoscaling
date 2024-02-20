# https://stackoverflow.com/questions/61532337/python-modulenotfounderror-no-module-named
from setuptools import find_packages, setup

setup(
  name='foo',
  version='1.0.0',
  packages=find_packages(),
  entrypoints={
    'console_scripts': [
      'foo=simulator.main:main',
    ],
  },
)