from setuptools import setup, find_packages

setup(
    name='gymrl',
    version='0.0.1',
    packages=find_packages(exclude=['tests', 'old'])
)
