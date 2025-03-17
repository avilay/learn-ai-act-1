from setuptools import setup, find_packages

setup(
    name='rl-toolkit',
    version='0.0.1',
    description='Reinforcement Learning',
    url='https://gitlab.com/avilay/rl-toolkit/',
    author='Avilay Parekh',
    author_email='avilay@avilaylabs.net',
    license='MIT',
    packages=find_packages(exclude=['rltk.tests'])
)
