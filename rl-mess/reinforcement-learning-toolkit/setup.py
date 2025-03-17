from setuptools import setup, find_packages

setup(
    name='reinforcement-learning-toolkit',
    version='0.0.1',
    description='Reinforcement Learning',
    url='http://bitbucket.org/avilay/reinforcement-learning-toolkit',
    author='Avilay Parekh',
    author_email='avilay@avilaylabs.net',
    license='MIT',
    packages=find_packages(exclude=['rltk.tests']),
    install_requires=['numpy', 'haikunator']
)
