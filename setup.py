from setuptools import setup, find_packages

setup(
    name='nueralnet',
    version='0.0.1',
    description='compute neural network',
    author='Kengo Matsuura',
    author_email='kengo11141996(at)gmail.com',  # convert (at) to @
    url='https://github.com/kmatsuuraHMC/neural_network',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
