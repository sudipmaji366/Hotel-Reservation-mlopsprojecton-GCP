from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(

    name="MLOPS-1",
    version="0.1.0",
    author="Sudip",
    packages=find_packages(),
    install_requires=requirements,
)
