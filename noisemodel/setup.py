from setuptools import setup, find_packages

setup(
    name='windturbinenoise',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'joblib',
    ],
    author='MB',
    description='Allows to predict sound power level from a given wind turbine and to simulate the sound emission',
)