from setuptools import setup, find_packages
from mlexp import __version__

with open("requirements.txt", encoding="utf-8-sig") as f:
    required = f.read().splitlines()

setup(
    name='mlexp',
    version=__version__,
    description='mlexp allows to train different types of machine learning models, \
                 optimize hyperparameters and log results with simple API',
    author='Ilia Avilov',
    author_email='ieavilov@yandex.ru',
    license='MIT',
    keywords=['logging', 'hyperparameters', 'training', 'models'],
    packages=find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=required
)
