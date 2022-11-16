from mlexp import __version__
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8-sig") as f:
    required = f.read().splitlines()

setup(
    name="mlexp",
    version=__version__,
    description="MLexp allows to train different types of machine learning models, \
                 optimize hyperparameters and log results with simple API",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Ilia Avilov",
    author_email="ieavilov@gmail.com",
    license="MIT",
    keywords=["machine learning"],
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.9",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=required,
    project_urls={"GitHub": "https://github.com/volico/mlexp"},
)
