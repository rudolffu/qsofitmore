import setuptools
from setuptools import find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "uncertainties",
    "astropy>=5.0"]

setuptools.setup(
    name="qsofitmore", 
    version="1.2.2",
    author="Yuming Fu",
    author_email="fuympku@outlook.com",
    description="A python package for fitting UV-optical spectra of quasars.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rudolffu/qsofitmore",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE V3",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires='>=3.8',
)
