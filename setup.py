import setuptools
from setuptools import find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "sfdmap",
    "PyAstronomy",
    "uncertainties"]

setuptools.setup(
    name="qsofitmore", 
    version="1.2.0-beta",
    author="Yuming Fu",
    author_email="fuympku@outlook.com",
    description="A python package for fitting UV-optical spectra of quasars. Currently a wrapper of PyQSOFit for customization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rudolffu/qsofitmore",
    packages=find_packages()+find_packages("qsofitmore/PyQSOFit/"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE V3",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    # extras_require={
    #     'dev': [
    #         f"PyQSOFit @ file://localhost{os.getcwd()}/qsofitmore/PyQSOFit/"
    #     ]},
    python_requires='>=3.6',
)
