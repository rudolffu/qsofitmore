import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyQSOFit", 
    version="1.0",
    author="Hengxiao Guo, Shu Wang, and Yue Shen",
    author_email="hengxiaoguo@gmail.com",
    description="A code to fit the spectrum of quasar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/legolason/PyQSOFit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE V3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)