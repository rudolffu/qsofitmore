import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qsofitmore", 
    version="1.1.1",
    author="Yuming Fu",
    author_email="fuympku@outlook.com",
    description="A wrapper of PyQSOFit for customization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rudolffu/qsofitmore",
    packages=setuptools.find_packages(),
    # package_data={'qsofitmore': ['data/pca/*.fits']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE V3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
