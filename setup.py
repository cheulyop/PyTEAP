import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyTEAP",
    version="0.1.1",
    author="Cheul Young Park",
    author_email="cheulyop@gmail.com",
    description="PyTEAP: A Python implementation of Toolbox for Emotion Analysis using Physiological signals (TEAP).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cheulyop/PyTEAP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)