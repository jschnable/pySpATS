#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pySpATS",
    version="0.2.0",
    author="James Schnable Lab",
    author_email="jschnable@unl.edu", 
    description="Python implementation of Spatial Analysis of Field Trials with Splines (SpATS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/schnablelab/python-spats",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=1.0.0",
    ],
    keywords="spatial analysis, field trials, splines, agriculture, plant breeding, mixed models",
    project_urls={
        "Bug Reports": "https://github.com/schnablelab/python-spats/issues",
        "Source": "https://github.com/schnablelab/python-spats",
    },
)