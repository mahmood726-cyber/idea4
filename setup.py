"""
Setup script for BART Meta-Regression package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bart-meta-regression",
    version="1.0.0",
    author="BART Meta-Regression Contributors",
    author_email="your.email@institution.edu",
    description="Advanced Bayesian Additive Regression Trees for Meta-Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahmood726-cyber/idea4",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    keywords="meta-analysis meta-regression BART Bayesian machine-learning statistics",
    project_urls={
        "Bug Reports": "https://github.com/mahmood726-cyber/idea4/issues",
        "Source": "https://github.com/mahmood726-cyber/idea4",
    },
)
