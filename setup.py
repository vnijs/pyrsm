from setuptools import setup, find_packages
from pyrsm import __version__

setup(
    name="pyrsm",
    version=__version__,
    description="Python functions for Customer Analytics at the Rady School of Management (RSM)",
    long_description="Python functions for Customer Analytics at the Rady School of Management (RSM)",
    long_description_content_type="text/markdown",
    license="AGPL",
    author="Vincent Nijs",
    author_email="vnijs@ucsd.edu",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "numpy>=1.17.3",
        "pandas>=0.25.2",
        "seaborn>=0.9.0",
        "matplotlib>=3.1.1",
        "statsmodels>=0.10.1",
        "scipy>=1.4.1",
    ],
    project_urls={
        "Bug Reports": "https://github.com/vnijs/pyrsm/issues",
        "Source": "https://github.com/vnijs/pyrsm",
    },
)
