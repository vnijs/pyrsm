from setuptools import setup, find_packages
from pyrsm import __version__

setup(
    name="pyrsm",
    version=__version__,
    description="Python functions for Customer Analytics at the Rady School of Management (RSM)",
    long_description="Python functions for Customer Analytics at the Rady School of Management (RSM)",
    long_description_content_type="text/markdown",
    license="AGPL",
    author="Vincent Nijs <vnijs@ucsd.edu>, Vikram Jambulapati <vikjam@ucsd.edu>, Suhas Goutham <sgoutham@ucsd.edu>",
    author_email="vnijs@ucsd.edu",
    include_package_data=True,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "examples"]),
    install_requires=[
        "numpy>=1.17.3",
        "pandas>=0.25.2",
        "seaborn>=0.9.0",
        "matplotlib>=3.1.1",
        "statsmodels>=0.10.1",
        "scipy>=1.4.1",
        "ipynbname>=2021.3.2",
    ],
    project_urls={
        "Bug Reports": "https://github.com/vnijs/pyrsm/issues",
        "Source": "https://github.com/vnijs/pyrsm",
    },
#     package_data={
#         "data/data": ["*.pkl"],
#         "data/design": ["*.pkl"],
#         "data/basics": ["*.pkl"],
#         "data/model": ["*.pkl"],
#         "data/multivariate": ["*.pkl"],
#     }
)
