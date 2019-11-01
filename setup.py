from setuptools import setup, find_packages

setup(
    name="pyrsm",
    version="0.1.1",
    description="Python functions for Customer Analytics at the Rady School of Management (RSM)",
    long_description="Python functions for Customer Analytics at the Rady School of Management (RSM)",
    long_description_content_type="text/markdown",
    license="AGPL",
    author="Vincent Nijs",
    author_email="vnijs@ucsd.edu",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=["numpy>=1.17.3"],
    project_urls={
        "Bug Reports": "https://github.com/vnijs/pyrsm/issues",
        "Source": "https://github.com/vnijs/pyrsm",
    },
)