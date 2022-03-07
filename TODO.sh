## select commands to run and use the Command Palette to send to open terminal

# use build to install locally
sudo pip3 uninstall -y pyrsm
sudo rm -rf ~/gh/pyrsm/dist
pip3 install -q build
sudo python3 -m build
pip3 install dist/pyrsm-*.tar.gz

# from https://github.com/vnijs/pypi-howto
# sudo pip3 install --upgrade twine keyring

# https://kynan.github.io/blog/2020/05/23/how-to-upload-your-package-to-the-python-package-index-pypi-test-server
# tokens are in ~/.pypirc

# try sending to pypi testing ground first
python3 -m twine check dist/*
python3 -m twine upload --repository testpypi dist/*

# if all goes well push to main pypi
python3 -m twine upload dist/*

# setting up for conda

# conda config --add channels conda-forge
# conda install -c conda-forge ipynbname
# conda install -c conda-forge importlib_resources
# conda install -c conda-forge grayskull

# grayskull seems to work a lot better
# conda skeleton pypi --extra-specs numpy \
#   --extra-specs pandas \
#   --extra-specs matplotlib \
#   --extra-specs seaborn \
#   --extra-specs statsmodels \
#   --extra-specs scipy \
#   --extra-specs ipython \
#   --extra-specs ipynbname \
#   --extra-specs scikit-learn \
#   --extra-specs importlib_resources \
#   pyrsm --version 0.5.2 --python-version 3.6

cd conda
grayskull pypi pyrsm
conda build pyrsm/
conda install --use-local pyrsm
conda build purge
cp pyrsm/meta.yaml ~/gh/conda-packages/recipes/pyrsm/meta.yaml
code ~/gh/conda-packages ## need to make manual edits for python >= 3.6 and license

# Adding packages to conda-forge: https://conda-forge.org/docs/maintainer/adding_pkgs.html
# PR created @ https://github.com/conda-forge/staged-recipes/pull/18174

# adding package to personal channel @ https://anaconda.org/vnijs/pyrsm
# conda install anaconda-client
# anaconda login
conda build .
#anaconda upload ~/miniconda3/conda-bld/noarch/pyrsm-0.5.8-py_0.tar.bz2
anaconda upload ~/miniconda3/conda-bld/noarch/pyrsm-*
