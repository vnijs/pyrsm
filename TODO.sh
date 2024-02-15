## set up for testing without needing to build!!!

## in a jupyter notebook setup pyrsm fo
## autoreload when you edit code and save
## code in the pyrsm repo
## nothing else needed!
%reload_ext autoreload
%autoreload 2
%aimport pyrsm

## check version and location of pyrsm
python -c "import pyrsm as rsm; print(rsm.__version__); print(rsm.__file__)"
pip install --user "pyrsm>=0.9.12"

## select commands to run and use the Command Palette to send to open terminal
# use python build to install locally testing
conda activate msba
conda activate pyrsm
pip uninstall -y pyrsm
# pip uninstall -y pyrsm
# sudo rm -rf ~/gh/pyrsm/dist
rm -rf ~/gh/pyrsm/dist
# sudo rm -rf ~/gh/pyrsm/build/*
# pip install -q build
python -m build ~/gh/pyrsm

# try sending to pypi testing ground first
# pip install -q twine
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*

# if all goes well push to main pypi
python -m twine upload --repository pypi dist/*

# see API keys in ~/.pypirc
# create here: https://testpypi.org/manage/account/token/
# create here: https://pypi.org/manage/account/token/

## now install in "editable" mode
conda activate pyrsm
pip uninstall -y pyrsm
rm -rf ~/gh/pyrsm/dist
pip install --user -e ~/gh/pyrsm


conda activate base
pip uninstall --user pyrsm
pip install --user -e ~/gh/pyrsm

# poetry
poetry env list

# create a new
sudo rm -rf ~/testenv
conda activate pyrsm
python -m venv ~/testenv
conda deactivate
source ~/testenv/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "pyrsm==0.9.12"
pip install "pyrsm>=0.9.14"
python -c "import pyrsm as rsm"
python -c "import pyrsm as rsm; rsm.radiant.radiant()"
python -c "import pyrsm as rsm; rsm.radiant.model.regress()"
python -c "import pyrsm as rsm; rsm.radiant.model.logistic()"
python -c "import pyrsm as rsm; rsm.radiant.basics.compare_means()"
python -c "import pyrsm asrsm; rsm.radiant.goodness()"
python -c "import pyrsm as rsm; rsm.radiant.cross_tabs()"
deactivate

## assuming you are using conda
## remove all version pyrsm you might be using
conda activate base
sudo pip uninstall -y pyrsm
sudo pip uninstall --user -y pyrsm
conda remove -y --force pyrsm

## now install in "editable" mode
sudo pip install -e ~/gh/pyrsm

## in a jupyter notebook setup pyrsm for 
## autoreload when you edit code and save
## code in the pyrsm repo
## nothing else needed!
%reload_ext autoreload
%autoreload 2
%aimport pyrsm

usethis::usecourse("https://www.dropbox.com/sh/qn6jf7qiek7aicm/AAD9FA5vQxq6Hkkk5EfDLxvaa?dl=1")

## select commands to run and use the Command Palette to send to open terminal

# use python build to install locally testing
conda deactivate
sudo pip3 uninstall -y pyrsm
sudo rm -rf ~/gh/pyrsm/dist
# pip3 install -q build
sudo python3 -m build
sudo pip3 install dist/pyrsm-*.tar.gz

python3 -c "import pyrsm; print(pyrsm.__version__)"
python3 -c "import pyrsm; print(pyrsm.__file__)"

# use pip to add to base (or other) environment
conda activate base
sudo pip uninstall
sudo rm -rf ~/gh/pyrsm/dist
sudo python3 -m build
sudo pip install dist/pyrsm-*.tar.gz

## might be useful when testing
conda remove -y --force pyrsm # remove current version
conda install -c conda-forge pyrsm 

# get the sha256 code on the built tar.gz file **before**
# building the conda version. You can get this from the version 
# built for pip
# add to meta.yaml
openssl sha256 dist/pyrsm-*.tar.gz

# Weird and annoying "Could not find a version that satisfies the requirement python>=3.6"
##### removed some references to python and python>=3.6 -- lets see if that helps #####
# conda activate base
# sudo pip3 install dist/pyrsm-*.tar.gz

# from https://github.com/vnijs/pypi-howto
# sudo pip3 install --upgrade twine keyring

# https://kynan.github.io/blog/2020/05/23/how-to-upload-your-package-to-the-python-package-index-pypi-test-server
# tokens are in ~/.pypirc

# try sending to pypi testing ground first
python3 -m twine check dist/*
python3 -m twine upload --repository testpypi dist/*

# if all goes well push to main pypi
python3 -m twine upload dist/*

# use conda for local 

# create a conda environment for testing
# cc pyrsm-dev pyrsm

# Use the conda directory in the pyrsm repo that contains meta.yaml.
# change the source to point to the local directory you are working
# in. Then in the repo directory issue `conda build .`
# do conda install -y conda-build first). This creates a .tar.bz2
# build file (and the output shows you the location of the same)
# and this local build of pyrsm can be installed using
# conda install -y /path/to/build/file



##
## For some reason you have to increment the version number in __init__.py
## and meta.yaml for changes to get picked up
##

conda activate pyrsm-dev
# conda install -y conda-build # only need this once
conda remove -y --force pyrsm # remove current version
rm -rf /opt/conda/conda-bld/broken/pyrsm*
rm -rf /opt/conda/conda-bld/pyrsm*

# get the sha256 code on the built tar.gz file
openssl sha256 dist/pyrsm-*.tar.gz
# add the sha256 sequence to conda/meta.yaml file **before** building
conda build ~/gh/pyrsm/conda/pyrsm
conda install /opt/conda/conda-bld/pyrsm*

# if this fails on the last step use the below
conda install /opt/conda/conda-bld/broken/pyrsm*

# adding to base environment
# note: need to change the sha256 code in the meta.yaml file 
# download the tar.gz file after login in to pypi and looking at the 
# releases then use "openssl sha256 ~/Downloads/pyrsm-0.6.3.tar.gz" 
# or similar and add the code to meta.yaml in the conda directory
conda activate base
# conda install -c conda-forge pyrsm
conda remove -y --force pyrsm # remove current version
rm -rf /opt/conda/conda-bld/broken/pyrsm*
rm -rf /opt/conda/conda-bld/pyrsm*
rm -rf /opt/conda/conda-bld/noarch/pyrsm*

# get the sha256 code on the built tar.gz file **before**
# building the conda version. You can get this from the version 
# built for pip
openssl sha256 dist/pyrsm-*.tar.gz

# add the sha256 sequence to conda/meta.yaml file **before** building (huh?)
conda install conda-build
conda build ~/gh/pyrsm/conda/pyrsm

# try the below, might work but
# conda install /opt/conda/conda-bld/pyrsm*

# if this fails on the last step use the below
# conda install /opt/conda/conda-bld/broken/pyrsm*

# check the pyrsm version number in
python -c "import pyrsm; print(pyrsm.__version__)"
python -c "import pyrsm; print(pyrsm.__file__)"

# if the builds above completed without issues use the below
# to upload to the vnijs user account on anaconda
# (check passwd manager as needed)
anaconda upload /opt/conda/conda-bld/noarch/pyrsm*

# else use the below to upload to the vnijs user
# account on anaconda (check passwd manager as needed)
# anaconda upload /opt/conda/conda-bld/broken/pyrsm*


## not sure if still needed

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
rm -rf /opt/conda/conda-bld/broken/pyrsm*
rm -rf /opt/conda/conda-bld/pyrsm*
conda build --skip-existing pyrsm/
conda install --use-local pyrsm
conda build purge
cp pyrsm/meta.yaml ~/gh/conda-packages/recipes/pyrsm/meta.yaml
code ~/gh/conda-packages ## need to make manual edits for python >= 3.6 and license

# Adding packages to conda-forge: https://conda-forge.org/docs/maintainer/adding_pkgs.html
# PR created @ https://github.com/conda-forge/staged-recipes/pull/18174

# adding package to personal channel @ https://anaconda.org/vnijs/pyrsm
conda install anaconda-client
anaconda login
# conda build .
#anaconda upload ~/miniconda3/conda-bld/noarch/pyrsm-0.5.8-py_0.tar.bz2
#anaconda upload ~/miniconda3/conda-bld/noarch/pyrsm-*

# if the builds above completed without issues use the below
anaconda upload /opt/conda/conda-bld/pyrsm*

# else use the below
anaconda upload /opt/conda/conda-bld/broken/pyrsm*

#conda install /opt/conda/conda-bld/pyrsm*
# if this fails on the last step use the below
#conda install /opt/conda/conda-bld/broken/pyrsm*


