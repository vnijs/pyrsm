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

grayskull pypi pyrsm
conda build pyrsm/

Adding packages to conda-forge: https://conda-forge.org/docs/maintainer/adding_pkgs.html
PR created