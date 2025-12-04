## set up for testing without needing to build!!!

## in a jupyter notebook setup pyrsm fo
## autoreload when you edit code and save
## code in the pyrsm repo
## nothing else needed!
%reload_ext autoreload
%autoreload 2
%aimport pyrsm

## check version and location of pyrsm
# python -c "import pyrsm as rsm; print(rsm.__version__); print(rsm.__file__)"
# pip install --user "pyrsm>=1.2.0"

## select commands to run and use the Command Palette to send to open terminal
# use python build to install locally testing
# conda activate msba
# conda activate pyrsm
uv remove pyrsm
# pip uninstall -y pyrsm
# sudo rm -rf ~/gh/pyrsm/dist
# sudo rm -rf ~/gh/pyrsm/build/*
# pip install -q build
# pip install -q twine
# python -m build ~/gh/pyrsm
# python -m build ~/gh/pyrsm > check.log 2>&1

# try sending to pypi testing ground first
# uv pip install twine pytest
rm -rf ~/gh/pyrsm/dist
uv build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*

# if all goes well push to main pypi
python -m twine upload --repository pypi dist/*

# The below should work with ~/.pypirc but doesn't for some reason
# uv publish --publish-url https://test.pypi.org/
# uv publish

# see API keys in ~/.pypirc
# create here: https://testpypi.org/manage/account/token/
# create here: https://pypi.org/manage/account/token/

## now install in "editable" mode
# conda activate pyrsm
pip uninstall -y pyrsm
rm -rf ~/gh/pyrsm/dist
uv pip install --user -e ~/gh/pyrsm


pip install --user pyrsm --upgrade
python -c "import pyrsm as rsm; print(rsm.__version__); print(rsm.__file__)"
