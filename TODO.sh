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
