
# to install locally
sudo pip3 uninstall -y pyrsm
sudo python3 setup.py install

# from https://github.com/vnijs/pypi-howto
sudo pip3 install --upgrade twine keyring

# https://kynan.github.io/blog/2020/05/23/how-to-upload-your-package-to-the-python-package-index-pypi-test-server
# tokens are in ~/.pypirc

sudo rm -rf ~/gh/pyrsm/dist
sudo python3 setup.py sdist bdist_wheel
twine check dist/*
twine upload --repository testpypi dist/*

# if all goes well push to main pypi
twine upload dist/*
