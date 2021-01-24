
# to install locally
sudo pip3 uninstall pyrsm
# doesn't work with testing for some reason

# sudo pip3 install --user -e .
sudo python3 setup.py install

# from https://github.com/vnijs/pypi-howto
# pip3 install --user twine
sudo pip3 install --upgrade twine keyring

# https://kynan.github.io/blog/2020/05/23/how-to-upload-your-package-to-the-python-package-index-pypi-test-server
# tokens are in ~/.pypirc

sudo python3 setup.py sdist bdist_wheel
twine check dist/*
twine upload --repository testpypi dist/*

# if all goes well
twine upload dist/*
