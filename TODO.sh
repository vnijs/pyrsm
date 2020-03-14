
# to install locally
sudo pip3 uninstall pyrsm
# doesn't work with testing for some reason
# sudo pip3 install --user -e .
sudo python3 setup.py install

# from https://github.com/vnijs/pypi-howto
# pip3 install --user twine
sudo pip3 install --upgrade twine keyring

# first upload to pypitest
rm dist/*
python3 setup.py sdist
python3 -m twine upload dist/* -r pypitest
# sudo pip3 install --upgrade --force-reinstall --index-url https://test.pypi.org/simple/ pyrsm

# then upload to pypi
rm dist/*; python3 setup.py sdist
python3 -m twine upload dist/* -r pypi
# sudo pip3 install --upgrade --force-reinstall pyrsm==0.1.3
