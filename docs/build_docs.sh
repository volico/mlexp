echo WARNING: documentation will now be updated with sphinx-apidoc and sphinx make html commands

sleep 7

export SPHINX_APIDOC_OPTIONS="members,inherited-members"
sphinx-apidoc -f --no-toc --templatedir=source/_templates -e -o source/api ../mlexp
sphinx-build -M html source build
