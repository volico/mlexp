build_docs: export SPHINX_APIDOC_OPTIONS = members,inherited-members

.PHONY: dev_reqs
dev_reqs:
	pip install -r requirements-dev.txt

.PHONY: reqs
reqs:
	pip install -r requirements.txt

.PHONY: sdist
sdist: clean reqs
	python setup.py sdist

.PHONY: build_docs
build_docs: clean dev_reqs
	pip install .[torch]
	sphinx-apidoc -f --no-toc --templatedir=docs/source/_templates -e -o docs/source/api mlexp
	sphinx-build -M html docs/source docs/build
	python setup.py clean --all
	rm -rf build dist mlexp.egg-info

.PHONY: install
install: clean
	pip install .

.PHONY: test
test: install dev_reqs
	coverage run -m pytest -x --neptune_project $(neptune_project)
	python setup.py clean --all
	rm -rf build dist mlexp.egg-info

.PHONY: test_docs
test_docs: install dev_reqs
	coverage run -m pytest tests/test_docs --neptune_project empty
	python setup.py clean --all
	rm -rf build dist mlexp.egg-info

.PHONY: clean
clean:
	python setup.py clean --all
	rm -rf build dist mlexp.egg-info