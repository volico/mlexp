# Contributing to MLexp

To contribute to MLexp, please follow guidelines here.

We use [`black`](https://black.readthedocs.io/en/stable/index.html) as a formatter to keep the coding style and format across all Python files consistent and compliant with [PEP8](https://www.python.org/dev/peps/pep-0008/). We recommend that you add `black` to your IDE as a formatter (see the [instruction](https://black.readthedocs.io/en/stable/integrations/editors.html)) or run `black` on the command line before submitting a PR as follows:
```bash
# move to the top directory of the causalml repository
$ cd causalml 
$ pip install -U black
$ black .
```

## Development Workflow

1. Fork the `mlexp` repo. 
2. Clone the forked repo locally
3. Create a branch for the change:
```bash
$ git checkout -b branch_name
```
4. Make a change
5. Test your change (code and docs, see below)
6. Commit the change to your local branch
7. Push your local branch to remote
```bash
$ git push origin branch_name
```
8. Create PR to merge your branch in develop branch of `mlexp` repo


## Requirements

Install requirements:
```bash
$ pip install -r requirements.txt
```
Install development requirements (tests, documentation, formatting):
```bash
$ pip install -r requirements-dev.txt
```

## Tests

To test code locally you have to start mlflow server
```bash
$ mlflow server
```

Neptune project is also required (check args for more details)

To run tests:
```bash
$ coverage run -m pytest --neptune_project <neptune project>
```

Minimum code coverage is 75%

To test only documentation:
```bash
$ pytest tests/test_docs --neptune_project <neptune project>
```

## Documentation

### Docstrings

All public classes and functions must have docstrings.

**MLexps** uses rst format of docstrings.


### Generating Documentation Locally

You can generate documentation in HTML locally as follows:
```bash
$ cd docs/
$ bash build_docs.sh
```

Documentation will be available in `docs/build/html/index.html`.
