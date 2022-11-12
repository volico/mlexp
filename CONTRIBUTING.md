# Contributing to MLexp

To contribute to MLexp, please follow guidelines here.

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
$ make test
```

Minimum code coverage is 75%

To test only documentation:
```bash
$ make test_docs
```

## Documentation

### Docstrings

All public classes and functions must have docstrings.

**MLexps** uses rst format of docstrings.


### Generating Documentation Locally

You can generate documentation in HTML locally as follows:
```bash
$ make build_docs
```

Documentation will be available in `docs/build/html/index.html`.

## Code formatting

**MLexps** uses [ufmt](https://pypi.org/project/ufmt/) for formatting code.

To format code:

```bash
$ ufmt format .
```
