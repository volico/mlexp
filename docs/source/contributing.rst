Contributing to MLexp
=====================

To contribute to MLexp, please follow guidelines here.

Development Workflow
####################

1. Fork the `mlexp` repo. 
2. Clone the forked repo locally
3. Create a branch for the change:

.. code-block:: bash

    git checkout -b branch_name

4. Make a change
5. Test your change (code and docs, see below)
6. Commit the change to your local branch
7. Push your local branch to remote

.. code-block:: bash

    git push origin branch_name

8. Create PR to merge your branch in develop branch of `mlexp` repo


Requirements
############

Install requirements:

.. code-block:: bash

    pip install -r requirements.txt

Install development requirements (tests, documentation, formatting):

.. code-block:: bash

    pip install -r requirements-dev.txt

Tests
#####

To test code locally you have to start mlflow server

.. code-block:: bash

    mlflow server

Also you have to `set environment variable with neptune token <https://docs.neptune.ai/setup/setting_api_token/>`_:

.. code-block:: bash

    export NEPTUNE_API_TOKEN=<neptune token>

Neptune project is also required (check args for more details)

To run tests:

.. code-block:: bash

    make test neptune_project=<neptune user/neptune project>

Minimum code coverage is 75%

To test only documentation:

.. code-block:: bash

    $ make test_docs

Documentation
#############

Docstrings
##########

All public classes and functions must have docstrings.

**MLexps** uses rst format of docstrings.


Generating Documentation Locally
################################
You can generate documentation in HTML locally as follows:

.. code-block:: bash

    make build_docs

Documentation will be available in `docs/build/html/index.html`.

Code formatting
###############
**MLexps** uses `ufmt <https://pypi.org/project/ufmt/>`_ for formatting code.

To format code:

.. code-block:: bash

    ufmt format .
