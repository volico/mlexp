Installation
=============

.. _PyPI: https://pypi.org/project/mlexp/
.. _source: https://github.com/volico/mlexp
.. _pytorch-lightning: https://github.com/Lightning-AI/lightning

**Install** the package with the following command from PyPI_:

.. code-block:: bash

    pip install mlexp

By default, MLexp is installed without support of torch training.

To support training torch models install MLexp as:

.. code-block:: bash

    pip install mlexp[torch]

This way pytorch-lightning_ will also be installed.

It is advised to install desired version of torch before installing mlexp[torch]
because pytorch-lightning depends on torch and by default downloads
torch with CUDA support (and heavy dependencies).

Installation from source_:

.. code-block:: bash

    git clone https://github.com/volico/mlexp
    cd mlexp
    python setup.py mlexp