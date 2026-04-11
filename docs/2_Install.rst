.. include:: definition.hrst

.. _doc-install:

Installation
===========================

Start here to confirm supported platforms, install |ADMM|, and verify that the Python environment is ready
for modeling and solving with |ADMM|.

Recommended order:

1. Check :ref:`Supported Platforms <install-supported-platforms>` to confirm OS and Python support.
2. Follow :ref:`Install ADMM Python Library <install-python-library>` to install the ``admm`` package.
3. Run the verification step before moving on to the User Guide.

.. _install-supported-platforms:

Supported Platforms
-------------------

The tables below summarize the supported operating systems and Python versions for the documented Python package.

.. table:: Supported operating systems

   =================  ===============================================
   Operating systems  Requirements
   =================  ===============================================
   Windows            Windows 10 or higher
   Linux              GLIBC 2.17 or higher
   macOS              Apple Silicon, macOS 12.0 or higher
   =================  ===============================================


.. table:: Supported Python versions

   =========  ==========================================================================
   Language   Requirements
   =========  ==========================================================================
   Python     Python 3.10 or higher
   =========  ==========================================================================

The examples in this documentation assume a standard scientific Python environment with ``numpy`` available.
If your platform matches the tables above, continue to :ref:`Install ADMM Python Library <install-python-library>`.

.. _install-python-library:

Install |ADMM| Python Library
-----------------------------

You can install ADMM from PyPI or from source.

Install from PyPI
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install admm

To upgrade an existing installation:

.. code-block:: bash

   pip install --upgrade admm


Install from Source
^^^^^^^^^^^^^^^^^^^

Prerequisites:

- Python >= 3.10
- C++ compiler (GCC, Clang, or MSVC)
- admmlib >= 2026.4.9 (admm C++ core dependency library)

Install steps:

.. code-block:: bash

   git clone https://github.com/alibaba-damo-academy/admm.git
   cd admm

   pip install . -r requirements.txt

Isolated Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using a virtual environment helps avoid dependency conflicts and keeps the installation reproducible.
If you use ``conda``, create and activate an environment first, then install the package with ``pip``:

.. code-block:: bash

   conda create --name admm-py310 python=3.10
   conda activate admm-py310
   pip install admm

On macOS, make sure the Python environment matches the machine architecture. When using ``conda``, it is
worth checking that the reported architecture matches ``uname -m``.


Installation Verification
^^^^^^^^^^^^^^^^^^^^^^^^^

After installation, first run a short import check:

.. code-block:: bash

   python -c "import admm; print('ADMM import OK')"

If the command prints ``ADMM import OK`` without errors, the Python package is ready to use.

Then run a minimal solve smoke test:

.. code-block:: python

   import admm
   import numpy as np

   model = admm.Model()
   x = admm.Var("x", 2)
   model.setObjective(admm.sum(admm.square(x - np.array([1.0, 2.0]))))
   model.optimize()

   print(model.StatusString)
   print(x.X)

If this runs without errors and returns a valid solution, the installation is ready for the
:ref:`User Guide <doc-user-guide>`.


Troubleshooting
^^^^^^^^^^^^^^^

If installation or import fails, check the following points:

- Confirm that you are using a supported Python version.
- Make sure the active environment is the one where ``admm`` was installed.
- Upgrade ``pip`` before retrying the installation.
- On macOS, check for architecture mismatches between Python, ``conda``, and the host machine.
