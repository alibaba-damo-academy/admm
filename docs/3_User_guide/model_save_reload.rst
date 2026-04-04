.. include:: ../definition.hrst

.. _user-guide-model-persistence:

Model Save and Reload
======================

Save a model when the symbolic formulation is already right and you want to reuse it later without
rebuilding the variables, objective, and constraints from code.

.. list-table:: Persistence operations
   :widths: 26 34 40
   :header-rows: 1
   :class: longtable

   * - Task
     - API
     - Typical use
   * - save a symbolic model
     - ``model.write("model.admm")``
     - keep a finished formulation for later runs or sharing
   * - restore a saved model
     - ``admm.Model(file="model.admm")``
     - reopen the formulation without rebuilding it from code

A minimal workflow looks like this:

.. code-block:: python

    model.write("model.admm")

    restored = admm.Model(file="model.admm")


What Is Saved
-------------

The file stores the **symbolic model structure** only:

- variable definitions — names, shapes, and structural attributes such as nonnegativity or PSD
- parameter definitions — names and shapes
- constraints — left-hand side, sense, and right-hand side
- expressions — the operation tree that represents the objective and constraint expressions
- objective sense — minimize or maximize

The saved file is a self-contained text representation of the formulation.


What Is Not Saved
-----------------

- **Parameter values** — parameters are bound at solve time via
  ``model.optimize({"alpha": data})``, not stored in the model file. After reloading, bind them
  again before solving.
- **Variable solutions** — the model reloads as an unsolved formulation. Call
  ``model.optimize()`` to obtain new solution values.
- **Solver options** — any options set via ``model.setOption(...)`` must be re-applied after
  reloading.


Compression
-----------

If the filename ends with ``.gz`` or ``.bz2``, compression and decompression happen automatically:

.. code-block:: python

    model.write("model.admm.gz")                  # writes a gzip-compressed file
    restored = admm.Model(file="model.admm.gz")   # reloads transparently


Example: Save, Reload, and Re-solve
------------------------------------

.. code-block:: python

    import admm
    import numpy as np

    # Build model
    model = admm.Model()
    x = admm.Var("x", 3)
    model.setObjective(admm.sum(admm.square(x - np.array([1.0, 2.0, 3.0]))))
    model.addConstr(x >= 0)

    # Save
    model.write("my_model.admm")

    # Reload and solve
    restored = admm.Model(file="my_model.admm")
    restored.optimize()

    # Access results on the restored model
    print(restored.StatusString)


For the broader end-to-end sequence, see :ref:`Modeling Workflow <user-guide-modeling-workflow>`.
