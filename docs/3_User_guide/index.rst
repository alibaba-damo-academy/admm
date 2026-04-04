.. include:: ../definition.hrst

.. _doc-user-guide:

User Guide
==========

Use this chapter as a guided path from your first |ADMM| model to the deeper ideas behind the modeling
workflow. For exact signatures, see the :doc:`../5_API_Document/index`.

.. rst-class:: user-guide-nav-heading

.. rubric:: How to Use This Chapter

Read this chapter in two passes. ``Getting Started`` is the hands-on path: begin with a minimal example,
confirm that your formulation fits the supported structure, and then work through the standard modeling
sequence from variables to solve-time results. ``Conceptual Background`` is the deeper path: return there
after the workflow feels familiar to understand the available building blocks, the symbolic rewrites
performed under the hood, and how to extend the library when built-in atoms are not enough.

If you are new to |ADMM|, follow the roadmap below from top to bottom. If you already know the basic
workflow and want the reasoning behind it, jump directly to the conceptual material.

.. list-table:: Guided roadmap
   :widths: 35 65
   :header-rows: 1
   :class: longtable

   * - Section
     - What you learn next
   * - :ref:`Minimal Model <user-guide-minimal-model>`
     - build a first working model with one parameter, one objective, and simple constraints
   * - :ref:`Supported Problem Structure <user-guide-supported-problem-structure>`
     - check early whether your formulation matches the patterns that |ADMM| is designed to handle
   * - :ref:`Modeling Workflow <user-guide-modeling-workflow>`
     - see the full end-to-end sequence so the rest of the chapter has a clear map
   * - :ref:`Variables <user-guide-variables>`
     - introduce scalar, vector, and matrix variables together with the structural attributes they need
   * - :ref:`Parameters <user-guide-parameters>`
     - separate reusable model structure from named data that you bind at solve time
   * - :ref:`Objective <user-guide-objective>`
     - assemble a scalar objective from affine terms, quadratics, and supported atoms
   * - :ref:`Constraints <user-guide-constraints>`
     - add equality, inequality, and cone-style conditions to complete the model
   * - :ref:`Solver Options <user-guide-solver-options>`
     - tune practical solver settings and learn the checks that matter for reliable runs
   * - :ref:`Solve the Model <user-guide-solve-the-model>`
     - run the solve step and interpret the returned results with the right expectations
   * - :ref:`Model Save and Reload <user-guide-model-persistence>`
     - save a finished model and reload it for later runs or downstream workflows
   * - :ref:`Supported Building Blocks <user-guide-objective-building-blocks>`
     - step back from the workflow and study the losses, penalties, indicators, and structural atoms
   * - :ref:`Symbolic Canonicalization <user-guide-symbolic-canonicalization>`
     - understand how |ADMM| rewrites expressions into solver-ready forms behind the scenes
   * - :ref:`User-Defined Proximal Extensions <user-guide-udf>`
     - extend the library with custom proximal operators when the standard path is no longer enough

This is the practical route through the modeling workflow. Follow these pages in order if you want to get
a model running before diving into the internals.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   minimal_model
   supported_problem_structure
   modeling_workflow
   variables
   parameters
   objective
   constraints
   solver_options
   solve
   model_save_reload

This is the follow-on reading path for understanding what the workflow is built from and how it can be
extended.

.. toctree::
   :maxdepth: 1
   :caption: Conceptual Background

   objective_building_blocks
   symbolic_canonicalization
   udf