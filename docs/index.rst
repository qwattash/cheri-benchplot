.. cheri-benchplot documentation master file, created by
   sphinx-quickstart on Mon Nov  7 09:46:28 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cheri-benchplot's documentation!
===========================================
Benchplot is a benchmark analysis tool intended to be used by CHERI-related projects.
The main purpose of the project is to support data collection and analysis in a reusable
and reproducible way. This can be easily integrated with more complex CI orchestration as well.

The main entry point is the *benchplot-cli* command line tool.
Benchplot activities are divided into sessions. Each session is associated to a directory tree upon
creation, there it will store all benchmark artefacts.
Once a session is created, it can be run or analysed. The run step runs the benchmark from the
configuration and generates data. The analysis step produces outputs from the data, and can be
run multiple times with different configurations to generate different representations of the results.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
