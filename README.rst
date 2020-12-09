=============================================
Mapping Music Performances Context to Context
=============================================

Setup
-----

#. Enter the root git directory
#. Install ``poetry``
#. Install ``pyenv``
#. Install python 3.6.9: ``pyenv install 3.6.9``
#. Activate it: ``pyenv shell 3.6.9``
#. Create a new venv with poetry and install the dependencies: ``poetry update``
#. Start a new shell in this venv: ``poetry shell``

Datasets
--------

#. Install 'Maestro' dataset from ``asmd``: ``python -m asmd.install``

1. Preprocess
-------------

#. Create the MIDI file for the initial template: ``python run.py --scale``
#. Synthesize the midi scale and name it ``pianoteq_scales.mp3``
#. Compute the initial template and save it to file: ``python run.py --template``

2. Training
-----------

#. Look for hyper-parameters for velocity: ``python run.py --train-velocity --skopt --redump``
#. Look for hyper-parameters for pedaling: ``python run.py --train-pedaling --skopt --redump``
#. Fully train velocity model: ``python run.py --train-velocity``
#. Fully train pedaling model: ``python run.py --train-pedaling``

N.B. option ``--redump`` preprocess the dataset using NMF; it should be used
only once per each type of model; each subsequent runs will use the already
dumped dataset

3. Synthesizing the contexts
----------------------------

-- TODO --

4. Modelling the contexts
-------------------------

-- TODO --

5. Evaluating error distributions
---------------------------------

-- TODO --

Credits
=======

#. `Federico Simonetta <https://federicosimonetta.eu.org>`_
