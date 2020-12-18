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

Config
------

#. Open ``mpc2c/settings.py`` and check the variables in section 'PATH',
   especially ``VELOCIY_DATA_PATH``, ``PEDALING_DATA_PATH``, and
   ``RESYNTHESIS_DATA_PATH``. Set them to meaningful paths for your system.
#. Make sure that you have ``jackd`` installed
#. Run ``poetry run python -m pycarla.carla -d`` to download Carla host
#. Prepare your Carla presets by using ``poetry run python -m pycarla.carla
   -r`` and put them in the ``mpc2c.settings.CARLA_PROJ`` directory; the
   included ones need:

    * ``Pianoteq 6 STAGE`` LV2 plugin installed and available to Carla (e.g. in ``~/.lv2/`` or ``/usr/lib/lv2/``)
    * ``Calf Reverb`` LV2 plugin installed and available to Carla (e.g. in ``~/.lv2/`` or ``/usr/lib/lv2/``)
    * ``SalamanderGrandPianoV3Retuned`` SFZ version installed in
      ``/opt/soundfonts/SalamanderGrandPianoV3+20161209_48khz24bit/SalamanderGrandPianoV3Retuned.sfz``


Datasets
--------

#. Install 'Maestro' dataset from ``asmd``: ``python -m asmd.install``
#. Prepare the new dataset with the resynthesized parts: ``python run.py --datasets``
#. If the process stops, relaunch it (it will skip the already synthesized songs)

1. Preprocess
-------------

#. Create the MIDI file for the initial template: ``python run.py --scale``
#. Synthesize the midi scale and name it ``pianoteq_scales.mp3`` (TODO: resynthesize using jack_synth)
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

3. Modelling the contexts
-------------------------

-- TODO --

4. Evaluating error distributions
---------------------------------

-- TODO --

Credits
=======

#. `Federico Simonetta <https://federicosimonetta.eu.org>`_
