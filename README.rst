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
#. Prepare the new dataset with the resynthesized parts: ``python run.py -d``
#. If the process stops, relaunch it (it will skip the already synthesized songs)

1. Preprocess
-------------

#. Create the MIDI file for the initial template: ``python run.py -sc``
#. Synthesize the midi scale and name it ``pianoteq_scales.mp3`` (TODO: resynthesize using pycarla)
#. Compute the initial template and save it to file: ``python run.py --template``

2. Training the generic model
-----------------------------

#. Apply NMF and extract notes for velocity estimation: ``python run.py -tv -r -c orig``
#. Apply NMF and extract frames for pedaling estimation: ``python run.py -tp -r -c orig``
#. Look for hyper-parameters for velocity using the original context: ``python
   run.py -tv -sk -c orig``. We obtained hyperparams defined in ``settings.py`` (TODO)
   and loss function of TODO. Note the preference for `AbsLayer` on `ReLU`.
   Learning rate: 6.41e-04
#. Look for hyper-parameters for pedaling using the original context: ``python
   run.py -tp -sk -c orig``. We obtained hyperparams defined in ``settings.py`` (TODO)
   and loss function of TODO. Learning rate: 1.19e-02
#. Fully train velocity model on the original context: ``python run.py -tv -c orig``

   * Model: 30 parameters::

      MIDIParameterEstimation(
        (stack): Sequential(
          (0): Conv2d(1, 1, kernel_size=(3, 4), stride=(1, 1), bias=False)
          (1): Identity()
          (2): AbsLayer()
          (3): Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (4): Identity()
          (5): AbsLayer()
          (6): Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (7): Identity()
          (8): AbsLayer()
          (9): Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (10): Identity()
          (11): AbsLayer()
          (12): Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (13): Identity()
          (14): AbsLayer()
          (15): Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), bias=False)
          (16): Sigmoid()
          (17): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
        )
      )

   * Dummy loss: 0.12082
   * Validation loss: TODO (TODO epochs, early-stop)
   * 1.004.974 batches in training
   * 73.066 batches in validation

#. Fully train pedaling model on the original context: ``python run.py -tp -c orig``

    TODO

---

* option ``-r`` preprocess the dataset using NMF; it should be used only once
  per each type of model; each subsequent runs will use the already dumped
  dataset
* option ``-r`` must be associated to some action that loads datasets (this
  could be improved...)
* option ``-sk`` reduces the dataset to 10% of its total; thus, ``-sk -r``
  would result in preprocessing only that 10%


3. Training the context-specific models
---------------------------------------

#. Apply NMF and extract notes for velocity estimation: ``python run.py -tv -r -c <context>``
#. Apply NMF and extract frames for pedaling estimation: ``python run.py -tp -r -c <context>``

-- TODO --
#. Fully train velocity model on the original context: ``python run.py -tv -c
   <context> -gm <path to generic model>``
#. Fully train pedaling model on the original context: ``python run.py -tp -c
   <context> -gm <path to generic model>``

Here ``<context>`` is any Carla preset name that you have used before.

4. Testing on a specific file
-----------------------------

-- TODO --

#. Fully train velocity model on the original context: ``python run.py -tv -gm <path to generic model> -cm <path to context model> -i <input midi> <input audio>``

5. Evaluating error distributions
---------------------------------

-- TODO --

Credits
=======

#. `Federico Simonetta <https://federicosimonetta.eu.org>`_
