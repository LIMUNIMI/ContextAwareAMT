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

#. Apply NMF and extract notes for velocity estimation: ``python run.py -v -r -c orig``
#. Apply NMF and extract frames for pedaling estimation: ``python run.py -p -r -c orig``
#. Look for hyper-parameters for velocity using the original context: ``python
   run.py -v -sk -c orig``. We obtained hyperparams defined in ``settings.py``
   and loss function of 0.10486 (about like the dummy predictor but there is
   the complexity cost!). Note the preference for `AbsLayer` on both `ReLU` and
   `Identity`.  Learning rate: 6.41e-04.
#. Look for hyper-parameters for pedaling using the original context: ``python
   run.py -p -sk -c orig``. We obtained hyperparams defined in ``settings.py``
   and loss function of 0.18612. Learning rate: 1.19e-02. Note the preference
   for `Identity` on `ReLU`.
#. Fully train velocity model on the original context: ``python run.py -v -t -c orig``

   * Dummy loss: 0.12082
   * Validation loss: TODO (TODO epochs, early-stop)
   * 1.004.974 batches in training
   * 73.066 batches in validation
   * Learning rate: TODO
   * TODO parameters::

      TODO

#. Fully train pedaling model on the original context: ``python run.py -p -t -c orig``

   * Dummy loss: 0.2640
   * Validation loss: 0.2098 (362 epochs with early-stop)
   * 847 batches in training
   * 77 batches in validation
   * Learning rate: 1.18e-2
   * 2358 parameters::

      MIDIParameterEstimation(
        (stack): Sequential(
          (0): Conv2d(1, 16, kernel_size=(4, 1), stride=(1, 1), bias=False)
          (1): InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Identity()
          (3): Conv2d(16, 16, kernel_size=(4, 1), stride=(1, 1), bias=False)
          (4): InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): Identity()
          (6): Conv2d(16, 16, kernel_size=(4, 1), stride=(1, 1), bias=False)
          (7): InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (8): Identity()
          (9): Conv2d(16, 3, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (10): Sigmoid()
          (11): Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), groups=3)
        )
      )

#. After each training, you will find a file named `checkpoint0.????.pt`
   containing the checkpoint with the trained parameters. Save it somewhere.

---

* option ``-r`` preprocess the dataset using NMF; it should be used only once
  per each type of model; each subsequent runs will use the already dumped
  dataset
* option ``-sk`` reduces the dataset to 10% of its total for pedaling and to
  1.5% for velocity; thus, ``-sk -r`` would result in preprocessing only that
  10% and 1.5%


3. Training the context-specific models
---------------------------------------

#. Apply NMF and extract notes for velocity estimation: ``python run.py -v -r -c <context>``
#. Apply NMF and extract frames for pedaling estimation: ``python run.py -p -r -c <context>``
#. Fully train velocity model on the original context: ``python run.py -v -t -c
   <context> -pt <path to generic model chekcpoint>``

#. Fully train pedaling model on the original context: ``python run.py -p -t -c
   <context> -pt <path to generic model chekcpoint>``
   
   * Learning rate: 0.05
   * Training 20 batches, validation 10 batches

   #. pianoteq0:

      * Dummy loss: 0.2521
      * Validation loss: 0.1775 (202 epochs with early-stop)
   
#. After each training, you will find a file named `checkpoint0.????.pt`
   containing the checkpoint with the trained parameters. Save theme somewhere.

Here ``<context>`` is any Carla preset name that you have used before.

4. Evaluating error distributions
---------------------------------

-- TODO --
#. Evaluate error distributions of velocity models whose checkpoints are in a given directory: ``python run.py -v -e <list of checkpoints> -cp``; you can use shell expansion like ``models/*_vel.pt``
#. Evaluate error distributions of pedaling models whose checkpoints are in a given directory: ``python run.py -p -e <list of checkpoints> -cp``; you can use shell expansion like ``models/*_ped.pt``

These commands will create a plotly plots with violin plots of generic and
specific contexts and Wilcoxon p-values.

Note that the usage of ``-cp`` is only possible if you name your checkpoints
with the relative context in the initial part of the filename (e.g.
``models/pianoteq0_vel.pt``).

5. Testing on a specific file
-----------------------------

N.B. Not yet implemented!

#. Fully test a velocity model on a specific audio/midi file: ``python run.py -v -pt <path to model checkpoint.pt> -i <input midi path> <input audio path>``
#. Fully test a pedaling model on a specific audio/midi file: ``python run.py -p -pt <path to model checkpoint.pt> -i <input midi path> <input audio path>``

Credits
=======

#. `Federico Simonetta <https://federicosimonetta.eu.org>`_
