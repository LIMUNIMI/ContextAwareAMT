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

#. Install 'Maestro' dataset from ``asmd``: ``python -m mpc2c.asmd.asmd.install``
#. Install Carla with ``python -m mpc2c.pycarla.pycarla.carla -d``
#. Install ``jackd`` in your path
#. Prepare the new dataset with the resynthesized parts: ``python run.py -d``
#. If the process stops, relaunch it (it will skip the already synthesized songs)
#. If it fails, it's probably because Carla died in zombie process; just stop
   (CTRL+C) and restart (it will skip already synthesized songs)
#. The datasets were split using PCA and retaining 0.89, 0.93, 0.91 of total
   variance for `train`, `validation` and `test` set respectively.

1. Preprocess
-------------

#. Create the MIDI file for the initial template: ``python run.py -sc``
#. Synthesize the midi scale and name it ``pianoteq_scales.mp3`` (TODO: resynthesize using pycarla)
#. Compute the initial template and save it to file: ``python run.py --template``
#. Apply NMF and extract notes for velocity estimation: ``python run.py -v -r -c orig``
#. Apply NMF and extract frames for pedaling estimation: ``python run.py -p -r -c orig``

   * Train set: 20 songs for specific contexts, 847 for the orig
   * Validation set: 10 songs for specific contexts, 77 for the orig
   * Test set: 25 songs for specific contexts, 28 for the orig

2. Training the generic model
-----------------------------

#. Look for hyper-parameters for velocity using the original context: ``python
   run.py -v -sk -c orig``. We obtained hyperparams defined in ``settings.py``
   and loss function of 0.0838 (about like the dummy predictor but there is
   the complexity cost!). Learning rate: 4.07e-04.
#. Look for hyper-parameters for pedaling using the original context: ``python
   run.py -p -sk -c orig``. We obtained hyperparams defined in ``settings.py``
   and loss function of 0.2206. Learning rate: 1.19e-01.
#. Fully train velocity model on the original context: ``python run.py -v -t -c orig``

   * Dummy loss: 0.1220
   * Validation loss: 0.1213 (20 epochs, early-stop)
   * 1.004.974 batches in training
   * 73.066 batches in validation
   * Learning rate: 9.88e-6
   * 721 parameters::

    MIDIParameterEstimation(
      (lstm): LSTM(12, 8, batch_first=True)
      (stack): Sequential(
        (0): Conv2d(1, 1, kernel_size=(3, 5), stride=(1, 1), bias=False)
        (1): Identity()
        (2): Tanh()
        (3): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
        (4): Sigmoid()
      )
    )

#. Fully train pedaling model on the original context: ``python run.py -p -t -c orig``

   * Dummy loss: 0.2604
   * Validation loss: 0.2068 (57 epochs with early-stop)
   * 847 batches in training
   * 77 batches in validation
   * Learning rate: 0.0118
   * 5924 parameters::

    MIDIParameterEstimation(
      (lstm): LSTM(12, 32, batch_first=True)
      (stack): Sequential(
        (0): Conv2d(3, 3, kernel_size=(3, 1), stride=(1, 1), groups=3, bias=False)
        (1): InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Identity()
        (3): Conv2d(3, 3, kernel_size=(3, 1), stride=(1, 1), groups=3, bias=False)
        (4): InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), groups=3)
        (7): Sigmoid()
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

#. Apply NMF to each context: ``python run.py -p -r -c <context>``, ``python
   run.py -v -r -c <context>``

#. Fully train velocity model on the original context: ``python run.py -v -t -c
   <context> -pt <path to generic model chekcpoint>``

#. Fully train pedaling model on the original context: ``python run.py -p -t -c
   <context> -pt <path to generic model chekcpoint>``

#. After each training, you will find a file named `checkpoint0.????.pt`
   containing the checkpoint with the trained parameters. Save theme somewhere.

Here ``<context>`` is any Carla preset name that you have used before.

Results for velocity
~~~~~~~~~~~~~~~~~~~~

   * Retrained parameters: 2 (last conv module)

   #. pianoteq0:

      * Dummy loss: 0.1216
      * Validation loss: 0.1214 (23 epochs with early-stop)
      * Training 22669 batches, validation 6567 batches
      * Learning rate: 2.21e-5

   #. pianoteq1:

      * Dummy loss: 0.1204
      * Validation loss: 0.1190 (30 epochs with early-stop)
      * Training 24457 batches, validation 6672 batches
      * Learning rate: 2.04e-4

   #. pianoteq2:

      * Dummy loss: 0.1217
      * Validation loss: TODO (TODO epochs with early-stop)
      * Training 22405 batches, validation 7729 batches
      * Learning rate: 2.23e-4

   #. pianoteq3:

      * Dummy loss: TODO
      * Validation loss: TODO (TODO epochs with early-stop)
      * Training TODO batches, validation TODO batches
      * Learning rate: TODO

   #. salamander0:

      * Dummy loss: TODO
      * Validation loss: TODO (TODO epochs with early-stop)
      * Training TODO batches, validation TODO batches
      * Learning rate: TODO

   #. salamander1:

      * Dummy loss: TODO
      * Validation loss: TODO (TODO epochs with early-stop)
      * Training TODO batches, validation TODO batches
      * Learning rate: TODO

Results for pedaling
~~~~~~~~~~~~~~~~~~~~

   * Retrained parameters: 6 (last conv module)
   * Training 20 batches, validation 10 batches
   * Learning rate: 0.025

   #. pianoteq0:

      * Dummy loss: 0.2723
      * Validation loss: 0.2022 (92 epochs with early-stop)

   #. pianoteq1:

      * Dummy loss: 0.2751
      * Validation loss: 0.2103 (302 epochs with early-stop)

   #. pianoteq2:

      * Dummy loss: 0.2721
      * Validation loss: 0.2168 (49 epochs with early-stop)

   #. pianoteq3:

      * Dummy loss: 0.2395
      * Validation loss: 0.19103 (500 epochs NO early-stop)

   #. salamander0:

      * Dummy loss: 0.2871
      * Validation loss: 0.2417 (53 epochs with early-stop)

   #. salamander1:

      * Dummy loss: 0.2623
      * Validation loss: 0.2255 (500 epochs with NO early-stop)


4. Evaluating error distributions
---------------------------------

#. Evaluate error distributions of velocity models whose checkpoints are in a
   given directory: ``python run.py -v -e <list of checkpoints> -cp``; you can
   use shell expansion like ``models/*_vel.pt``
#. Evaluate error distributions of pedaling models whose checkpoints are in a
   given directory: ``python run.py -p -e <list of checkpoints> -cp``; you can
   use shell expansion like ``models/*_ped.pt``

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

Notes
-----

We used 6 different artificial contexts:

#. `pianoteq0` is based on `Pianoteq Stage Steinway Model B`; linear mapping of
   velocities (0-127) -> (ppp-fff) and small/no reverb ("Jazz Studio")
#. `pianoteq1` is based on `Pianoteq Stage  Grotrian Recording 3`; linear mapping of
   velocities (0-127) -> (p-f) and medium reverb ("Medium Hall")
#. `pianoteq2` is based on `Pianoteq Stage  Grotrian Player`; linear mapping of
   velocities (23-94) -> (ppp-fff) and  small/no reverb ("Jazz Studio")
#. `pianoteq3` is based on `Pianoteq Stage  Grotrian Player`; almost exponential mapping of
   velocities (0-127) -> (ppp-fff) and large reverb ("Large Hall")
#. `salamander0` is based on `SalamnderGrandPianoV3Retuned` with no reverb
#. `salamander1` is based on `SalamnderGrandPianoV3Retuned` with `Calf` reverb
   ("Large", 2.15 sec decay)


Credits
=======

#. `Federico Simonetta <https://federicosimonetta.eu.org>`_
