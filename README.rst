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
   * TODO: after resynthesis with maestro v3.0, update these data

#. Apply NMF to each context: ``python run.py -p -r -c <context>``, ``python
   run.py -v -r -c <context>``


2. Training the generic model
-----------------------------

#. Look for hyper-parameters for velocity using the original context: ``python
   run.py -v -sk``. We obtained hyperparams defined in ``settings.py``
   and loss function of 0.1143.
#. Look for hyper-parameters for pedaling using the original context: ``python
   run.py -p -sk``. We obtained hyperparams defined in ``settings.py``
   and loss function of 0.1803.
#. Fully train velocity model on the original context: ``python run.py -v -t -c orig``

   * Dummy loss: 0.1207
   * Validation loss: 0.1409 (69 epochs)
   * 354845 batches in training
   * 55008 batches in validation
   * Learning rate: 1.41e-5
   * 73287 parameters::

      MIDIParameterEstimation(
        (dropout): Dropout(p=0.1, inplace=False)
        (lstm): LSTM(13, 128, batch_first=True)
        (stack): Sequential(
          (0): Conv2d(1, 1, kernel_size=(3, 6), stride=(1, 1), bias=False)
          (1): Identity()
          (2): ReLU()
          (3): Conv2d(1, 1, kernel_size=(3, 6), stride=(1, 1), bias=False)
          (4): Identity()
          (5): ReLU()
          (6): Conv2d(1, 1, kernel_size=(3, 6), stride=(1, 1), bias=False)
          (7): Identity()
          (8): ReLU()
          (9): Conv2d(1, 1, kernel_size=(1, 15), stride=(1, 1), bias=False)
          (10): Identity()
          (11): ReLU()
          (12): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (13): Sigmoid()
        )
      )

#. Fully train pedaling model on the original context: ``python run.py -p -t -c orig``

   * Dummy loss: 0.2578
   * Validation loss: 0.1963 (500 epochs)
   * 247 batches in training
   * 47 batches in validation
   * Learning rate: 2.02e-2
   * 6052 parameters::

      MIDIParameterEstimation(
        (dropout): Dropout(p=0.1, inplace=False)
        (lstm): LSTM(13, 32, batch_first=True)
        (stack): Sequential(
          (0): Conv2d(3, 3, kernel_size=(4, 1), stride=(1, 1), groups=3, bias=False)
          (1): InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Tanh()
          (3): Conv2d(3, 3, kernel_size=(2, 1), stride=(1, 1), groups=3, bias=False)
          (4): InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): Tanh()
          (6): Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), groups=3)
          (7): Sigmoid()
        )
      )

#. After each training, you will find a checkpoint file in the `models` directory

----

* option ``-r`` preprocess the dataset using NMF; it should be used only once
  per each context; each subsequent runs will use the already dumped
  dataset
* option ``-sk`` reduces the dataset to 10% of its total for pedaling and to
  5% for velocity; thus, ``-sk -r`` would result in preprocessing only that
  10% and 5%


3. Training the context-specific models
---------------------------------------

#. Fully train velocity model on the specific context: ``python run.py -v -t -c
   <context> -pt <path to generic model chekcpoint>``

#. Fully train pedaling model on the specific context: ``python run.py -p -t -c
   <context> -pt <path to generic model chekcpoint>``

#. After each training, you will find 3 checkpoints in the `models`
   directory, each corresponding to a different size of transferred
   knowledge. For each size, the procedure stops and wait for an input
   before going on with the next size of transferred layers

Here ``<context>`` is any Carla preset name that you have used before.

Results for velocity
~~~~~~~~~~~~~~~~~~~~

+-------------+---------+---------------+------------+-----------------+--------+
| context     | batches | learning rate | dummy loss | validation loss | epochs |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq0   | 13658,  | 7.32e-6       |            |  0.1335         |  20    |
|             | 1201    +               +            +-----------------+--------+
|             |         |               |            |  0.1335         |  20    |
|             |         +               +            +-----------------+--------+
|             |         |               |            |  0.1335         |  20    |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq1   | 12598,  | 7.94e-6       |            |  0.1225         |  20    |
|             | 1356    +               +            +-----------------+--------+
|             |         |               |            |  0.1225         |  20    |
|             |         +               +            +-----------------+--------+
|             |         |               |            |  0.1225         |  20    |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq2   | 13106,  | 7.63e-6       |  0.1109    |  0.1116         |  20    |
|             | 1052    +               +            +-----------------+--------+
|             |         |               |            |  0.1116         |  20    |
|             |         +               +            +-----------------+--------+
|             |         |               |            |  0.1116         |  20    |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq3   | 12568,  | 7.96e-6       |  0.1207    |  0.1208         |  20    |
|             | 1179    +               +            +-----------------+--------+
|             |         |               |            |  0.1208         |  20    |
|             |         +               +            +-----------------+--------+
|             |         |               |            |  0.1208         |  20    |
+-------------+---------+---------------+------------+-----------------+--------+
| salamander0 | 13877,  | 7.21e-6       |  0.1161    |  0.1171         |  20    |
|             | 1320    +               +            +-----------------+--------+
|             |         |               |            |  0.1171         |  20    |
|             |         +               +            +-----------------+--------+
|             |         |               |            |  0.1171         |  20    |
+-------------+---------+---------------+------------+-----------------+--------+
| salamander1 | 13227,  | 7.56e-6       |  0.1227    |  0.1242         |  20    |
|             | 1180    +               +            +-----------------+--------+
|             |         |               |            |  0.1242         |  20    |
|             |         +               +            +-----------------+--------+
|             |         |               |            |  0.1242         |  20    |
+-------------+---------+---------------+------------+-----------------+--------+

Results for pedaling
~~~~~~~~~~~~~~~~~~~~

Training batches: 120
Validation batches: 15
Learning rates: 8.33e-4

+-------------+------------+-----------------+--------+
| context     | dummy loss | validation loss | epochs |
+-------------+------------+-----------------+--------+
| pianoteq0   |            |   0.2135        |  24    |
|             +            +-----------------+--------+
|             |            |   0.2099        |  23    |
|             +            +-----------------+--------+
|             |            |   0.2097        |  23    |
+-------------+------------+-----------------+--------+
| pianoteq1   |            |   0.2333        |  500   |
|             +            +-----------------+--------+
|             |            |   0.2312        |  500   |
|             +            +-----------------+--------+
|             |            |   0.2314        |  500   |
+-------------+------------+-----------------+--------+
| pianoteq2   |            |   0.2150        |  41    |
|             +            +-----------------+--------+
|             |            |   0.2162        |  42    |
|             +            +-----------------+--------+
|             |            |   0.2136        |  20    |
+-------------+------------+-----------------+--------+
| pianoteq3   |            |   0.2052        |  22    |
|             +            +-----------------+--------+
|             |            |   0.1998        |  38    |
|             +            +-----------------+--------+
|             |            |   0.1996        |  20    |
+-------------+------------+-----------------+--------+
| salamander0 |            |   0.2374        |  24    |
|             +            +-----------------+--------+
|             |            |   0.2335        |  39    |
|             +            +-----------------+--------+
|             |            |   0.2334        |  20    |
+-------------+------------+-----------------+--------+
| salamander1 |            |   0.2086        |  45    |
|             +            +-----------------+--------+
|             |            |   0.1997        |  30    |
|             +            +-----------------+--------+
|             |            |   0.1995        |  20    |
+-------------+------------+-----------------+--------+

4. Evaluating error distributions
---------------------------------

#. Evaluate error distributions of velocity models whose checkpoints are in a
   given directory: ``python run.py -v -e <list of checkpoints> -cp``; you can
   use shell expansion like ``models/*vel*.pt``
#. Evaluate error distributions of pedaling models whose checkpoints are in a
   given directory: ``python run.py -p -e <list of checkpoints> -cp``; you can
   use shell expansion like ``models/*ped*.pt``

These commands will create a plotly plots with violin plots of generic and
specific contexts and Wilcoxon p-values.

You can plot the tests multiple times without retesting: ``python run.py -p -cp -cf
results/*.csv``.

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


=======
Credits
=======

#. `Federico Simonetta <https://federicosimonetta.eu.org>`_
