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
   and loss function of 0.1074 (about like the dummy predictor but there is
   the complexity cost!).
#. Look for hyper-parameters for pedaling using the original context: ``python
   run.py -p -sk -c orig``. We obtained hyperparams defined in ``settings.py``
   and loss function of 0.2189.
#. Fully train velocity model on the original context: ``python run.py -v -t -c orig``

   * Dummy loss: 0.1207
   * Validation loss:0.1210 (62 epochs, early-stop)
   * 354845 in training
   * 55008 batches in validation
   * Learning rate: 1.41e-05
   * 32 parameters::

      MIDIParameterEstimation(
        (dropout): Dropout(p=0.1, inplace=False)
        (stack): Sequential(
          (0): Conv2d(1, 1, kernel_size=(3, 5), stride=(1, 1), bias=False)
          (1): Identity()
          (2): Tanh()
          (3): Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (4): Identity()
          (5): Tanh()
          (6): Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (7): Identity()
          (8): Tanh()
          (9): Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (10): Identity()
          (11): Tanh()
          (12): Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (13): Identity()
          (14): Tanh()
          (15): Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
          (16): Identity()
          (17): Tanh()
          (18): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (19): Sigmoid()
        )
      )

#. Fully train pedaling model on the original context: ``python run.py -p -t -c orig``

   * Dummy loss: 0.2578
   * Validation loss: 0.2016 (142 epochs with early-stop)
   * 247 batches in training
   * 47 batches in validation
   * Learning rate: 2.02e-2
   * 69 parameters::

      MIDIParameterEstimation(
        (dropout): Dropout(p=0.1, inplace=False)
        (stack): Sequential(
          (0): Conv2d(3, 3, kernel_size=(6, 1), stride=(1, 1), groups=3, bias=False)
          (1): InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Identity()
          (3): Conv2d(3, 3, kernel_size=(6, 1), stride=(1, 1), groups=3, bias=False)
          (4): InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): Identity()
          (6): Conv2d(3, 3, kernel_size=(3, 1), stride=(1, 1), groups=3, bias=False)
          (7): InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (8): Identity()
          (9): Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), groups=3)
          (10): Sigmoid()
        )
      )

#. After each training, you will find a checkpoint file in the `models` directory

----

* option ``-r`` preprocess the dataset using NMF; it should be used only once
  per each type of model; each subsequent runs will use the already dumped
  dataset
* option ``-sk`` reduces the dataset to 10% of its total for pedaling and to
  3% for velocity; thus, ``-sk -r`` would result in preprocessing only that
  10% and 3%


3. Training the context-specific models
---------------------------------------

#. Apply NMF to each context: ``python run.py -p -r -c <context>``, ``python
   run.py -v -r -c <context>``

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
| pianoteq0   | 1366,   |  3.66e-3      |   0.13372  |      0.1324     |   64   |
|             | 121     +---------------+            +-----------------+--------+
|             |         |  6.89e-3      |            |      0.1340     |   86   |
|             |         +---------------+            +-----------------+--------+
|             |         |  2.34e-2      |            |      0.1336     |   34   |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq1   | 1260,   |  3.97e-3      |   0.12367  |      0.1220     |   53   |
|             | 136     +---------------+            +-----------------+--------+
|             |         |  7.47e-3      |            |      0.1227     |   29   |
|             |         +---------------+            +-----------------+--------+
|             |         |  2.54e-2      |            |      0.1224     |   30   |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq2   | 1311,   |  3.81e-3      |   0.11159  |      0.1119     |   26   |
|             | 106     +---------------+            +-----------------+--------+
|             |         |  7.18e-3      |            |      0.1115     |   38   |
|             |         +---------------+            +-----------------+--------+
|             |         |  2.44e-2      |            |      0.1112     |   38   |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq3   | 1257,   |  3.40e-3      |   0.12067  |      0.1207     |   72   |
|             | 118     +---------------+            +-----------------+--------+
|             |         |  7.49e-3      |            |      0.1206     |   23   |
|             |         +---------------+            +-----------------+--------+
|             |         |  2.55e-2      |            |      0.1203     |   34   |
+-------------+---------+---------------+------------+-----------------+--------+
| salamander0 | 1388,   |  3.60e-3      |   0.11713  |      0.1165     |   23   |
|             | 132     +---------------+            +-----------------+--------+
|             |         |  6.68e-3      |            |      0.1164     |   64   |
|             |         +---------------+            +-----------------+--------+
|             |         |  2.31e-2      |            |      0.1160     |   31   |
+-------------+---------+---------------+------------+-----------------+--------+
| salamander1 | 1323,   |  3.78e-3      |   0.12268  |      0.1233     |   30   |
|             | 118     +---------------+            +-----------------+--------+
|             |         |  7.11e-3      |            |      0.1230     |   32   |
|             |         +---------------+            +-----------------+--------+
|             |         |  2.42e-2      |            |      0.1229     |   31   |
+-------------+---------+---------------+------------+-----------------+--------+

Results for pedaling
~~~~~~~~~~~~~~~~~~~~

Training batches: 120
Validation batches: 15
Learning rates: 4.17e-2, 6.39-2, 1.37e-1

+-------------+------------+-----------------+--------+
| context     | dummy loss | validation loss | epochs |
+-------------+------------+-----------------+--------+
| pianoteq0   |   0.26464  |     0.2272      |  124   |
|             +            +-----------------+--------+
|             |            |     0.2225      |  116   |
|             +            +-----------------+--------+
|             |            |     0.2269      |  84    |
+-------------+------------+-----------------+--------+
| pianoteq1   |   0.27372  |     0.2331      |  142   |
|             +            +-----------------+--------+
|             |            |     0.2327      |  87    |
|             +            +-----------------+--------+
|             |            |     0.2399      |  66    |
+-------------+------------+-----------------+--------+
| pianoteq2   |   0.25987  |     0.2160      |  110   |
|             +            +-----------------+--------+
|             |            |     0.2170      |  92    |
|             +            +-----------------+--------+
|             |            |     0.2191      |  65    |
+-------------+------------+-----------------+--------+
| pianoteq3   |   0.26856  |     0.2205      |  182   |
|             +            +-----------------+--------+
|             |            |     0.2246      |  140   |
|             +            +-----------------+--------+
|             |            |     0.2358      |  82    |
+-------------+------------+-----------------+--------+
| salamander0 |   0.26678  |     0.2402      |  145   |
|             +            +-----------------+--------+
|             |            |     0.2413      |  125   |
|             +            +-----------------+--------+
|             |            |     0.2409      |  70    |
+-------------+------------+-----------------+--------+
| salamander1 |   0.25548  |     0.2100      |  174   |
|             +            +-----------------+--------+
|             |            |     0.2100      |  131   |
|             +            +-----------------+--------+
|             |            |     0.2114      |  81    |
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
