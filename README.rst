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

2. Training the generic model
-----------------------------

#. Look for hyper-parameters for velocity using the original context: ``python
   run.py -v -sk``. We obtained hyperparams defined in ``settings.py``
   and loss function of TODO (about like the dummy predictor but there is
   the complexity cost!).
#. Look for hyper-parameters for pedaling using the original context: ``python
   run.py -p -sk``. We obtained hyperparams defined in ``settings.py``
   and loss function of TODO.
#. Fully train velocity model on the original context: ``python run.py -v -t -c orig``

   * Dummy loss: TODO
   * Validation loss: TODO (TODO epochs)
   * TODO batches in training
   * TODO batches in validation
   * Learning rate: TODO
   * TODO parameters::

     TODO

#. Fully train pedaling model on the original context: ``python run.py -p -t -c orig``

   * Dummy loss: TODO
   * Validation loss: TODO (TODO epochs)
   * TODO batches in training
   * TODO batches in validation
   * Learning rate: TODO
   * TODO parameters::

     TODO

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
| pianoteq0   |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq1   |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq2   |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
+-------------+---------+---------------+------------+-----------------+--------+
| pianoteq3   |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
+-------------+---------+---------------+------------+-----------------+--------+
| salamander0 |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
+-------------+---------+---------------+------------+-----------------+--------+
| salamander1 |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
|             |         +---------------+            +-----------------+--------+
|             |         |               |            |                 |        |
+-------------+---------+---------------+------------+-----------------+--------+

Results for pedaling
~~~~~~~~~~~~~~~~~~~~

Training batches: 120
Validation batches: 15
Learning rates: TODO

+-------------+------------+-----------------+--------+
| context     | dummy loss | validation loss | epochs |
+-------------+------------+-----------------+--------+
| pianoteq0   |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
+-------------+------------+-----------------+--------+
| pianoteq1   |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
+-------------+------------+-----------------+--------+
| pianoteq2   |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
+-------------+------------+-----------------+--------+
| pianoteq3   |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
+-------------+------------+-----------------+--------+
| salamander0 |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
+-------------+------------+-----------------+--------+
| salamander1 |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
|             +            +-----------------+--------+
|             |            |                 |        |
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
