=============================================
Mapping Music Performances Context to Context
=============================================

Setup
-----

#. Enter the root git directory
#. Install ``poetry``
#. Install ``pyenv``
#. Install python 3.9.0: ``pyenv install 3.9.0``
#. Activate it: ``pyenv shell 3.9.0``
#. Create a new venv with poetry and install the dependencies: ``poetry update``
#. Start a new shell in this venv: ``poetry shell``

Config
------

#. Open ``mpc2c/settings.py`` and check the variables in section 'PATH',
   especially ``VELOCITY_DATA_PATH``, ``PEDALING_DATA_PATH``, and
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
#. If the process stops, re-launch it (it will skip the already synthesized songs)
#. If it fails, it's probably because Carla crashed; just stop
   (CTRL+C) and restart (it will skip already synthesized songs); try to
   ``killall -9 jackd`` before restarting the process.
#. After having synthesized , you can do a full check that everything has
   correctly been synthesized by ``rm asmd_resynth.txt`` and relaunching the
   process ``python run.py -d``
#. The datasets were split using PCA and retaining 0.89, 0.93, 0.91 of total
   variance for `train`, `validation` and `test` set respectively.

1. Preprocess
-------------

#. Create the MIDI file for the initial template: ``python run.py -sc``
#. Synthesize the midi scale and name it ``pianoteq_scales.mp3`` (TODO: resynthesize using pycarla)
#. Compute the initial template and save it to file: ``python run.py --template``
#. Apply NMF and extract notes for velocity estimation: ``python run.py -v -r``
#. Apply NMF and extract frames for pedaling estimation: ``python run.py -p -r``
#. You can restrict to each single context by using option ``-c``

2. Training the models
----------------------

N.B. TODO

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


4. Evaluating error distributions
---------------------------------

N.B. TODO

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

Notes
-----

We used 6 different artificial contexts generated from Pianoteq LV2 plugin.
The following table shows the differences among the 6 contexts:

+-----------+--------------+---------------+---------------------+
|  Context  | Velocity Map |    Reverb     |     Instrument      |
+-----------+--------------+---------------+---------------------+
| pianoteq0 |    Linear    |  Jazz Studio  |  Steinway B Prelude |
+-----------+--------------+---------------+---------------------+
| pianoteq1 | Logarithmic  |  Jazz Studio  |  Steinway B Prelude |
+-----------+--------------+---------------+---------------------+
| pianoteq2 | Logarithmic  |   Cathedral   |  Steinway B Prelude |
+-----------+--------------+---------------+---------------------+
| pianoteq3 |    Linear    |  Jazz Studio  |  Grotrian Cabaret   |
+-----------+--------------+---------------+---------------------+
| pianoteq4 | Logarithmic  |  Jazz Studio  |  Grotrian Cabaret   |
+-----------+--------------+---------------+---------------------+
| pianoteq5 | Logarithmic  |   Cathedral   |  Grotrian Cabaret   |
+-----------+--------------+---------------+---------------------+

The Carla project files available in the repo allow to see each individual
parameter of the contexts.

=======
Credits
=======

#. `Federico Simonetta <https://federicosimonetta.eu.org>`_
