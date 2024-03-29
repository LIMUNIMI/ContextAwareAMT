===========================================
Context-aware Automatic Music Transcription
===========================================

**Supplementary information are available on the** `MIA website <https://limunimi.github.io/MIA-Music-Interpretation-Analysis>`_.

Setup
-----

#. Enter the root git directory
#. Install ``poetry``
#. Install ``pyenv``
#. Install python 3.8.6: ``pyenv install 3.8.6``
#. Create a new venv with poetry and install the dependencies: ``poetry install --noroot``
#. Start a new shell in this venv: ``poetry shell``
#. Check that the version is correct with `python --version`

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

Command-line options
--------------------

For all the available commands, using `-v` allows to extract velocity data,
while using `-p` refers to pedaling.

Other options are described below.

1. Preprocess
-------------

#. Create the MIDI file for the template, synthesize and 
   compute the template: ``python run.py -sc``
#. Apply NMF and extract notes for [velocity|pedaling] estimation: ``python run.py [-v|-p] -r``

2. Training the models
----------------------

#. Evaluate [velocity|pedaling] configurations using: ``python run.py [-v|-p] -sk``.
#. Option `-cm` cleans MLFlow runs, use it if the previous command fails for
   some reason, because the final evaluation is based on MLFlow
  
3. Evaluation
-------------

Run `python run.py -e [-v|-p]` to evaluate the average L1 error in each configuration

----

The previous steps can be done in a single command: ``python run.py [-v|-p] -sc -r -sk -e``

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

Optional
~~~~~~~~~~

Fully train models with: ``-t`` option. After each training, you will find a
few checkpoint files in the relative directory directory; find the most recent
one with ``ls -t | head -1``. The hyperparameters used are not the best ones
(you can change them in `settings.py`) and models are trained using multiple
performers. Context specificity can be turned on with `-cs` option.


=======
Credits
=======

#. `Federico Simonetta <https://federicosimonetta.eu.org>`_
