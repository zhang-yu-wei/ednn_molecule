ednn_molecule
=============
This model is buid to use ednn to calculate the total energy of a bunch of molecules.

environment
-----------
* python2 or python3
* tensoflow
* progressbar

prepare data
------------
Use `gen-data.py` to generate data, this process will generate a __256*256__ grid with random size molecules lying in random locations. This grid will be too big for coarse model to calculate the electric energy. So there is a modifying process which will average the original grid to a __32*32__ grid, but it will still contain some important information. The output also gives you the total energy, electric energy and ising energy. These are all saved in the directory you give it. For more information, please us `python gen-data.py -h`.

Train
-----
Training process will be seperated into 2 parts. One is to calculate ising energy which is a local energy, the other one is to calculate the electric energy which represents the energy between molecules. And total energy is the total of the 2 parts. Use `ednn-c-gpus.py` to train coarse model. Use `ednn-f.py` to calculate finer model. Please use `-h` for more information about how to use them.

Test
----
Use `test.py` to show the results. By running this file, you will get the predicted results vs. real energy.

Show
----
You can get pictures of the data, the histogram of the energy distribution and the losses vs. epochs of the training.
