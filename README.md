Artifical Neural Networks
===================

Implementation of neural network architectures using the Google TensorFlow machine-learning library.

Currently, autoencoders, denoising Autoencoders, RBMs, and RNNs are implemented, along with their training on MNIST. 

### Notable modules:
- **networks** : Implements the various NN architectures. 

### Test script:
- **test.py** : Train the network specified on MNIST data, (optionally) saving params learnt to disk. Also optionally displays features learnt, and image reconstruction output.

Installation
------------
From the directory where you downloaded the files, run the following command-line to install the library:

```
 $ python setup.py install
```

Or simply place all files into a directory called `tamarind/` somewhere in your `$PYTHONPATH`.

**Dependencies** : TensorFlow and its dependencies, matplotlib, Fenugreek/tamarind, and dmishin/tsp-solver.

Usage
---------------

```
$ python test.py --help
usage: test.py [-h] [--model <model>] [--params <filename.dat>]
               [--data <filename.dat>] [--output <path/prefix>]
               [--learning_rate R] [--batch N] [--hidden N] [--epochs N]
               [--random_seed N] [--use_tsp] [--mosaic] [--verbose]

Train various TensorFlow models from the commandline, (optionally) saving
params learnt to disk.

optional arguments:
  -h, --help            show this help message and exit
  --model <model>       TensorFlow code to run. e.g. auto
  --params <filename.dat>
                        previous params_final.dat file to load from and resume
                        training.
  --data <filename.dat>
                        data file to use for training. Default: MNIST
  --output <path/prefix>
                        output params and figure to
                        <path/prefix>params_init.dat, etc.
  --learning_rate R     learning rate for gradient descent algorithm
  --batch N             size of each mini-batch
  --hidden N            number of hidden units
  --epochs N            No. of epochs to train.
  --random_seed N       Seed random number generator with this, for repeatable
                        results.
  --use_tsp             Use Traveling Salesman Problem solver when arranging
                        features for display (takes time).
  --mosaic              test learnt model on sample.
  --verbose             print progress
```

Sample command-line:

```
$ python source/test.py --model RBM --epochs 8 --learning_rate 0.05 --verbose --hidden 500 --batch 40 --mosaic
```
