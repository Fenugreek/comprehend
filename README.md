Unsupervised Learning with Artifical Neural Networks
===================

Implementation of neural network architectures using the Google TensorFlow machine-learning library, primarily for unsupervised learning.

Currently, autoencoders, denoising autoencoders, RBMs, and RNNs are implemented, along with their training on MNIST. 

### Notable modules:
- **networks** : Implements the various NN architectures. 

### Test script:
- **test.py** : Train the network specified on MNIST data, (optionally) saving params learnt to disk. (Optionally) display features learnt, and results of reconstruction on corrupted input.

Installation
------------
From the directory where you downloaded the files, run the following command-line to install the library:

```
 $ python setup.py install
```

Or simply place all files into a directory called `comprehend/` somewhere in your `$PYTHONPATH`.

**Dependencies** : [TensorFlow](https://www.tensorflow.org/) and its dependencies, [matplotlib](http://matplotlib.org/), [Fenugreek/tamarind](https://github.com/Fenugreek/tamarind), and [dmishin/tsp-solver](https://github.com/dmishin/tsp-solver).

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
  --model <model>       network architecture to load from networks module.
                        Auto, Denoising, RBM or RNN.
  --params <filename.dat>
                        previous params_final.dat file to load from and resume
                        training.
  --data <filename.dat>
                        data file to use for training. Default: MNIST
  --output <path/prefix>
                        output params and figures to
                        <path/prefix>{params,features,mosaic}.dat.
  --learning_rate R     learning rate for gradient descent algorithm
  --batch N             size of each mini-batch
  --hidden N            number of hidden units
  --epochs N            No. of epochs to train.
  --random_seed N       Seed random number generator with this, for repeatable
                        results.
  --use_tsp             Use Traveling Salesman Problem solver when arranging
                        features for display (takes time).
  --mosaic              Display learnt model's reconstruction of corrupted
                        input.
  --verbose             print progress
```

Sample command-line:

```
$ python source/test.py --model RBM --epochs 8 --learning_rate 0.05 --verbose --hidden 500 --batch 40 --mosaic
```
