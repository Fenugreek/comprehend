Learning with Artifical Neural Networks
===================

Implementation of neural network architectures using the Google TensorFlow machine-learning library.

Autoencoders, denoising autoencoders, RBMs, Conv nets and RNNs are implemented, along with their training in both unsupervised and supervised contexts. 

### Notable modules:
- **networks** : Implements the various NN architectures. 

- **layers**: Implements multiple layers of architectures defined in **networks**.

### Test script:
- **run.py** : Train the specified network on data in given filenames (MNIST by default), (optionally) saving params learnt to disk. (Optionally) display features learnt, and if MNIST, results of reconstruction on corrupted input.

Installation
------------
From the directory where you downloaded the files, run the following command-line to install the library:

```
 $ python setup.py install
```

Or simply place all files into a directory called `comprehend/` somewhere in your `$PYTHONPATH`.

**Dependencies** : [TensorFlow](https://www.tensorflow.org/) and its dependencies, [matplotlib](http://matplotlib.org/),
[Fenugreek/tamarind](https://github.com/Fenugreek/tamarind), and [dmishin/tsp-solver](https://github.com/dmishin/tsp-solver).

Usage
---------------

```
$ python run.py --help
usage: run.py [-h] [--model <model>] [--module <module>]
              [--fromfile <filename.dat>] [--add <model>] [--visible N]
              [--hidden N] [--options <option>=<integer>[,<integer>...]
              [<option>=<integer>[,<integer>...] ...]]
              [--data <filename.blp> [<filename.blp> ...]]
              [--labels <filename.blp>] [--targets <filename.blp>]
              [--permute <filename.blp>] [--prep_data <filename.blp>]
              [--validation R] [--batch N] [--bptt N] [--epochs N]
              [--learning R] [--random_seed N] [--output <path/prefix>]
              [--loss] [--dump {hidden|recode|<custom method>}] [--features]
              [--log log_level]

Load/construct various NN models from the commandline, train/run them on data,
save to disk.

optional arguments:
  -h, --help            show this help message and exit
  --model <model>       network architecture to load from networks module.
                        e.g. Auto, Denoising, RBM, RNN, VAE.
  --module <module>     find <model> in this module (defaults to module
                        supplied with this library).
  --fromfile <filename.dat>
                        previous params.dat file to load from and resume
                        training.
  --add <model>         network layer to add to architecture loaded fromfile.
  --visible N           number of visible units; inferred from data if not
                        specified
  --hidden N            number of hidden units
  --options <option>=<integer>[,<integer>...] [<option>=<integer>[,<integer>...] ...]
                        options to pass to constructor of network
                        architecture; only integers supported for now.
  --data <filename.blp> [<filename.blp> ...]
                        data file(s) to use for training. If multiple, they
                        are joined.
  --labels <filename.blp>
                        data file with labels for classification training.
  --targets <filename.blp>
                        data file with targets for mapping training.
  --permute <filename.blp>
                        Permute order of rows in input data according to
                        indices in this file.
  --prep_data <filename.blp>
                        Supply optional auxiliary file here for post-
                        processing data by NN object's prep_data method, if
                        any.
  --validation R        fraction of dataset to use as validation set.
  --batch N             size of each mini-batch
  --bptt N              backpropagation through time; no. of timesteps
  --epochs N            No. of epochs to train.
  --learning R          learning rate for gradient descent algorithm
  --random_seed N       Seed random number generator with this, for repeatable
                        results.
  --output <path/prefix>
                        output params and figures to
                        <path/prefix>{params,features,mosaic}.dat.
  --loss                Print r.m.s. loss on validation data.
  --dump {hidden|recode|<custom method>}
                        dump computed hidden or recoded values to disk. Needs
                        --output to be specified also.
  --features            Print image visualization of weights to disk.
  --log log_level       debug, info, warning, error or critical.
```

Sample command-line and output:

```
$ python run.py --model RBM --epochs 8  --hidden 500 --batch 40 --features --output rbm_8_

[INFO  run.py 11:22:02] Initial cost 0.7152 r.m.s. loss 0.4912
[INFO  run.py 11:23:02] Training epoch 0 cost 0.1182 r.m.s. loss 0.1277
[INFO  run.py 11:24:01] Training epoch 1 cost 0.1004 r.m.s. loss 0.1047
[INFO  run.py 11:25:00] Training epoch 2 cost 0.0928 r.m.s. loss 0.0939
[INFO  run.py 11:25:59] Training epoch 3 cost 0.0888 r.m.s. loss 0.0879
[INFO  run.py 11:26:59] Training epoch 4 cost 0.0862 r.m.s. loss 0.0837
[INFO  run.py 11:27:58] Training epoch 5 cost 0.0844 r.m.s. loss 0.0807
[INFO  run.py 11:28:57] Training epoch 6 cost 0.0833 r.m.s. loss 0.0789
[INFO  run.py 11:29:56] Training epoch 7 cost 0.0822 r.m.s. loss 0.0770
```

This will create a set of files beginning with the prefix `rbm_8_`.

- `rbm_8_params.dat`:
This file has the weights learned, and can be used to continue training where it was left off by using `--params rbm_8_params.dat` option when executing `run.py`.

- `rbm_8_features.png, rbm_8_mosaic.png`:
These are visualizations of the features learnt, and performance of the model on validation input. They look as follows:

![Visualization of features learnt](http://www.subburam.org/files/features.png "Heatmap of node weights")
![MNIST image reconstruction test output](http://www.subburam.org/files/mosaic.png "Input (top half) output (bottom half) test")
