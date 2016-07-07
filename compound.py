"""
Coumpound NN architectures, built from the fundamental types in networks module.
"""
import tensorflow as tf
import networks, train, functions
import numpy

class RNNRBM(object):
    """Recurrent neural network with an RBM on top of the hidden layer."""

    def __init__(self, rnn, rbm, clean_energy=None, verbose=False):

        self.rnn = rnn
        self.rbm = rbm
        self.rnn.params['Why'] = self.rbm.params['W']
        self.rnn.params['bvis'] = self.rbm.params['bhid']

        self.verbose = verbose
        self.clean_energy = clean_energy
    

    def energy(self, states, outputs):
        states = tf.transpose(tf.pack(states), perm=[1, 0, 2])
        outputs = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])
        energy = self.rbm.energy(tf.reshape(states, (-1, self.rbm.n_visible)),
                                 tf.reshape(outputs, (-1, self.rnn.n_visible)))
        if self.verbose: print tf.reduce_mean(energy).eval()
        return energy


    def recode(self, inputs, max_iters=20, rate=.5):

        if self.clean_energy is None: return self.rnn.recode(inputs)
        
        outputs, states = \
           self.rnn.get_outputs(inputs, None, states=[], outputs=[], as_list=True)

        params = self.rnn.params
                                 
        rows = tf.unpack(tf.transpose(inputs, perm=[2, 0, 1]))
        iters = 0
        while iters < max_iters and \
            tf.reduce_mean(self.energy(states[3:-4], rows[4:-3])).eval() > \
                                    self.clean_energy:
            #Proceed with gradient descent to match clean energy.
            #Needs to be row by row, as subsequent rows affected by earlier rows.    
            states = []
            for i in range(len(rows)):
                operand = tf.matmul(rows[i], params['Wxh']) + params['bhid']
                if states: operand += tf.matmul(clean_state, params['Whh'])
                state = tf.sigmoid(operand)
                states.append(state)
                if i == len(rows) - 1: continue

                out = rows[i+1]
                gradient = self.rbm.grad_energy_v(out) * state * (1 - state) #presigmoid
                clean_state = tf.sigmoid(functions.logit(state) - rate * gradient)
                gradient = self.rbm.grad_energy_h(state) * out * (1 - out) #presigmoid
                rows[i+1] = tf.sigmoid(functions.logit(out) - rate * gradient)
                
            iters += 1
                    
        return tf.transpose(tf.pack(rows), perm=[1, 2, 0])



def data_to_RBM(rnn, data):
    """
    Given an RNN object, convert input data to hidden states and associated
    visible outputs. These can then be used to train an RBM on top of the RNN.
    """

    states = [tf.zeros((data.shape[0], rnn.n_hidden))]
    states.extend(rnn.get_hidden_values(data[:, :, :-1], as_list=True))
    states = rnn.get_hidden_values(data[:, :, :-1], as_list=True)
    states = tf.transpose(tf.pack(states), perm=[1, 0, 2])
    outputs = data[:, :, 1:].swapaxes(1, 2)

    return tf.reshape(states, (-1, rnn.n_hidden)), \
           outputs.reshape((-1, rnn.n_visible))
