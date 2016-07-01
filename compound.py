"""
Coumpound NN architectures, built from the fundamental types in networks module.
"""
import tensorflow as tf
import networks, train
import numpy

class RNNRBM(networks.RNN):
    """Recurrent neural network with an RBM on top of the hidden layer."""

    def __init__(self, rnn, rbm, verbose=False):

        self.rnn = rnn
        self.rbm = rbm
        self.verbose = verbose
        self.autoencode = False
        self.clean_energy = None

    def update(self, states, inputs):
        """
        Update states given old states and new inputs, autoencoding states.
        Return new states and outputs corresponding to inputs.
        """

        params = self.rnn.params
        operand = tf.matmul(inputs, params['Wxh']) + params['bhid']
        if states is not None: operand += tf.matmul(states, params['Whh'])

        states = tf.sigmoid(operand)
        if self.autoencode: states = self.rbm.recode(states)
        
        outputs = tf.sigmoid(tf.matmul(states, params['Why']) + params['bvis'])

        return states, outputs
    

    def recode(self, inputs):

        if self.clean_energy is None:
            return networks.RNN.recode(self, inputs)
        
        outputs, states = \
           self.get_outputs(inputs, None, states=[], outputs=[], as_list=True)

        states = tf.transpose(tf.pack(states), perm=[1, 0, 2])
        #Gradient descent to match clean energy.
        states = train.descent(states.eval().reshape((-1, self.rbm.n_visible)),
                   lambda x: self.rbm.free_energy(x).eval(),
                   self.rbm.grad_free_energy,
                   target_objective=self.clean_energy,
                   verbose=self.verbose, adaptive_rate=.01)
        outputs = tf.sigmoid(tf.matmul(states, self.rnn.params['Why']) +
                                     self.rnn.params['bvis'])
        return tf.transpose(tf.reshape(outputs,
                                       (-1, self.rnn.seq_length, self.rnn.n_visible)),
                            perm=[0, 2, 1])
#        return outputs.reshape((-1, self.rnn.seq_length, self.rnn.n_visible)).swapaxes(1, 2)
