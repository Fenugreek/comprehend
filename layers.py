"""
Implement multiple layers of networks objects.
"""
import cPickle
import tensorflow as tf
from comprehend import networks, features
from tamarind.functions import ielu, unit_scale
import numpy as np

class Layers(networks.Conv):
    """
    Container for a list of networks objects, each representing a layer.
    Each layer's output is fed to the next layer's input.
    """
    
    def __init__(self, coders=None, fromfile=None, batch_size=100,
                 verbose=False, **kwargs):
        """
        Provide a list of networks objects <coders>, or a saved params file.
        """
        
        self.coders = [] if coders is None else coders
        if fromfile is not None:
            is_handle = type(fromfile) == file
            save_file = fromfile if is_handle else open(fromfile)
            while True:
                try:
                    coder_class = cPickle.load(save_file)
                    self.coders.append(coder_class(fromfile=save_file,
                                                   batch_size=batch_size,
                                                   verbose=verbose, **kwargs))
                except EOFError: break
            if not is_handle: save_file.close()

        self.verbose = verbose
        self.train_args = self.coders[0].train_args
        self.input_dims = self.coders[0].input_dims
        self.output_dims = self.coders[-1].output_dims
        self.n_visible = self.coders[0].n_visible
        self.n_hidden = self.coders[-1].n_hidden
        self.dtype = self.coders[0].dtype

        if batch_size is not None:
            self.set_batch_size(batch_size)
            self.batch_size = batch_size            
        

    def pop(self):
        """Remove and return network object at the top of the layer."""
        self.output_dims = self.coders[-2].output_dims
        self.n_hidden = self.coders[-2].n_hidden
        return self.coders.pop()


    def push(self, coder):
        """Add network object to the top of the layer."""
        self.coders.append(coder)
        self.output_dims = self.coders[-1].output_dims
        self.n_hidden = self.coders[-1].n_hidden

        
    def set_batch_size(self, batch_size):
        
        for coder in self.coders:
            if hasattr(coder, 'set_batch_size'):
                coder.set_batch_size(batch_size)
        self.batch_size = batch_size            


    def input_shape(self):
        return self.coders[0].input_shape()


    def output_shape(self, **kwargs):
        return self.coders[-1].output_shape(**kwargs)


    def reset_state(self):
        for c in self.coders: c.reset_state()


    def save_params(self, tofile):

        is_handle = type(tofile) == file
        save_file = tofile if is_handle else open(tofile, 'wb')

        for coder in self.coders:
            cPickle.dump(type(coder), save_file, -1)
            coder.save_params(save_file)
            
        if not is_handle: save_file.close()


    def get_hidden_values(self, inputs, reduced=True, store=False, layer=-1, **kwargs):

        if layer < 0: layer += len(self.coders)
        coders = self.coders[:layer+1]

        values = coders[0].get_hidden_values(inputs, store=store, reduced=reduced)
        for i in range(1, len(coders)):
            if coders[i-1].output_shape(reduced=reduced) != coders[i].input_shape():
                values = tf.reshape(values, coders[i].input_shape())
            values = coders[i].get_hidden_values(values, store=store, reduced=reduced)

        return values

    
    def recode(self, inputs, layer=-1, **kwargs):

        if layer < 0: layer += len(self.coders)
        coders = self.coders[:layer+1]

        values = coders[0].get_hidden_values(inputs, store=True, reduced=True, **kwargs)
        for i in range(1, len(coders)):
            if coders[i-1].output_shape() != coders[i].input_shape():
                values = tf.reshape(values, coders[i].input_shape())
            if i != len(coders) - 1:
                values = coders[i].get_hidden_values(values, store=True, reduced=True, **kwargs)

        values = coders[-1].recode(values, **kwargs)
        for i in range(2, len(coders) + 1):
            if coders[-i+1].input_shape() != coders[-i].output_shape():
                values = tf.reshape(values, coders[-i].output_shape())
            values = coders[-i].get_reconstructed_input(values, reduced=True, **kwargs)

        return values


    def get_reconstructed_input(self, hidden, layer=-1, **kwargs):

        if layer < 0: layer += len(self.coders)
        coders = self.coders[:layer+1]

        values = coders[-1].get_reconstructed_input(hidden, reduced=True, **kwargs)
        for i in range(2, len(coders) + 1):
            if coders[-i+1].input_shape() != coders[-i].output_shape():
                values = tf.reshape(values, coders[-i].output_shape())
            values = coders[-i].get_reconstructed_input(values, reduced=True, **kwargs)

        return values


    def features(self, layer=0, *args):
        return self.coders[layer].features(*args)
    

    def recoded_features(self, inputs, layer=-1, inverse_fn=ielu):

        hidden = self.get_hidden_values(inputs, store=True, layer=layer).eval()

        bench = self.get_reconstructed_input(np.zeros_like(hidden),
                                             layer=layer).eval().squeeze()
        if inverse_fn: ibench = inverse_fn(bench)
        
        results = []
        for h in range(hidden.shape[-1]):
            hidden_h = np.zeros_like(hidden)
            hidden_h[..., h] = hidden[..., h]
            feature = self.get_reconstructed_input(hidden_h, layer=layer).eval().squeeze()
            if inverse_fn:
                iresult = inverse_fn(feature) - ibench
                results.append(self.coders[0].coding(iresult).eval())
            else:
                results.append(feature - bench)

        return np.array(results), bench

            
    def stimuli(self, layer=-1, location=[.5], corrsort=True, activation=1.0,
                static_hidden=True, overlay=None):

        if np.isscalar(location): location = [location]
        coders = self.coders
        if layer < 0: layer += len(coders)
        out_shape = coders[layer].output_shape(reduced=False)
        n_hidden = out_shape[-1]

        values = np.zeros([n_hidden] + list(out_shape[1:]),
                          dtype=self.dtype.as_numpy_dtype)

        mid_indices = [0 for j in range(len(out_shape) - 2)]
        for i in range(n_hidden):
            for loc in location:
                if len(mid_indices):
                    mid_indices[0] = int(out_shape[1] * loc)
                indices = [i] + mid_indices + [i]
                values[tuple(indices)] = activation                   

        self.set_batch_size(n_hidden)
        values = coders[layer].get_reconstructed_input(values, reduced=False, overlay=overlay,
                                                       static_hidden=static_hidden)
        for i in range(layer - 1, -1, -1):
            if coders[i].output_shape() != coders[i+1].input_shape():
                values = tf.reshape(values, coders[i].output_shape())
            values = coders[i].get_reconstructed_input(values, reduced=True, overlay=overlay,
                                                       static_hidden=static_hidden)

        values = values.eval().squeeze()
        if corrsort: return values[features.corrsort(values, use_tsp=True)]
        else: return values


    def all_stimuli(self, scale=True, concatenate=True, **kwargs):

        results = [self.stimuli(layer=l, **kwargs) for l in range(len(self.coders))]
        if scale: results = [unit_scale(r) for r in results]
        if concatenate: return np.concatenate(results)
        else: return results
        

        
def add_layer(existing, additional):
    """
    Add a network layer <additional> to existing network layer(s) <existing>,
    returning a Layers object.
    """

    if type(existing) == Layers:
        existing.push(additional)
        return existing
    else:
        return Layers([existing, additional])


def add_class_layer(coder, n_classes, coding=tf.nn.elu):
    """
    Add a classification layer to existing network.
    """

    return add_layer(coder, networks.Auto(n_visible=np.prod(coder.output_shape()[1:]),
                                          n_hidden=n_classes, verbose=coder.verbose,
                                          coding=coding, dtype=coder.dtype))


class DRBM(Layers, networks.RBM):
    """
    Deep RBM -- multiple layers. Not stacked RBMs, but a deep RBM.
    """
    
    def __init__(self, **kwargs):

        Layers.__init__(self, **kwargs)
        self.CDk = self.coders[0].CDk
        self.persistent = self.coders[0].persistent
        self.chain_end = None
        

    def free_energy(self, v):

        f = self.coders[0].free_energy(v)
        for i in range(1, len(self.coders)):
            v = self.coders[i - 1].get_hidden_values(v)
            f += self.coders[i].free_energy(v)

        return f


    def sample_h_given_v(self, v):

        for coder in self.coders:
            v = coder.sample_h_given_v(v)
        return v


    def sample_v_given_h(self, h):

        for coder in self.coders[::-1]:
            h = coder.sample_v_given_h(h)
        return h


class RNN(Layers, networks.RNN):
    """
    Multi-layer RNN -- multiple layers of RNN, where output values
    of each layer is fed to the next.
    """
    
    def recode(self, inputs, skips=None, store=False, layer=-1, **kwargs):

        if layer < 0: layer += len(self.coders)
        coders = self.coders[:layer+1]

        values = coders[0].recode(inputs, skips=skips, store=store, **kwargs)
        for i in range(1, len(coders)):
            values = coders[i].recode(values, skips=skips, store=store, **kwargs)

        return values


class RNNint(Layers, networks.RNN):
    """
    Multi-layer 'internal' RNN -- multiple layers of RNN, where hidden values
    of each layer is fed as input to the next.
    """
    
    def recode(self, inputs, skips=None, store=False, layer=-1, **kwargs):

        if layer < 0: layer += len(self.coders)
        coders = self.coders[:layer+1]

        values = inputs
        for c in coders[:-1]:
            values = c.get_hidden_values(values, state=c.state.get('hidden'),
                                         skips=skips, store=store, **kwargs)

        top = coders[-1]
        if 'hidden' in top.state:
            outputs, states = \
                top.seq_update(values, top.state['hidden'], skips=skips,
                               outputs=[top.state['output']],
                               states=[], as_list=True, store=store)
        else:            
            outputs, states = \
                top.seq_update(values, None, skips=skips,
                               outputs=[inputs[:, :, 0]],
                               states=[], as_list=True, store=store)

        return tf.transpose(tf.pack(outputs[:-1]), perm=[1, 2, 0])


    def predict_sequence(self, inputs, state, length, as_list=False):
        """
        Given initial sequence of inputs, extend by given length, feeding
        RNN outputs back to itself.

        Return full output states.
        Specify initial state via <state>. Can be None.
        """

        coders = self.coders
        
        if inputs is None:
            outputs = [coders[-1].get_output(state[-1])]
            new_states = [state[i] for i in range(len(coders))]
        else:
            
            new_states = []
            values = inputs
            for i in range(len(coders) - 1):
                state_i = None if state is None else state[i]
                new_states.append(coders[i].seq_update(values, state=state_i, states=[],
                                                       outputs=None, as_list=True))
                values = tf.transpose(tf.pack(new_states[-1]), perm=[1, 2, 0])

            outputs, states = coders[-1].seq_update(values,
                                   state=None if state is None else state[-1],
                                   outputs=[inputs[:, :, 0]], states=[], as_list=True)
            new_states.append(states)
            
            
        for index in range(1, length):
            value = outputs[-1]
            for i in range(len(coders) - 1):
                states_i = new_states[i]
                states_i.append(coders[i].update(states_i[-1], value,
                                                 output=False))
                value = states_i[-1]

            state, output = coders[-1].update(new_states[-1][-1], value)
            outputs.append(output)
            new_states[-1].append(state)
            
        if not as_list:
            outputs = tf.transpose(tf.pack(outputs), perm=[1, 2, 0])
            new_states = [tf.transpose(tf.pack(states), perm=[1, 2, 0])
                          for states in new_states]
            
        return outputs, new_states
