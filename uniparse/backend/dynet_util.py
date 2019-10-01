"""
    High level abstraction for dense forward pass ala. Keras. Followed by helper functions.
"""

import dynet as dy
import numpy as np


class Dense(object):
    def __init__(self, model_parameters, input_dim: int, hidden_dim: int, activation, bias: bool):
        self.activation = activation
        self.bias = bias
        self.w = model_parameters.add_parameters((hidden_dim, input_dim))
        self.b = model_parameters.add_parameters((hidden_dim,)) if bias else None

    def __call__(self, inputs):
        """ todo """
        output = dy.parameter(self.w) * inputs
        if self.bias:
            output = output + dy.parameter(self.b)
        if self.activation:
            return self.activation(output)

        return output


def _attend(s, axis):
    e = dy.exp(s - dy.max_dim(s, axis))
    s = dy.sum_dim(e, [axis])
    r = dy.cdiv(e, s + 1e-6)
    return r


def softalign(mods, heads):
    score_matrix = mods * dy.transpose(heads)

    mod_with_respect_to_head = _attend(score_matrix, 0)
    mods_hat = mod_with_respect_to_head * mods

    head_with_respect_to_mod = _attend(score_matrix, 1)
    heads_hat = head_with_respect_to_mod * heads

    return mods_hat, heads_hat


def saxe(shape, gain=1):
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q[:shape[0], :shape[1]]


def init_birnn_orthogonal(bilstm):
    def init_rnn(lstm):
        for kernel, recurrent_kernel, o in lstm.get_parameters():
            for e in [kernel, recurrent_kernel]:
                shape = e.as_array().shape
                input_dim, units = shape
                shape = (input_dim // 4, units * 4)
                kernel = saxe(shape)

                kernel_i = kernel[:, :units]
                kernel_f = kernel[:, units: units * 2]
                kernel_c = kernel[:, units * 2: units * 3]
                kernel_o = kernel[:, units * 3:]

                w = np.concatenate([kernel_i, kernel_f, kernel_c, kernel_o], axis=0)
                e.set_value(w)

            d = o.as_array().shape[0]
            e = np.zeros(d, dtype=np.float64)
            fourth = d // 4
            e[fourth: 2 * fourth] = -1
            o.set_value(e)

        return lstm

    for forward, backward in bilstm.builder_layers:
        init_rnn(forward)
        init_rnn(backward)
