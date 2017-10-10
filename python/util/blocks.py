"""

Functions and components that can be slotted into tensorflow models.

TODO: Write functions for various types of attention.

"""

import tensorflow as tf


def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes, 
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask



def biLSTM(inputs, dim, seq_len, name):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

    Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)

    return hidden_states, cell_states


def last_output(output, true_length):
    """
    To get the last hidden layer form a dynamically unrolled RNN.
    Input of shape (batch_size, max_seq_length, hidden_dim).

    true_length: Tensor of shape (batch_size). Such a tensor is given by the length() function.
    Output of shape (batch_size, hidden_dim).
    """
    max_length = int(output.get_shape()[1])
    length_mask = tf.expand_dims(tf.one_hot(true_length-1, max_length, on_value=1., off_value=0.), -1)
    last_output = tf.reduce_sum(tf.multiply(output, length_mask), 1)
    return last_output


def masked_softmax(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.

    Input shape: (batch_size, max_seq_length, hidden_dim). 
    mask parameter: Tensor of shape (batch_size, max_seq_length). Such a mask is given by the length() function.
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 1, keep_dims=True))) * mask
    denominator = tf.reduce_sum(numerator, 1, keep_dims=True)
    weights = tf.div(numerator, denominator)
    return weights
