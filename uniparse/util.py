"""Utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os.path
import functools


import numpy as np
import random
import json
from six.moves import range
import six


def validate_line(line):
    """Validate UD line."""
    if line.startswith("#"):
        return False
    if line == "\n":
        return True
    # think this is the one we want
    if not re.match(r"\d+\t", line):
        return False

    return True


UD_DEFAULT_LINE = [
    "1",  # index
    "_",  # form
    "_",  # lemma
    "_",  # upos
    "_",  # xpos
    "_",  # feats
    "_",  # head
    "_",  # deprel
    "_",  # deps
    "_",  # misc
]


def convert_delimited_to_conllu(filename, delimiter="\t", *, output_filename=None):
    """
    Convert a delimited file and converts it into UD.

    This converter accepts between 1-3 delimited values, each with their mening:
        1. Form
        2. Tag
    """

    filehandler_in = open(filename, encoding="UTF-8")
    if not output_filename:
        output_filename = os.path.splitext(filename)[0] + ".conllu"

    filehandler_out = open(output_filename, "w", encoding="UTF-8")

    index = 1
    for line in filehandler_in:
        if line == "\n":
            line = ""
            index = 1
        else:
            columns = line.strip().split(delimiter)
            line_list = UD_DEFAULT_LINE.copy()
            line_list[0] = str(index)
            line_list[1] = columns[0]
            line_list[3] = columns[1] if len(columns) > 1 else line_list[3]
            line = "\t".join(line_list)
            index += 1

        print(line, file=filehandler_out)  # print empty line

    print(file=filehandler_out)

    filehandler_in.close()
    filehandler_out.close()

    return output_filename


def write_predictions_to_file(predictions, reference_file, output_file, vocab):
    if not predictions:
        raise ValueError("No predictions to write to file.")
    indices, arcs, rels = zip(*predictions)
    flat_arcs = _flat_map(arcs)
    flat_rels = _flat_map(rels)

    idx = 0
    with open(reference_file, encoding="UTF-8") as f, open(output_file, 'w', encoding="UTF-8") as fo:
        for line in f.readlines():
            if re.match(r'\d+\t', line):
                info = line.strip().split()
                assert len(info) == 10, 'Illegal line: %s' % line
                info[6] = str(flat_arcs[idx])
                info[7] = vocab.id2rel(flat_rels[idx])
                fo.write('\t'.join(info) + '\n')
                idx += 1
            else:
                fo.write(line)


def _flat_map(lst):
    return functools.reduce(lambda x, y: x + y, [list(result) for result in lst])

# -*- coding: utf-8 -*-
"""Utilities for preprocessing sequence data. [TAKEN FROM KERAS]"""



def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
