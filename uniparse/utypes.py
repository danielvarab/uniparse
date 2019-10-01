"""(uniparse)Types module for inheritence in UniParse."""

import numpy as np


class Parser(object):
    """Parser abstract class"."""

    _decoder = None
    _backend = None
    backend_name = "dynet"  # defaults to dynet

    def set_decoder(self, decoder):
        self._decoder = decoder

    def set_backend(self, backend):
        self._backend = backend

    def get_backend_name(self):
        return self.backend_name

    def decode(self, arc_scores, clip=None):
        decode = self._decoder
        arc_scores = self._backend.to_numpy(arc_scores)

        if clip is not None:
            batch_size, n, _ = arc_scores.shape
            result = np.zeros((batch_size, n), dtype=np.int)  # batch, n
            for i in range(batch_size):
                i_len = clip[i]
                tree = decode(arc_scores[i, :i_len, :i_len])
                result[i, :i_len] = tree
        else:
            result = np.array([decode(s) for s in arc_scores])

        return result

    def save_to_file(self, filename: str) -> None:
        raise NotImplementedError("You need to implement the save procedure your self")

    def load_from_file(self, filename: str) -> None:
        raise NotImplementedError("You need to implement the load procedure your self")

    def __call__(self, x):
        raise NotImplementedError(
            "You need to implement the forward pas procedure your self"
        )
