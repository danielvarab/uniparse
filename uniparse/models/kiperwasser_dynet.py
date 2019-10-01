from collections import defaultdict

import dynet as dy
import numpy as np

from uniparse.utypes import Parser


def Dense(model_parameters, input_dim, hidden_dim, activation, use_bias):
    """Dense layer without dropout as by Kiperwasser and Goldberg (2016)."""
    w = model_parameters.add_parameters((hidden_dim, input_dim))
    b = model_parameters.add_parameters((hidden_dim,)) if use_bias else None

    def call(inputs):
        """ todo """
        output = w * inputs
        if use_bias:
            output = output + b
        if activation:
            return activation(output)

        return output

    return call


class DependencyParser(Parser):
    """Kiperwasser and Goldbergs (2016) bilstm parser paper."""

    def __init__(self, vocab, embs):
        params = dy.ParameterCollection()

        upos_dim = 25
        word_dim = 100
        hidden_dim = 100
        bilstm_out = (word_dim + upos_dim) * 2  # 250


        self.word_count = vocab.vocab_size
        self.upos_count = vocab.upos_size
        self.i2c = defaultdict(int, vocab._id2freq)
        self.label_count = vocab.label_count
        self._vocab = vocab

        if embs is not None:
            self.embs = True
            word_dim = embs.shape[1]  # replacing default 100 dim size
            bilstm_out = (word_dim + upos_dim) * 2  # 250
            self.wlookup = params.add_lookup_parameters(embs.shape)
            self.wlookup.init_from_array(embs)
            if no_update_embeddings:
                self.wlookup.set_updated(False)
        else:
            self.embs = False
            self.wlookup = params.add_lookup_parameters((self.word_count, word_dim))
        self.tlookup = params.add_lookup_parameters((self.upos_count, upos_dim))

        self.deep_bilstm = dy.BiRNNBuilder(
            2, word_dim + upos_dim, bilstm_out, params, dy.VanillaLSTMBuilder
        )

        # edge encoding
        self.edge_head = Dense(
            params, bilstm_out, hidden_dim, activation=None, use_bias=False
        )
        self.edge_modi = Dense(
            params, bilstm_out, hidden_dim, activation=None, use_bias=False
        )
        self.edge_bias = params.add_parameters((hidden_dim,))

        # edge scoring
        self.e_scorer = Dense(params, hidden_dim, 1, activation=None, use_bias=True)

        # rel encoding
        self.label_head = Dense(
            params, bilstm_out, hidden_dim, activation=None, use_bias=False
        )
        self.label_modi = Dense(
            params, bilstm_out, hidden_dim, activation=None, use_bias=False
        )
        self.label_bias = params.add_parameters((hidden_dim,))

        # label scoring
        self.l_scorer = Dense(
            params, hidden_dim, self.label_count, activation=None, use_bias=True
        )
        self.params = params

    def parameters(self):
        return self.params

    def save_to_file(self, filename):
        self.params.save(filename)

    def load_from_file(self, filename):
        self.params.populate(filename)

    @staticmethod
    def _propability_map(matrix, dictionary):
        return np.vectorize(dictionary.__getitem__)(matrix)

    def __call__(self, x):
        word_ids, lemma_ids, upos_ids, gold_arcs, gold_rels, chars = x
        return self.run(word_ids, upos_ids, gold_arcs, gold_rels)

    def run(self, word_ids, upos_ids, target_arcs, rel_targets):
        batch_size, n = word_ids.shape

        training_mode = target_arcs is not None

        n = word_ids.shape[-1]
        if training_mode:
            c = self._propability_map(word_ids, self.i2c)
            drop_mask = np.greater(0.25 / (c + 0.25), np.random.rand(*word_ids.shape))
            word_ids = np.where(drop_mask, self._vocab.OOV, word_ids)

        word_embs = [dy.lookup_batch(self.wlookup, word_ids[:, i]) for i in range(n)]
        upos_embs = [dy.lookup_batch(self.tlookup, upos_ids[:, i]) for i in range(n)]
        words = [dy.concatenate([w, p]) for w, p in zip(word_embs, upos_embs)]
        word_exprs = self.deep_bilstm.transduce(words)

        word_h = [self.edge_head(word) for word in word_exprs]
        word_m = [self.edge_modi(word) for word in word_exprs]

        arc_edges = [
            dy.tanh(word_h[head] + word_m[modifier] + self.edge_bias.expr())
            for modifier in range(n)
            for head in range(n)
        ]

        # edges scoring
        arc_scores = [self.e_scorer(e) for e in arc_edges]
        arc_scores = dy.concatenate_cols(arc_scores)
        arc_scores = dy.reshape(arc_scores, d=(n, n), batch_size=batch_size)

        if training_mode:
            # Loss augmented inference
            margin = np.ones((n, n, batch_size))
            for bi in range(batch_size):
                for m in range(n):
                    h = target_arcs[bi, m]
                    margin[h, m, bi] -= 1

            margin_tensor = dy.inputTensor(margin, batched=True)
            arc_scores = arc_scores + margin_tensor

        # only are utilizing padding strategies
        sentence_lengths = n - np.argmax(word_ids[:, ::-1] > self._vocab.PAD, axis=1)
        parsed_tree = self.decode(arc_scores, clip=sentence_lengths)

        tree_for_rels = target_arcs if training_mode else parsed_tree

        rel_heads = [self.label_head(word) for word in word_exprs]
        rel_modifiers = [self.label_modi(word) for word in word_exprs]
        stacked = dy.concatenate_cols(rel_heads)
        # (d, n) x batch_size

        golds = []
        tree_for_rels[:, 0] = 0  # root is currently negative. mask this
        for column in tree_for_rels.T:
            m_gold = dy.pick_batch(stacked, indices=column, dim=1)
            golds.append(m_gold)

        rel_arcs = []
        for m_indx, (modifier, gold) in enumerate(zip(rel_modifiers, golds)):
            rel_arc = dy.tanh(modifier + gold + self.label_bias.expr())
            rel_arcs.append(rel_arc)

        rel_arcs = dy.concatenate_cols(rel_arcs)
        rel_scores = self.l_scorer(rel_arcs)

        predicted_rels = rel_scores.npvalue().argmax(0)
        predicted_rels = (
            predicted_rels[:, np.newaxis] if predicted_rels.ndim < 2 else predicted_rels
        )
        predicted_rels = predicted_rels.T

        if not training_mode:
            arcs, rels = [], []
            for i_tree, i_rel, i_n in zip(
                parsed_tree, predicted_rels, sentence_lengths
            ):
                arcs.append(i_tree[:i_n])
                rels.append(i_rel[:i_n])

            parsed_tree = arcs
            predicted_rels = rels

        return parsed_tree, predicted_rels, arc_scores, rel_scores
