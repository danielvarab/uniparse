"""PyTorch implementation of Kiperwasser and Goldenberg (2016)."""

from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn

from uniparse.utypes import Parser


class BiRNN(nn.Module):
    """BiLSTM pytorch implementation."""
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        """BiLSTM forward pass."""
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        return out


class DependencyParser(nn.Module, Parser):
    """Class to encapsulate model parameters."""
    def save_to_file(self, filename):
        torch.save(self.state_dict(), filename)

    def load_from_file(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename))

    def __init__(self, vocab):
        super().__init__()

        upos_dim = 25
        word_dim = 100
        hidden_dim = 100
        bilstm_out = (word_dim+upos_dim) * 2

        self.word_count = vocab.vocab_size
        self.upos_count = vocab.upos_size
        self.i2c = defaultdict(int, vocab.wordid2freq)
        self.label_count = vocab.label_count
        self._vocab = vocab

        self.wlookup = nn.Embedding(self.word_count, word_dim)
        self.tlookup = nn.Embedding(self.word_count, upos_dim)

        self.deep_bilstm = BiRNN(word_dim+upos_dim, word_dim+upos_dim, 2)

        # edge encoding
        self.edge_head = nn.Linear(bilstm_out, hidden_dim)
        self.edge_modi = nn.Linear(bilstm_out, hidden_dim, bias=True)

        # edge scoring
        self.e_scorer = nn.Linear(hidden_dim, 1, bias=True)

        # rel encoding
        self.label_head = nn.Linear(bilstm_out, hidden_dim)
        self.label_modi = nn.Linear(bilstm_out, hidden_dim, bias=True)

        # label scoring
        self.l_scorer = nn.Linear(hidden_dim, vocab.label_count, bias=True)

    def get_backend_name(self):
        return "pytorch"
        # return "dynet"

    @staticmethod
    def _propability_map(matrix, dictionary):
        return np.vectorize(dictionary.__getitem__)(matrix)

    def forward(self, x):
        word_ids, lemma_ids, upos_ids, target_arcs, rel_targets, chars = x

        batch_size, n = word_ids.shape

        is_train = target_arcs is not None

        # if is_train:
        #     # Frequency word dropout. replace with UNK / OOV
        #     c = self._propability_map(word_ids, self.i2c)
        #     drop_mask = np.greater(0.25/(c+0.25), np.random.rand(*word_ids.shape))
        #     word_ids = np.where(drop_mask, self._vocab.OOV, word_ids)

        word_embs = self.wlookup(word_ids)
        upos_embs = self.tlookup(upos_ids)

        words = torch.cat([word_embs, upos_embs], dim=-1)

        word_exprs = self.deep_bilstm(words)

        word_h = self.edge_head(word_exprs)
        word_m = self.edge_modi(word_exprs)

        arc_score_list = []
        for i in range(n):
            modifier_i = word_h[:, i, None, :] + word_m  # we would like have head major
            modifier_i = torch.tanh(modifier_i)
            modifier_i_scores = self.e_scorer(modifier_i)
            arc_score_list.append(modifier_i_scores)

        arc_scores = torch.stack(arc_score_list, dim=1)
        arc_scores = arc_scores.view(batch_size, n, n)

        # Loss augmented inference
        if is_train:
            # Root dep guy contains negatives.. watch out for that.
            target_arcs[:, 0] = 0

            margin = np.ones((batch_size, n, n))
            for bi in range(batch_size):
                for m in range(n):
                    h = target_arcs[bi, m]
                    margin[bi, m, h] -= 1
            arc_scores = arc_scores + torch.Tensor(margin)

        # since we are major
        parsed_trees = self.decode(arc_scores.transpose(1, 2))

        tree_for_rels = target_arcs if is_train else parsed_trees
        tree_for_rels[:, 0] = 0
        batch_indicies = np.repeat(np.arange(batch_size), n)
        pred_tree_tensor = tree_for_rels.reshape(-1)

        rel_heads = word_exprs[batch_indicies, pred_tree_tensor, :]
        rel_heads = self.label_head(rel_heads).view((batch_size, n, -1))
        rel_modifiers = self.label_modi(word_exprs)

        rel_arcs = torch.tanh(rel_modifiers + rel_heads)

        rel_scores = self.l_scorer(rel_arcs)
        predicted_rels = rel_scores.argmax(-1).data.numpy()

        return parsed_trees, predicted_rels, arc_scores, rel_scores

