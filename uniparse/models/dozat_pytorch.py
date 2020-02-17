"""TBA."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import (PackedSequence, pack_padded_sequence,
                                pad_packed_sequence, pad_sequence)

from uniparse.utypes import Parser


class DozatManning(nn.Module, Parser):
    def get_backend_name(self):
        return "pytorch"

    def save_to_file(self, filename):
        torch.save(self.state_dict(), filename)

    def load_from_file(self, filename: str):
        self.load_state_dict(torch.load(filename))
        return self

    def __init__(self, args, vocab):
        super(DozatManning, self).__init__()

        word_dim = args.n_embed
        feat_dim = args.n_feats
        # hidden_dim = 100
        # bilstm_out = (word_dim + feat_dim) * 2  # 250
        #dropout_p = 0.0 # args.dropout
        self._vocab = vocab
        self.args = args
        # the embedding layer
        # self.word_embed = nn.Embedding(num_embeddings=args.n_words,
                                    #    embedding_dim=args.n_embed)
        self.word_embed = nn.Embedding(num_embeddings=vocab.vocab_size,
                                       embedding_dim=word_dim)
        # if args.feat == 'char':
        #     self.feat_embed = CHAR_LSTM(n_chars=args.n_feats,
        #                                 n_embed=args.n_char_embed,
        #                                 n_out=args.n_embed)
        # elif args.feat == 'bert':
        #     self.feat_embed = BertEmbedding(model=args.bert_model,
        #                                     n_layers=args.n_bert_layers,
        #                                     n_out=args.n_embed)
        # else:
        # self.feat_embed = nn.Embedding(num_embeddings=args.n_feats,
        #                                embedding_dim=args.n_embed)
        self.feat_embed = nn.Embedding(num_embeddings=vocab.upos_size,
                                       embedding_dim=feat_dim)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)


        # the word-lstm layer
        #  self.lstm = BiLSTM(input_size=args.n_embed*2,
        #                    hidden_size=args.n_lstm_hidden,
        #                    num_layers=args.n_lstm_layers,
        #                    dropout=args.lstm_dropout)
        # self.lstm_dropout = SharedDropout(p=args.lstm_dropout)
        self.lstm = BiLSTM(input_size=word_dim+feat_dim,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        # self.mlp_arc_h = MLP(n_in=args.n_lstm_hidden*2,
        #                      n_hidden=args.n_mlp_arc,
        #                      dropout=args.mlp_dropout)
        # self.mlp_arc_d = MLP(n_in=args.n_lstm_hidden*2,
        #                      n_hidden=args.n_mlp_arc,
        #                      dropout=args.mlp_dropout)
        # self.mlp_rel_h = MLP(n_in=args.n_lstm_hidden*2,
        #                      n_hidden=args.n_mlp_rel,
        #                      dropout=args.mlp_dropout)
        # self.mlp_rel_d = MLP(n_in=args.n_lstm_hidden*2,
        #                      n_hidden=args.n_mlp_rel,
        #                      dropout=args.mlp_dropout)

        # # the Biaffine layers
        # self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
        #                          bias_x=True,
        #                          bias_y=False)
        # self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
        #                          n_out=args.n_rels,
        #                          bias_x=True,
        #                          bias_y=True)
        # self.pad_index = args.pad_index
        # self.unk_index = args.unk_index
        self.mlp_arc_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_rel,
                             dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_rel,
                             dropout=args.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
                                 n_out=vocab.label_count,
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = vocab.PAD
        self.unk_index = vocab.UNK

        self.label_count = vocab.label_count

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    # def forward(self, words, feats):
    def forward(self, x):
        words, lemma_ids, feats, target_arcs, rel_targets, chars = x
        # batch_size, seq_len = words.shape
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.word_embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        # if hasattr(self, 'pretrained'):
        #     word_embed += self.pretrained(words)
        # if self.args.feat == 'char':
        #     feat_embed = self.feat_embed(feats[mask])
        #     feat_embed = pad_sequence(feat_embed.split(lens.tolist()), True)
        # elif self.args.feat == 'bert':
        #     feat_embed = self.feat_embed(*feats)
        # else:
        #     feat_embed = self.feat_embed(feats)
        feat_embed = self.feat_embed(feats)

        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), dim=-1)

        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        parsed_tree = s_arc.max(2)[1].cpu().data.numpy()
        # parsed_tree = self.decode(s_arc)

        targets = target_arcs if target_arcs is not None else parsed_tree

        batch_idx = np.repeat(np.arange(batch_size), seq_len)
        modif_idx = np.tile(np.arange(seq_len), batch_size)
        # targt_idx = torch.Tensor(targets).long().reshape(-1)
        targt_idx = targets.reshape(-1)

        s_rel = s_rel[batch_idx, modif_idx, targt_idx, :]
        s_rel = s_rel.reshape((batch_size, seq_len, -1))
        predicted_labels = s_rel.argmax(-1).cpu().data.numpy()
        return parsed_tree, predicted_labels, s_arc, s_rel

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class SharedDropout(nn.Module):
    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask


class IndependentDropout(nn.Module):
    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p) for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1) for item, mask in zip(items, masks)]

        return items


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            input_size = hidden_size * 2

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        s += ')'

        return s

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size]))
                        for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(self, sequence, hx=None):
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            x_f, (h_f, c_f) = self.layer_forward(x=x,
                                                 hx=(h[i, 0], c[i, 0]),
                                                 cell=self.f_cells[i],
                                                 batch_sizes=batch_sizes)
            x_b, (h_b, c_b) = self.layer_forward(x=x,
                                                 hx=(h[i, 1], c[i, 1]),
                                                 cell=self.b_cells[i],
                                                 batch_sizes=batch_sizes,
                                                 reverse=True)
            x = torch.cat((x_f, x_b), -1)
            h_n.append(torch.stack((h_f, h_b)))
            c_n.append(torch.stack((c_f, c_b)))
        x = PackedSequence(x,
                           sequence.batch_sizes,
                           sequence.sorted_indices,
                           sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)

        return x, hx


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s
