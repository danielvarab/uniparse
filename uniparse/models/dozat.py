"""
Implementation of Dozat and Manning (2018) in DyNet.

This implementation is an adaptation of the one found at:
    https://github.com/jcyk/Dynet-Biaffine-dependency-parser
"""
from uniparse.utypes import Parser

import uniparse
import dynet as dy
import numpy as np

from uniparse.decoders.tarjan import Tarjan


def arc_argmax(parse_probs, length, tokens_to_keep, ensure_tree=True):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
    """
    if ensure_tree:
        I = np.eye(len(tokens_to_keep))
        # block loops and pad heads
        parse_probs = parse_probs * tokens_to_keep * (1-I)
        parse_preds = np.argmax(parse_probs, axis=1)
        tokens = np.arange(1, length)
        roots = np.where(parse_preds[tokens] == 0)[0]+1
        # ensure at least one root
        if len(roots) < 1:
            # The current root probabilities
            root_probs = parse_probs[tokens, 0]
            # The current head probabilities
            old_head_probs = parse_probs[tokens, parse_preds[tokens]]
            # Get new potential root probabilities
            new_root_probs = root_probs / old_head_probs
            # Select the most probable root
            new_root = tokens[np.argmax(new_root_probs)]
            # Make the change
            parse_preds[new_root] = 0
            # ensure at most one root
        elif len(roots) > 1:
            # The probabilities of the current heads
            root_probs = parse_probs[roots, 0]
            # Set the probability of depending on the root zero
            parse_probs[roots,0] = 0
            # Get new potential heads and their probabilities
            new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1)+1
            new_head_probs = parse_probs[roots, new_heads] / root_probs
            # Select the most probable root
            new_root = roots[np.argmin(new_head_probs)]
            # Make the change
            parse_preds[roots] = new_heads
            parse_preds[new_root] = 0
        # remove cycles
        tarjan = Tarjan(parse_preds, tokens)
        # cycles = tarjan.SCCs
        for SCC in tarjan.SCCs:
            if len(SCC) > 1:
                dependents = set()
                to_visit = set(SCC)
                while len(to_visit) > 0:
                    node = to_visit.pop()
                    if not node in dependents:
                        dependents.add(node)
                        to_visit.update(tarjan.edges[node])
                # The indices of the nodes that participate in the cycle
                cycle = np.array(list(SCC))
                # The probabilities of the current heads
                old_heads = parse_preds[cycle]
                old_head_probs = parse_probs[cycle, old_heads]
                # Set the probability of depending on a non-head to zero
                non_heads = np.array(list(dependents))
                parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1)+1
                new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
                # Select the most probable change
                change = np.argmax(new_head_probs)
                changed_cycle = cycle[change]
                old_head = old_heads[change]
                new_head = new_heads[change]
                # Make the change
                parse_preds[changed_cycle] = new_head
                tarjan.edges[new_head].add(changed_cycle)
                tarjan.edges[old_head].remove(changed_cycle)
        return parse_preds
    else:
        # block and pad heads
        parse_probs = parse_probs * tokens_to_keep
        parse_preds = np.argmax(parse_probs, axis=1)
        return parse_preds


def rel_argmax(rel_probs, length, ensure_tree=True):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
    """
    if ensure_tree:
        rel_probs[:, 0] = 0
        root = 1
        tokens = np.arange(1, length)
        rel_preds = np.argmax(rel_probs, axis=1)
        roots = np.where(rel_preds[tokens] == root)[0]+1
        if len(roots) < 1:
            rel_preds[1+np.argmax(rel_probs[tokens,root])] = root
        elif len(roots) > 1:
            root_probs = rel_probs[roots, root]
            rel_probs[roots, root] = 0
            new_rel_preds = np.argmax(rel_probs[roots], axis=1)
            new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
            new_root = roots[np.argmin(new_rel_probs)]
            rel_preds[roots] = new_rel_preds
            rel_preds[new_root] = root
        return rel_preds
    else:
        rel_probs[:, 0] = 0
        rel_preds = np.argmax(rel_probs, axis=1)
        return rel_preds


def leaky_relu(x):
    """Leaky ReLU implementation."""
    return dy.bmax(.1*x, x)


def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    # x,y: (input_size x seq_len) x batch_size
    if bias_x:
        x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
    if bias_y:
        y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])

    nx, ny = input_size + bias_x, input_size + bias_y
    lin = W * x
    if num_outputs > 1:
        lin = dy.reshape(lin, (ny, num_outputs*seq_len), batch_size = batch_size)
    blin = dy.transpose(y) * lin
    if num_outputs > 1:
        blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size = batch_size)
    return blin


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05/(output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc):
    builder = dy.VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc)
    for layer, params in enumerate(builder.get_parameters()):
        W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + (lstm_hiddens if layer > 0 else input_dims))
        W_h, W_x = W[:, :lstm_hiddens], W[:, lstm_hiddens:]
        params[0].set_value(np.concatenate([W_x] * 4, 0))
        params[1].set_value(np.concatenate([W_h] * 4, 0))
        b = np.zeros(4 * lstm_hiddens, dtype=np.float32)
        b[lstm_hiddens: 2*lstm_hiddens] = -1.0
        params[2].set_value(b)
    return builder


def biLSTM(builders, inputs, batch_size=None, dropout_x=0., dropout_h = 0.):
    for fb, bb in builders:
        f, b = fb.initial_state(), bb.initial_state()
        fb.set_dropouts(dropout_x, dropout_h)
        bb.set_dropouts(dropout_x, dropout_h)
        if batch_size is not None:
            fb.set_dropout_masks(batch_size)
            bb.set_dropout_masks(batch_size)
        fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
        inputs = [dy.concatenate([f,b]) for f, b in zip(fs, reversed(bs))]
    return inputs


def dynet_flatten_numpy(ndarray):
    return np.reshape(ndarray, (-1,), 'F')


def gen_lookup_param(count, dim):
    return np.random.randn(count, dim).astype(np.float32)


class DozatManning(Parser):
    def __init__(self, vocab,
                       word_dims,
                       tag_dims,
                       dropout_dim,
                       lstm_layers,
                       lstm_hiddens,
                       dropout_lstm_input,
                       dropout_lstm_hidden,
                       mlp_arc_size,
                       mlp_rel_size,
                       dropout_mlp,
                       pretrained_embeddings):
        pc = dy.ParameterCollection()

        self._vocab = vocab

        random_np_word_emb = gen_lookup_param(vocab.words_in_train, word_dims)
        self.word_embs = pc.lookup_parameters_from_numpy(random_np_word_emb)
        if pretrained_embeddings is not None:
            self.pret_word_embs = pc.lookup_parameters_from_numpy(pretrained_embeddings)
        else:
            self.pret_word_embs = None
        random_np_tag_emb = gen_lookup_param(vocab.tag_size, tag_dims)
        self.tag_embs = pc.lookup_parameters_from_numpy(random_np_tag_emb)

        lstm = orthonormal_VanillaLSTMBuilder
        self.LSTM_builders = []
        f = lstm(1, word_dims+tag_dims, lstm_hiddens, pc)
        b = lstm(1, word_dims+tag_dims, lstm_hiddens, pc)

        self.LSTM_builders.append((f, b))
        for i in range(lstm_layers-1):
            f = lstm(1, 2*lstm_hiddens, lstm_hiddens, pc)
            b = lstm(1, 2*lstm_hiddens, lstm_hiddens, pc)
            self.LSTM_builders.append((f, b))
        self.dropout_lstm_input = dropout_lstm_input
        self.dropout_lstm_hidden = dropout_lstm_hidden

        mlp_size = mlp_arc_size + mlp_rel_size
        w = orthonormal_initializer(mlp_size, 2 * lstm_hiddens)
        self.mlp_dep_W = pc.parameters_from_numpy(w)
        self.mlp_head_W = pc.parameters_from_numpy(w)
        
        self.mlp_dep_b = pc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_head_b = pc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_arc_size = mlp_arc_size
        self.mlp_rel_size = mlp_rel_size
        self.dropout_mlp = dropout_mlp

        self.arc_W = pc.add_parameters((mlp_arc_size, mlp_arc_size + 1), init=dy.ConstInitializer(0.))
        self.rel_W = pc.add_parameters((vocab.rel_size*(mlp_rel_size + 1), mlp_rel_size + 1), init=dy.ConstInitializer(0.))

        self._pc = pc
        self.params = pc

        def _emb_mask_generator(seq_len, batch_size):
            ret = []
            for i in range(seq_len):
                word_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
                tag_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
                scale = 3. / (2.*word_mask + tag_mask + 1e-12)
                word_mask *= scale
                tag_mask *= scale
                word_mask = dy.inputTensor(word_mask, batched = True)
                tag_mask = dy.inputTensor(tag_mask, batched = True)
                ret.append((word_mask, tag_mask))
            return ret
        self.generate_emb_mask = _emb_mask_generator

    @property 
    def parameter_collection(self):
        return self._pc

    def parameters(self):
        return self.params

    def save_to_file(self, filename: str) -> None:
        self.params.save(filename)

    def load_from_file(self, filename: str) -> None:
        self.params.populate(filename)

    def __call__(self, x):
        word_ids, lemma_ids, upos_ids, gold_arcs, gold_rels, chars = x
        return self.run(word_ids, upos_ids, gold_arcs, gold_rels)

    def run(self, word_inputs, tag_inputs, arc_targets=None, rel_targets=None):

        is_train = arc_targets is not None

        # @djam modification
        word_inputs = word_inputs.T
        tag_inputs = tag_inputs.T

        if arc_targets is not None:
            arc_targets[:, 0] = 0
            arc_targets = arc_targets.T
            targets_1D = dynet_flatten_numpy(arc_targets)


        batch_size = word_inputs.shape[1]
        seq_len = word_inputs.shape[0]

        mask = np.greater(word_inputs, self._vocab.ROOT).astype(np.float32)
        
        if self.pret_word_embs:
            word_embs = [
                dy.lookup_batch(self.word_embs, np.where(w < self._vocab.words_in_train, w, self._vocab.UNK))
                + dy.lookup_batch(self.pret_word_embs, w, update=False)
                for w in word_inputs
            ]
        else:
            word_embs = [dy.lookup_batch(self.word_embs, np.where(w < self._vocab.words_in_train, w, self._vocab.UNK)) for w in word_inputs]
        tag_embs = [dy.lookup_batch(self.tag_embs, pos) for pos in tag_inputs]
        
        if is_train:
            emb_masks = self.generate_emb_mask(seq_len, batch_size)
            emb_inputs = [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos, posm)]) for w, pos, (wm, posm) in zip(word_embs, tag_embs, emb_masks)]
        else:
            emb_inputs = [dy.concatenate([w, pos]) for w, pos in zip(word_embs, tag_embs)]

        top_recur = dy.concatenate_cols(biLSTM(self.LSTM_builders, emb_inputs, batch_size, self.dropout_lstm_input if is_train else 0., self.dropout_lstm_hidden if is_train else 0.))
        if is_train:
            top_recur = dy.dropout_dim(top_recur, 1, self.dropout_mlp)

        W_dep, b_dep = dy.parameter(self.mlp_dep_W), dy.parameter(self.mlp_dep_b)
        W_head, b_head = dy.parameter(self.mlp_head_W), dy.parameter(self.mlp_head_b)
        dep, head = leaky_relu(dy.affine_transform([b_dep, W_dep, top_recur])), leaky_relu(dy.affine_transform([b_head, W_head, top_recur]))
        if is_train:
            dep, head= dy.dropout_dim(dep, 1, self.dropout_mlp), dy.dropout_dim(head, 1, self.dropout_mlp)
        
        dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
        head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

        W_arc = dy.parameter(self.arc_W)
        arc_logits = bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size, seq_len, batch_size, num_outputs= 1, bias_x = True, bias_y = False)
        # (#head x #dep) x batch_size

        arc_preds = arc_logits.npvalue().argmax(0)
        arc_preds = arc_preds if arc_preds.ndim == 2 else arc_preds[:, None]
        # seq_len x batch_size
        
        W_rel = dy.parameter(self.rel_W)
        
        rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len, batch_size, num_outputs = self._vocab.rel_size, bias_x = True, bias_y = True)
        # (#head x rel_size x #dep) x batch_size
        
        flat_rel_logits = dy.reshape(rel_logits, (seq_len, self._vocab.rel_size), seq_len * batch_size)
        # (#head x rel_size) x (#dep x batch_size)

        partial_rel_logits = dy.pick_batch(flat_rel_logits, targets_1D if is_train else dynet_flatten_numpy(arc_preds))
        # (rel_size) x (#dep x batch_size)

        # @djam - restored shape
        partial_rel_logits = dy.reshape(partial_rel_logits, (self._vocab.rel_size, seq_len), batch_size)
        
        # if not isTrain:
        arc_probs = np.transpose(np.reshape(dy.softmax(arc_logits).npvalue(), (seq_len, seq_len, batch_size), 'F'))
        # #batch_size x #dep x #head
        rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(), (self._vocab.rel_size, seq_len, seq_len, batch_size), 'F'))
        # batch_size x #dep x #head x #nclasses

        # @djam contribution
        if is_train:
            # 'decode' with argmax
            # Why on earth can't i get this guy to work
            # arc_predictions = arc_probs.argmax(1)
            arc_predictions = arc_preds.T
            # batch_size x dep

            _1 = np.repeat(range(batch_size), seq_len)  # batches
            _2 = np.tile(range(seq_len), batch_size)  # modifiers
            _3 = arc_predictions.reshape(-1)  # predicted arcs

            rel_predictions = rel_probs[_1, _2, _3].argmax(-1)
            rel_predictions = rel_predictions.reshape(batch_size, seq_len)
            # batch_size x dep

            return arc_predictions, rel_predictions, arc_logits, partial_rel_logits
        else:
            arc_predictions, rel_predictions = [], []
            for msk, arc_prob, rel_prob in zip(np.transpose(mask), arc_probs, rel_probs):
                msk[0] = 1.
                sent_len = int(np.sum(msk))
                arc_pred = uniparse.arc_argmax(arc_prob, sent_len, msk)
                rel_prob = rel_prob[np.arange(len(arc_pred)), arc_pred]
                rel_pred = uniparse.rel_argmax(rel_prob, sent_len)

                arc_predictions.append(arc_pred[:sent_len])
                rel_predictions.append(rel_pred[:sent_len])
            
            return arc_predictions, rel_predictions, None, None
