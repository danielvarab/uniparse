import numpy as np


from uniparse.decoders.eisner import Eisner

# helper


def unwrap_linear_encoding(ndarray):
    n, label_count, _, _, d = ndarray.shape
    result = np.zeros((n, n, label_count, d*2), dtype=np.int)
    for head in range(n):
        for modi in range(n):
            if head == modi:
                continue
            right_arc = 1 if head < modi else 0

            result[head, modi, :, :d] = ndarray[head, :, right_arc, 0, :]
            result[head, modi, :, d:] = ndarray[modi, :, right_arc, 1, :]

    return result


class MST(object):
    def __init__(self, dim):
        self.W = np.zeros((dim,), np.int32)
        self.decoder = Eisner

    def _score(self, arc_edges, rel_edges, target_arcs):
        # arcs
        arc_mask = arc_edges > 0
        arc_edges = self.W[arc_edges]
        arc_scores = arc_edges * arc_mask  # masked scores
        arc_scores = arc_scores.sum(-1)
        # (b, n, n)

        rel_edges = unwrap_linear_encoding(rel_edges)

        # rels
        rel_mask = rel_edges > 0
        rel_features = self.W[rel_edges.astype(np.int)]
        rel_features = rel_features * rel_mask 
        rel_scores = rel_features.sum(-1)

        rel_scores = arc_scores[:, :, None] + rel_scores

        arc_scores = rel_scores.max(-1).astype(np.float64)

        parsed_arcs = self.decoder(arc_scores, None)

        arcs_to_use = parsed_arcs if target_arcs is None else target_arcs

        n = arcs_to_use.shape[0]
        _3 = range(n)
        flat_gold_arcs = arcs_to_use.reshape((-1,))
        label_preds = rel_scores[flat_gold_arcs, _3].argmax(-1)

        return parsed_arcs, label_preds, parsed_arcs, rel_edges

    def _update(self, pred_labels, parsed_arcs, targets_arcs, target_rels, arc_edges_fv, rel_edges_fv):
        # book keeping
        n = targets_arcs.shape[0]
        ntokens = parsed_arcs.size

        arc_edges_fv = arc_edges_fv.astype(np.int)

        # batch indices and modifier indices
        _3 = range(n)

        # flatten targets
        flat_gold_arcs = targets_arcs.reshape((-1,))
        flat_gold_rels = target_rels.reshape((-1,))
        flat_pred_arcs = parsed_arcs.reshape((-1,))

        # extract label predictions
        label_preds = pred_labels  # label_scores[_1, flat_gold_arcs, _3].argmax(-1)

        # flatten label prediction
        flat_pred_rels = label_preds.reshape((-1,))

        # calculate precision
        arc_accuracy = np.equal(parsed_arcs, targets_arcs)
        arc_accuracy = arc_accuracy.sum()/ntokens
        rel_accuracy = np.equal(label_preds, target_rels)
        rel_accuracy = rel_accuracy.sum() / ntokens

        # extract gold/predicted arc feature vectors
        gold_arc_features = arc_edges_fv[flat_gold_arcs, _3]
        pred_arc_features = arc_edges_fv[flat_pred_arcs, _3]

        # extract gold/predicted label feature vectors
        gold_rel_features = rel_edges_fv[flat_gold_arcs, _3, flat_gold_rels]
        pred_rel_features = rel_edges_fv[flat_gold_arcs, _3, flat_pred_rels]

        # collect into two arrays ndarray (positive updates & negative updates)
        gold_features = np.concatenate([gold_arc_features, gold_rel_features], axis=-1)
        pred_features = np.concatenate([pred_arc_features, pred_rel_features], axis=-1)

        # filter out invalid features
        valid_gold_features = gold_features[gold_features > 0]
        valid_pred_features = pred_features[pred_features > 0]

        # add and subtract features accordingly
        np.add.at(self.W, valid_gold_features, 1)
        np.subtract.at(self.W, valid_pred_features, 1)

        return arc_accuracy, rel_accuracy

    def __call__(self, arc_edges, rel_edges, target_arcs, target_rels):
        pred_arcs, pred_rels, parsed_arcs, rel_edges = self._score(arc_edges, rel_edges, target_arcs)

        if target_arcs is not None:
            uas, las = self._update(pred_rels, parsed_arcs, target_arcs, target_rels, arc_edges, rel_edges)
        else:
            uas, las = None, None
        
        return pred_arcs, pred_rels.astype(np.int32), uas, las
