import numpy as np
import dynet as dy


class _DynetOptimizers(object):
    def __init__(self):
        self.sgd = dy.SimpleSGDTrainer
        self.rmsprop = dy.RMSPropTrainer
        self.adam = dy.AdamTrainer
        self.adadelta = dy.AdadeltaTrainer
        self.adagrad = dy.AdagradTrainer


def dynet_flatten(ndarray):
    return np.reshape(ndarray, (-1), "F")


def generate_mask(shape, target):
    d, n, batch_size = shape
    _mask = np.ones((d, n, batch_size))
    for b in range(target.shape[0]):
        for i in range(target.shape[1]):
            _mask[target[b, i], i, b] -= 1
    _mask = np.reshape(_mask, (d, n * batch_size), "F")
    return _mask


class _DynetLossFunctions(object):

    @staticmethod  # actually used for arcs on
    def kipperwasser_loss(scores, preds, golds, mask):
        (h, m), batch_size = scores.dim()

        # given that scores are shaped ((h,m), batch)
        # we can reshape it to the shape ((h,), m*batch)
        # where each batch is a modifier with
        # respect to its heads
        scores = dy.reshape(scores, (h,), batch_size=m * batch_size)

        # this is merely to avoid the out of bounds problem
        # root has its head = -1. trying to 'pick_batch' will
        # will of course cause an issue
        preds[:, 0] = golds[:, 0] = 0

        # to satisfy dynet being column major as oppposed to numpy / default encoding being row major
        preds = preds.T
        golds = golds.T

        # create boolean mask. 1s for all the wrong values and 0s for all the correct values
        incorrect_mask = preds != golds
        mask = incorrect_mask * mask.T if mask is not None else incorrect_mask
        incorrect_mask_tensor = dy.inputTensor(dynet_flatten(mask), batched=True)

        # masks for padding and root
        pred_tensor = dy.pick_batch(scores, dynet_flatten(preds))
        gold_tensor = dy.pick_batch(scores, dynet_flatten(golds))

        loss = pred_tensor - gold_tensor
        masked_loss = loss * incorrect_mask_tensor

        return dy.sum_batches(masked_loss) / batch_size

    @staticmethod
    def crossentropy(x, pred_y, y, mask):
        # for now lets assume the inverted nature
        # => this means batch last, like (head,modi,batch_size) or (labels, modi, batch_size)
        num_tokens = int(np.sum(mask))
        batch_size, seq_len = y.shape
        (d, _), _ = x.dim()

        y[:, 0] = 0  # to ensure that we don't use the -1 output of a decoder / expected output

        mask_1d = dynet_flatten(mask.T)
        mask_1d_tensor = dy.inputTensor(mask_1d, batched=True)

        targets_1d = dynet_flatten(y.T)
        flat_x = dy.reshape(x, (d,), seq_len * batch_size)

        losses = dy.pickneglogsoftmax_batch(flat_x, targets_1d)
        arc_loss = dy.sum_batches(losses * mask_1d_tensor) / num_tokens

        return arc_loss

    @staticmethod # actually used for labels
    def kipperwasser_hinge(x, pred_y, y, mask):
        (label_count, n), batch_size = x.dim()

        rel_losses = [dy.zeros(1)]
        for bi in range(batch_size):
            batch_score = dy.pick_batch_elem(x, bi)
            for i in range(n):
                if mask is not None and not mask[bi, i]:
                    continue
                pred_vec = dy.pick(batch_score, i, dim=1)
                gold_index = y[bi, i]
                non_golds = [dy.pick(pred_vec, l) for l in range(label_count) if l != gold_index]
                wrong_value = dy.emax(non_golds)
                correct_value = pred_vec[int(gold_index)]
                loss = dy.bmax(dy.ones(1) + wrong_value - correct_value, dy.zeros(1))
                rel_losses.append(loss)

        rel_losses = dy.esum(rel_losses) / batch_size

        return rel_losses

    @staticmethod
    def hinge(scores, preds, golds, mask, margin=1):
        (h, m), batch_size = scores.dim()

        # given that scores are shaped ((h,m), batch)
        # we can reshape it to the shape ((h,), m*batch)
        # where each batch is a modifier with
        # respect to its heads
        scores = dy.reshape(scores, (h,), batch_size=m * batch_size)

        # this is merely to avoid the out of bounds problem
        # root has its head = -1. trying to 'pick_batch' will
        # will of course cause an issue
        golds[:, 0] = 0
        if preds is not None:
            preds[:, 0] = 0
            preds = preds.T
        else:
            _mask = generate_mask((h, m, batch_size), golds)
            mask_tensor = dy.inputTensor(_mask, batched=True)
            scores = scores + mask_tensor
            preds = scores.npvalue().argmax(0)
            preds = preds[:, None] if preds.ndim < 2 else preds

        # to satisfy dynet being column major as oppposed to numpy being row major
        golds = golds.T

        # create boolean mask. 1s for all the wrong values and 0s for all the correct values
        incorrect_mask = preds != golds
        mask = incorrect_mask * mask.T if mask is not None else incorrect_mask
        incorrect_mask_tensor = dy.inputTensor(dynet_flatten(mask), batched=True)

        # masks for padding and root
        pred_tensor = dy.pick_batch(scores, dynet_flatten(preds))
        gold_tensor = dy.pick_batch(scores, dynet_flatten(golds))

        loss = dy.bmax(dy.zeros(1, batch_size=m * batch_size), pred_tensor - gold_tensor)
        masked_loss = loss * incorrect_mask_tensor

        return dy.sum_batches(masked_loss) / batch_size


class DynetBackend(object):
    def __init__(self):
        self.K = dy
        self.optimizers = _DynetOptimizers()
        self.loss = _DynetLossFunctions()

    @staticmethod
    def to_numpy(expression):
        """ Convert expression into numpy ndarray """
        values = expression.npvalue().astype(np.float64)
        # Since dynet squeezes the batch dimensions if it is equals to 1, we expand
        # to get a dimensionality of at least 3 if the the dimensionality is correct,
        # then we push the batch_dimension to the first to 0, for compatibility
        values = values[None, :, :] if values.ndim < 3 else np.moveaxis(values, 2, 0)
        return values

    @staticmethod
    def get_scalar(expression):
        return expression.value()

    @staticmethod
    def shape(expression):
        shape, batch_size = expression.dim()
        return (batch_size,) + shape

    @staticmethod
    def tensor(ndarray, dtype=None):  # dtype is ignored in
        return dy.inputTensor(ndarray)

    @staticmethod
    def input_tensor(ndarray, dtype=None):
        return ndarray

    @staticmethod
    def renew_cg():
        dy.renew_cg()

    @staticmethod
    def step(optimizer):
        optimizer.update()

