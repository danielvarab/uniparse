import torch
import torch.optim as optim
import numpy as np


def generate_mask(shape, target):
    batch_size, n, d = shape
    mask = np.ones((batch_size, n, d))
    target[:, 0] = 0
    for b in range(batch_size):
        for i in range(n):
            mask[b, i, target[b, i]] -= 1
    return mask


class _PytorchOptimizers(object):
    def __init__(self):
        self.sgd = optim.SGD
        self.rmsprop = optim.RMSprop
        self.adam = optim.Adam
        self.adadelta = optim.Adadelta
        self.adagrad = optim.Adagrad


class _PytorchLossFunctions(object):
    # actually used for arcs on
    @staticmethod
    def kipperwasser_loss(scores, preds, golds, mask):
        raise NotImplementedError()

    @staticmethod
    def crossentropy(x, pred_y, y, mask):
        raise NotImplementedError()

    @staticmethod  # actually used for labels
    def kipperwasser_hinge(x, pred_y, y, mask):
        raise NotImplementedError()

    @staticmethod
    def hinge(scores, preds, golds, mask):
        batch_size, n, d = scores.shape
        n_tokens = batch_size * n

        # this is merely to avoid the out of bounds problem
        # root has its head = -1. trying to 'pick_batch' will
        # will of course cause an issue
        golds[:, 0] = 0
        if preds is not None:
            preds[:, 0] = 0
        else:
            # margin at loss / prediction time
            margin_mask = generate_mask((batch_size, n, d), golds)
            mask_tensor = torch.Tensor(margin_mask)
            scores = scores + mask_tensor
            preds = scores.argmax(2).data.numpy()
            preds = preds[None, :] if preds.ndim < 2 else preds

        # create boolean mask. 1s for all the wrong values and 0s for all the correct values
        incorrect_mask = np.not_equal(preds, golds)
        mask = np.logical_and(mask, incorrect_mask).astype(np.float64)
        incorrect_mask_tensor = torch.Tensor(mask.reshape(-1))

        b = np.arange(n_tokens)
        pred_tree_tensor = preds.reshape(-1)
        gold_tree_tensor = golds.reshape(-1)

        # extract scores
        scores = scores.view(n_tokens, d)
        pred_tensor = scores[b, pred_tree_tensor]
        gold_tensor = scores[b, gold_tree_tensor]

        loss = torch.max(torch.zeros(n_tokens), pred_tensor - gold_tensor)
        masked_loss = loss * incorrect_mask_tensor

        return torch.sum(masked_loss) / batch_size


class PyTorchBackend(object):
    def __init__(self):
        self.K = torch
        self.optimizers = _PytorchOptimizers()
        self.loss = _PytorchLossFunctions()


    @staticmethod
    def to_numpy(tensor):
        values = tensor.data.numpy().astype(np.float64)
        return values

    @staticmethod
    def get_scalar(tensor):
        values = tensor.data.numpy().astype(np.float64)
        return float(values)

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def tensor(ndarray, dtype="float"):
        if dtype == "int":
            return torch.LongTensor(ndarray)
        elif dtype == "float":
            return torch.Tensor(ndarray)
        else:
            raise ValueError("don't know the dtype utilized")

    def input_tensor(self, ndarray, dtype="float"):
        return self.tensor(ndarray, dtype)

    @staticmethod
    def step(optimizer):
        optimizer.step()
        optimizer.zero_grad()

    @staticmethod
    def renew_cg():
        pass

