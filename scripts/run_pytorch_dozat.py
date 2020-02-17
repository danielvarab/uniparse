"""Pytorch implementation of Kiperwasser and Goldberg."""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from uniparse import Trainer
from uniparse.callbacks import Callback, ModelSaveCallback
from uniparse.config import ParameterConfig, pprint_dict
from uniparse.models.dozat_pytorch import DozatManning
from uniparse.vocabulary import Vocabulary


class SchedulerCallback(Callback):
    def __init__(self, decay, steps, optimizer, params):
        self.scheduler = ExponentialLR(optimizer, decay ** (1 / steps))
        self.params = params

    def on_batch_end(self, info):
        nn.utils.clip_grad_norm_(self.params(), 5.0)
        self.scheduler.step()


def train(args):
    vocab = Vocabulary()
    vocab = vocab.fit(args.train, pretrained_embeddings=args.femb)
    vocab.save(args.vocab)

    # prep train + dev + (?test)
    training_data = vocab.tokenize_conll(args.train)
    dev_data = vocab.tokenize_conll(args.dev)
    if args.test:
        test_data = vocab.tokenize_conll(args.test)

    wembs = vocab.load_embedding()
    wembs /= np.std(wembs)  # crucial hyperparameter of the paper
    model = DozatManning(args, vocab).load_pretrained(torch.Tensor(wembs))

    adam_optimizer = Adam(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.mu, args.nu),
        eps=args.epsilon,
    )

    # prep params
    trainer = Trainer(
        model,
        decoder=args.decoder,
        loss="crossentropy",
        optimizer=adam_optimizer,
        strategy="bucket",
        vocab=vocab,
    )

    # GPU stuff
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    trainer.train(
        training_data,
        args.dev,
        dev_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        callbacks=[
            ModelSaveCallback(args.model),
            SchedulerCallback(
                decay=args.decay,
                steps=args.decay_steps,
                optimizer=adam_optimizer,
                params=model.parameters,
            ),
        ],
    )

    # load best model
    trainer.load_from_file(args.model)

    if args.test:
        metrics = trainer.evaluate(args.test, test_data, batch_size=args.batch_size)
    else:
        metrics = trainer.evaluate(args.dev, dev_data, batch_size=args.batch_size)
    print(metrics)


def get_train_parser(subparsers):
    parser = subparsers.add_parser("train")
    parser.add_argument("--config", action=ParameterConfig, required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--femb", required=True)
    parser.set_defaults(_func=train)


def evaluate(args):
    vocab = Vocabulary()
    vocab = vocab.load(args.vocab)

    test_data = vocab.tokenize_conll(args.test)

    model = DozatManning(args, vocab).load_from_file(args.model)

    adam_optimizer = Adam(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.mu, args.nu),
        eps=args.epsilon,
    )

    trainer = Trainer(
        model,
        decoder=args.decoder,
        loss="crossentropy",
        optimizer=adam_optimizer,
        strategy="bucket",
        vocab=vocab,
    )

    # load best model
    trainer.load_from_file(args.model)

    metrics = trainer.evaluate(args.test, test_data, batch_size=args.batch_size)

    print(f"Evaluation: {args.test}")
    pprint_dict(metrics)


def get_eval_parser(subparsers):
    parser = subparsers.add_parser("eval")
    parser.add_argument("--config", action=ParameterConfig, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--test", required=True)
    parser.set_defaults(_func=evaluate)


def run(args):
    raise NotImplementedError()


def get_run_parser(subparsers):
    pass


def main():
    """Main function."""
    description = "Script for train/eval/running Dozat and Manning 2017"
    parser = argparse.ArgumentParser(description=description)

    subparsers = parser.add_subparsers()
    subparsers.required = True

    for procedure in [get_train_parser, get_eval_parser, get_run_parser]:
        procedure(subparsers)

    args, _ = parser.parse_known_args()

    pprint_dict(vars(args))

    # start procedure (train|eval|run)
    args._func(args)


if __name__ == "__main__":
    main()
