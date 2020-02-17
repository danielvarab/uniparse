"""Pytorch implementation of Kiperwasser and Goldberg."""

import argparse

from uniparse import Model
from uniparse.callbacks import ModelSaveCallback
from uniparse.config import ParameterConfig, pprint_dict
from uniparse.models.kiperwasser_pytorch import DependencyParser
from uniparse.vocabulary import Vocabulary


def train(args):
    vocab = Vocabulary().fit(args.train)  # , args.min_freq)
    vocab.save(args.vocab)

    # prepare data
    training_data = vocab.tokenize_conll(args.train)
    dev_data = vocab.tokenize_conll(args.dev)
    if args.test:
        test_data = vocab.tokenize_conll(args.test)

    model = DependencyParser(args, vocab)

    # prep params
    parser = Model(
        model,
        decoder=args.decoder,
        loss="hinge",
        optimizer="adam",
        strategy="bucket",
        vocab=vocab,
    )

    parser.train(
        training_data,
        args.dev,
        dev_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ModelSaveCallback(args.model)],
    )

    # load best model
    model.load_from_file(args.model)
    if args.test:
        metrics = parser.evaluate(args.test, test_data, args.batch_size)

    pprint_dict(metrics)


def get_train_parser(subparsers):
    parser = subparsers.add_parser("train")
    parser.add_argument("--config", action=ParameterConfig, required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.set_defaults(_func=train)


def evaluate(args):
    vocab = Vocabulary().load(args.vocab)

    testdata = vocab.tokenize_conll(args.test)

    model = DependencyParser(args, vocab).load_from_file(args.model)

    parser = Model(
        model,
        decoder=args.decoder,
        loss="hinge",
        optimizer="adam",
        strategy="bucket",
        vocab=vocab,
    )

    metrics = parser.evaluate(args.test, testdata, args.batch_size)

    pprint_dict(metrics)


def get_eval_parser(subparsers):
    parser = subparsers.add_parser("eval")
    parser.add_argument("--config", action=ParameterConfig, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--test", required=False)
    parser.set_defaults(_func=evaluate)


def run(args):
    raise NotImplementedError()


def get_run_parser(subparsers):
    pass


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    for procedure in [get_train_parser, get_eval_parser, get_run_parser]:
        procedure(subparsers)

    args, _ = parser.parse_known_args()

    pprint_dict(vars(args))

    args._func(args)

    args, _ = parser.parse_known_args()


def _main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    for procedure in [get_train_parser, get_eval_parser, get_run_parser]:
        procedure(subparsers)

    args, _ = parser.parse_known_args()

    pprint_dict(args)

    args._func(args)

    args, _ = parser.parse_known_args()

    vocab = Vocabulary().fit(args.train)
    vocab.save(args)

    # prep data
    training_data = vocab.tokenize_conll(args.train)
    dev_data = vocab.tokenize_conll(args.dev)
    test_data = vocab.tokenize_conll(args.test)

    model = DependencyParser(args, vocab)

    save_callback = ModelSaveCallback(args.model)

    # prep params
    parser = Model(
        model,
        decoder=args.decoder,
        loss="hinge",
        optimizer="adam",
        strategy="bucket",
        vocab=vocab,
    )

    parser.train(
        training_data,
        args.dev,
        dev_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[save_callback],
    )

    # load best model
    model.load_from_file(args.model)

    metrics = parser.evaluate(args.test, test_data, batch_size=args.batch_size)

    print(metrics)


if __name__ == "__main__":
    main()
