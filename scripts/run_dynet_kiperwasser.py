"""Run script for training/evaluating/running Kiperwasser and Golberg (2017)."""

import argparse

import uniparse.util
from uniparse import Trainer, Vocabulary
from uniparse.callbacks import ModelSaveCallback
from uniparse.config import ParameterConfig, pprint_dict
from uniparse.models.kiperwasser_dynet import DependencyParser


def get_train_parser(subparsers):
    """Create parser for the 'train' command."""
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", action=ParameterConfig, required=True)
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--dev", required=False)
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--vocab", required=True)
    train_parser.set_defaults(_func=_train)

    return train_parser


def _train(args):
    vocab = Vocabulary().fit(args.train, args.embs)
    word_embeddings = vocab.load_embedding() if args.embs else None
    if word_embeddings:
        print("> Embedding shape", word_embeddings.shape)

    # save vocab for reproducability later
    print("> Saving vocabulary to %s" % args.vocab)
    vocab.save(args.vocab)

    # prepare data
    training_data = vocab.tokenize_conll(args.train)
    dev_data = vocab.tokenize_conll(args.dev) if args.dev else None

    # instantiate model
    model = DependencyParser(args, vocab, word_embeddings)

    # prep params
    trainer = Trainer(
        model,
        decoder="eisner",
        loss="kiperwasser",
        optimizer="adam",
        #strategy="bucket",
        strategy="scaled_batch",
        vocab=vocab,
    )

    trainer.train(
        training_data,
        args.dev,
        dev_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ModelSaveCallback(args.model)],
        patience=args.patience,
    )


def get_eval_model_parser(subparsers):
    parser = subparsers.add_parser("eval")
    parser.add_argument("--config", action=ParameterConfig, required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)

    parser.set_defaults(_func=eval_model)
    return parser


def eval_model(args):
    vocab = Vocabulary().load(args.vocab)
    model = DependencyParser(args, vocab, embs=None)

    trainer = Trainer(
        model,
        decoder="eisner",
        loss="kiperwasser",
        optimizer="adam",
        strategy="bucket",
        vocab=vocab,
    )

    trainer.load_from_file(args.model)
    test_data = vocab.tokenize_conll(args.test)
    metrics = trainer.evaluate(args.test, test_data, args.batch_size)

    pprint_dict(metrics)


def get_run_parser(subparsers):
    parser = subparsers.add_parser("run")
    parser.add_argument("--config", action=ParameterConfig, required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--no-gold")
    parser.add_argument("--batch-size", default=32)

    parser.set_defaults(_func=run_model)
    return parser


def run_model(args):
    vocab = Vocabulary().load(args.vocab)
    model = DependencyParser(args, vocab, embs=None)
    parser = Trainer(
        model,
        decoder="eisner",
        loss="kiperwasser",
        optimizer="adam",
        strategy="bucket",
        vocab=vocab,
    )

    parser.load_from_file(args.model)

    run_data = vocab.tokenize_conll(args.test)
    predictions = parser.run(run_data, args.batch_size)
    uniparse.util.write_predictions_to_file(
        predictions=predictions,
        reference_file=args.test,
        output_file=args.output,
        vocab=vocab,
    )

    print(">> Wrote predictions to conllu file %s" % args.output)


def main():
    """Set up command line parser."""
    description = "Script for train/eval/running Kiperwasser and Goldberg (2017)."
    parser = argparse.ArgumentParser(description=description)

    subparsers = parser.add_subparsers()
    subparsers.required = True

    for subparsers_func in [get_train_parser, get_eval_model_parser, get_run_parser]:
        subparsers_func(subparsers)

    # second parameter is unrecognized arguments
    args, _ = parser.parse_known_args()

    pprint_dict(vars(args))

    args._func(args)


if __name__ == "__main__":
    main()
