"""CLI interface to train, evaluate and run dependency parsers."""

import argparse

import uniparse.models
from uniparse import Vocabulary, Model
from uniparse.callbacks import ModelSaveCallback
import uniparse.models.kiperwasser_dynet
import uniparse.models.kiperwasser_pytorch
from uniparse.util import write_predictions_to_file

MAINPARSER_HELP = "print current version number"
SUBPARSERS_HELP = "%(prog)s must be called with a command:"
EVAL_PARSER_HELP = "Evaluate dependency parser."
TRAIN_PARSER_HELP = "Train a dependency parser."

VERSION = 0.2


INCLUDED_MODELS = {
    "kiperwasser-dynet": uniparse.models.kiperwasser_dynet.DependencyParser,
    "kiperwasser-pytorch": uniparse.models.kiperwasser_pytorch.DependencyParser
}


def main():
    """Set up command line parser."""
    description = "CLI for training model."
    parser = argparse.ArgumentParser(description=description)

    subparsers = parser.add_subparsers(help=SUBPARSERS_HELP, dest="train|dev|run")
    subparsers.required = True

    for subparsers_func in [_get_train_parser, _get_eval_model_parser, _get_run_parser]:
        subparsers_func(subparsers)

    # second parameter is unrecognized arguments
    args, _ = parser.parse_known_args()
    args.func(parser, args)


def _get_train_parser(subparsers):
    """Create parser for the 'train' command."""
    train_parser = subparsers.add_parser("train", help=EVAL_PARSER_HELP)
    train_parser.add_argument("--model-name", required=True)
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--dev", required=False)
    train_parser.add_argument("--parameter-file", required=True)
    train_parser.add_argument("--vocab", required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.set_defaults(func=_train_model)
    return train_parser


def _train_model(_, args):
    train_file = args.train
    dev_file = args.dev
    epochs = args.epochs
    vocab_dest = args.vocab
    model_dest = args.parameter_file
    batch_size = args.batch_size
    embedding_file = None

    model_class = INCLUDED_MODELS.get(args.model_name)

    if not model_class:
        raise ValueError("Model %s doesn't exist." % args.model)

    # Disable patience if there is no dev. set
    patience = args.patience if dev_file else -1

    vocab = Vocabulary().fit(train_file, embedding_file)
    word_embeddings = vocab.load_embedding() if embedding_file else None
    if word_embeddings:
        print("> Embedding shape", word_embeddings.shape)

    # save vocab for reproducability later
    print("> Saving vocabulary to", vocab_dest)
    vocab.save(vocab_dest)

    # prep data
    print(">> Loading in data")
    training_data = vocab.tokenize_conll(train_file)

    dev_data = vocab.tokenize_conll(dev_file) if dev_file else None

    # instantiate model
    model = model_class(vocab, word_embeddings)

    # 'best' only saves models that improve results on the dev. set
    # 'epoch' saves models on each epoch to a file appended with the epoch number
    save_mode = "best" if dev_file else "epoch"
    save_callback = ModelSaveCallback(model_dest, mode=save_mode)
    callbacks = [save_callback]

    # prep params
    parser = Model(
        model,
        decoder="eisner",
        loss="kiperwasser",
        optimizer="adam",
        strategy="bucket",
        vocab=vocab,
    )
    parser.train(
        training_data,
        dev_file,
        dev_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        patience=patience,
    )


def _get_eval_model_parser(subparsers):
    eval_parser = subparsers.add_parser("eval-model", help=EVAL_PARSER_HELP)
    eval_parser.add_argument("--model-name", required=True)
    eval_parser.add_argument("--filename", required=True)
    eval_parser.add_argument("--parameter-file", required=True)
    eval_parser.add_argument("--vocab", required=True)
    eval_parser.add_argument("--no-gold")
    eval_parser.add_argument("--batch-size", default=32)

    eval_parser.set_defaults(func=_eval_model)
    return eval_parser


def _eval_model(_, args):
    test_file = args.filename
    vocab_file = args.vocab
    model_file = args.parameter_file
    batch_size = args.batch_size
    word_embeddings = None

    model_class = INCLUDED_MODELS.get(args.model_name)

    vocab = Vocabulary().load(vocab_file)
    model = model_class(vocab, word_embeddings)
    parser = Model(
        model,
        decoder="eisner",
        loss="kiperwasser",
        optimizer="adam",
        strategy="bucket",
        vocab=vocab,
    )

    parser.load_from_file(model_file)
    test_data = vocab.tokenize_conll(test_file)
    metrics = parser.evaluate(test_file, test_data, batch_size=batch_size)

    for key, value in metrics.items():
        print(key, round(value, 3))


def _get_run_parser(subparsers):
    eval_parser = subparsers.add_parser("run", help=EVAL_PARSER_HELP)
    eval_parser.add_argument("--model-name", required=True)
    eval_parser.add_argument("--test", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--parameter-file", required=True)
    eval_parser.add_argument("--vocab", required=True)
    eval_parser.add_argument("--no-gold")
    eval_parser.add_argument("--batch-size", default=32)

    eval_parser.set_defaults(func=_run_model)
    return eval_parser


def _run_model(_, args):
    run_file = args.test
    out_file = args.output
    vocab_file = args.vocab
    model_file = args.parameter_file
    batch_size = args.batch_size
    word_embeddings = None

    model_class = INCLUDED_MODELS.get(args.model_name)

    vocab = Vocabulary().load(vocab_file)
    model = model_class(vocab, word_embeddings)
    parser = Model(
        model,
        decoder="eisner",
        loss="kiperwasser",
        optimizer="adam",
        strategy="bucket",
        vocab=vocab,
    )

    parser.load_from_file(model_file)

    run_data = vocab.tokenize_conll(run_file)
    predictions = parser.run(run_data, batch_size)
    write_predictions_to_file(
        predictions, reference_file=run_file, output_file=out_file, vocab=vocab
    )

    print(">> Wrote predictions to conllu file %s" % out_file)


main()
