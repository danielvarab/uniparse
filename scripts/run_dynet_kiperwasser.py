"""Run script for training/evaluating/running Kiperwasser and Golberg (2017)."""

import argparse

from uniparse import Vocabulary, Model
from uniparse.callbacks import ModelSaveCallback
from uniparse.models.kiperwasser_dynet import DependencyParser
from uniparse.util import write_predictions_to_file

SUBPARSERS_HELP = "%(prog)s must be called with a command:"
TRAIN_PARSER_HELP = "Train a dependency parser."
EVAL_PARSER_HELP = "Evaluate dependency parser."
RUN_PARSER_HELP = "Run dependency parser."


def main():
    """Set up command line parser."""
    description = "CLI for training/evaluating/running Kiperwasser and Goldberg (2017)."
    parser = argparse.ArgumentParser(description=description)

    subparsers = parser.add_subparsers(help=SUBPARSERS_HELP, dest="train|dev|run")
    subparsers.required = True

    for subparsers_func in [_get_train_parser, _get_eval_model_parser, _get_run_parser]:
        subparsers_func(subparsers)

    # second parameter is unrecognized arguments
    args, _ = parser.parse_known_args()
    args.func(args)


def _get_train_parser(subparsers):
    """Create parser for the 'train' command."""
    train_parser = subparsers.add_parser("train", help=EVAL_PARSER_HELP)
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--dev", required=False)
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--vocab", required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.set_defaults(func=_train_model)
    return train_parser


def _train_model(args):
    train_file = args.train
    dev_file = args.dev
    epochs = args.epochs
    vocab_dest = args.vocab
    model_dest = args.model
    batch_size = args.batch_size
    embedding_file = None

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
    model = DependencyParser(vocab, word_embeddings)

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
    eval_parser = subparsers.add_parser("eval", help=EVAL_PARSER_HELP)
    eval_parser.add_argument("--filename", required=True)
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--vocab", required=True)
    eval_parser.add_argument("--batch-size", default=32)

    eval_parser.set_defaults(func=_eval_model)
    return eval_parser


def _eval_model(args):
    test_file = args.filename
    vocab_file = args.vocab
    model_file = args.model
    batch_size = args.batch_size
    word_embeddings = None

    vocab = Vocabulary().load(vocab_file)
    model = DependencyParser(vocab, word_embeddings)

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

    keys, values = zip(*metrics.items())
    print("\t".join(keys))
    print("\t".join([str(round(v, 3)) for v in values]))


def _get_run_parser(subparsers):
    eval_parser = subparsers.add_parser("run", help=EVAL_PARSER_HELP)
    eval_parser.add_argument("--filename", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--vocab", required=True)
    eval_parser.add_argument("--no-gold")
    eval_parser.add_argument("--batch-size", default=32)

    eval_parser.set_defaults(func=_run_model)
    return eval_parser


def _run_model(args):
    run_file = args.filename
    out_file = args.output
    vocab_file = args.vocab
    model_file = args.model
    batch_size = args.batch_size
    word_embeddings = None

    vocab = Vocabulary().load(vocab_file)
    model = DependencyParser(vocab, word_embeddings)
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


if __name__ == "__main__":
    main()
