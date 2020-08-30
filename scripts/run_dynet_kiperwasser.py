from uniparse import Vocabulary, Model
from uniparse.callbacks import TensorboardLoggerCallback, ModelSaveCallback
from uniparse.models.kiperwasser_dynet import DependencyParser


def train(
    train_file: str,
    validation_file: str,
    vocab_file: str,
    model_file: str,
    patience: int,
    batch_size: int,
    epochs: int,
    tensorboard_dir: str = None,
    embedding_file: str = None,
):

    vocab = Vocabulary()
    if embedding_file:
        vocab = vocab.fit(train_file, embedding_file)
        embs = vocab.load_embedding()
        print("shape", embs.shape)
    else:
        vocab = vocab.fit(train_file)
        embs = None

    # save vocab for reproducability later
    if vocab_file:
        print("> saving vocab to", vocab_file)
        vocab.save(vocab_file)

    # prep data
    print(">> Loading in data")
    training_data = vocab.tokenize_conll(train_file)
    dev_data = vocab.tokenize_conll(validation_file)

    # instantiate model
    model = DependencyParser(vocab, embs)

    callbacks = []
    tensorboard_logger = None
    if tensorboard_dir:
        tensorboard_logger = TensorboardLoggerCallback(tensorboard_dir)
        callbacks.append(tensorboard_logger)

    save_callback = ModelSaveCallback(model_file)
    callbacks.append(save_callback)

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
        validation_file,
        dev_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        patience=patience,
    )


def init_train_parser(subparsers):
    parser = subparsers.add_parser("train")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=-1)
    # optionals
    parser.add_argument("--no_update_pretrained_emb", action="store_true")
    parser.add_argument("--embs")
    parser.add_argument("--tensorboard")

    def f():
        train(
            train_file=args.train,
            validation_file=args.dev,
            vocab_file=args.vocab,
            model_file=args.model,
            patience=args.patience,
            epochs=args.epochs,
            tensorboard_dir=args.tensorboard,
            embedding_file=args.embs,
            batch_size=args.batch_size,
        )

    parser.set_defaults(func=f)


def evaluate(ud_file: str, vocab_file: str, model_file: str, batch_size: int = 32):
    vocab = Vocabulary().load(vocab_file)
    model = DependencyParser(vocab, embs=None)

    trainer = Model(
        model,
        decoder="eisner",
        loss="kiperwasser",
        optimizer="adam",
        strategy="bucket",
        vocab=vocab,
    )

    trainer.load_from_file(model_file)
    eval_data = vocab.tokenize_conll(ud_file)
    metrics = trainer.evaluate(ud_file, eval_data, batch_size)
    print(metrics)


def init_eval_parser(subparsers):
    parser = subparsers.add_parser("eval")
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)

    def f():
        evaluate(
            ud_file=args.data,
            vocab_file=args.vocab,
            model_file=args.model,
            batch_size=args.batch_size,
        )

    parser.set_defaults(func=f)


if __name__ == "__main__":
    import argparse

    description = "Script for train/evaluating/running Kiperwasser and Goldberg (2017)"
    parser = argparse.ArgumentParser(description)

    subparsers = parser.add_subparsers(dest="cmd", help="geagwa")
    subparsers.required = True

    for init_cmd in [init_train_parser, init_eval_parser]:
        init_cmd(subparsers)

    args = parser.parse_args()

    args.func()

