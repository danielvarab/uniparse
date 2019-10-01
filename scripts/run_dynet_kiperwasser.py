import argparse

from uniparse import Vocabulary, Model
from uniparse.callbacks import TensorboardLoggerCallback, ModelSaveCallback
from uniparse.models.kiperwasser_dynet import DependencyParser


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        dest="train",
        help="Annotated CONLL train file",
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "--dev",
        dest="dev",
        help="Annotated CONLL dev file",
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "--test",
        dest="test",
        help="Annotated CONLL dev test",
        metavar="FILE",
        required=True,
    )
    parser.add_argument("--epochs", dest="epochs", type=int, default=30)
    parser.add_argument("--tb_dest", dest="tb_dest")
    parser.add_argument("--vocab_dest", dest="vocab_dest")
    parser.add_argument("--model_dest", dest="model_dest", required=True)
    parser.add_argument(
        "--embs", dest="embs", help="pre-trained embeddings file name", required=False
    )
    parser.add_argument(
        "--no_update_pretrained_emb",
        dest="no_update_pretrained_emb",
        help="don't update the pretrained embeddings during training",
        default=False,
        action="store_true",
    )
    parser.add_argument("--patience", dest="patience", type=int, default=-1)

    arguments, unknown = parser.parse_known_args()

    n_epochs = arguments.epochs

    vocab = Vocabulary()
    if arguments.embs:
        vocab = vocab.fit(arguments.train, arguments.embs)
        embs = vocab.load_embedding()
        print("shape", embs.shape)
    else:
        vocab = vocab.fit(arguments.train)
        embs = None

    # save vocab for reproducability later
    if arguments.vocab_dest:
        print("> saving vocab to", arguments.vocab_dest)
        vocab.save(arguments.vocab_dest)

    # prep data
    print(">> Loading in data")
    training_data = vocab.tokenize_conll(arguments.train)
    dev_data = vocab.tokenize_conll(arguments.dev)
    test_data = vocab.tokenize_conll(arguments.test)

    # instantiate model
    model = DependencyParser(vocab, embs)

    callbacks = []
    tensorboard_logger = None
    if arguments.tb_dest:
        tensorboard_logger = TensorboardLoggerCallback(arguments.tb_dest)
        callbacks.append(tensorboard_logger)

    save_callback = ModelSaveCallback(arguments.model_dest)
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
        arguments.dev,
        dev_data,
        epochs=n_epochs,
        batch_size=32,
        callbacks=callbacks,
        patience=arguments.patience,
    )
    parser.load_from_file(arguments.model_dest)

    metrics = parser.evaluate(arguments.test, test_data, batch_size=32)
    test_UAS = metrics["nopunct_uas"]
    test_LAS = metrics["nopunct_las"]

    print(metrics)

    if arguments.tb_dest and tensorboard_logger:
        tensorboard_logger.raw_write("test_UAS", test_UAS)
        tensorboard_logger.raw_write("test_LAS", test_LAS)

    print()
    print(">>> Model maxed on dev at epoch", save_callback.best_epoch)
    print(">>> Test score:", test_UAS, test_LAS)


if __name__ == "__main__":
    main()
