"""Script for running Dozat and Manning (2017.)"""

import argparse

import dynet as dy

from uniparse import Vocabulary, Model
from uniparse.models.dozat import DozatManning
from uniparse.callbacks import Callback, ModelSaveCallback


class UpdateParamsCallback(Callback):
    """Parameter decay callback."""

    def __init__(self, optimizer, learning_rate, decay, decay_steps):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._decay = decay
        self._decay_steps = decay_steps

    def on_batch_end(self, info):
        global_step = info["global_step"]
        if global_step % 2 == 0:
            new_lr = self._learning_rate * self._decay ** (
                global_step / self._decay_steps
            )
            self._optimizer.learning_rate = new_lr


def main():
    """Main function."""
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--train", required=True)
    argparser.add_argument("--dev", required=True)
    argparser.add_argument("--test", required=True)
    argparser.add_argument("--emb", dest="emb")
    argparser.add_argument("--epochs", dest="epochs", type=int, default=283)
    argparser.add_argument("--vocab_dest", dest="vocab_dest", required=True)
    argparser.add_argument("--model_dest", dest="model_dest", required=True)

    argparser.add_argument("--lstm_layers", dest="lstm_layers", type=int, default=3)
    argparser.add_argument("--dropout", type=int, default=0.33)

    arguments, _ = argparser.parse_known_args()

    # [Data]
    min_occur_count = 2
    train_file = arguments.train
    dev_file = arguments.dev
    vocab_destination = arguments.vocab_dest
    model_destination = arguments.model_dest

    # [Network]
    word_dims = 100
    tag_dims = 100
    lstm_hiddens = 400
    mlp_arc_size = 500
    mlp_rel_size = 100
    lstm_layers = arguments.lstm_layers
    dropout_emb = arguments.dropout
    dropout_lstm_input = arguments.dropout
    dropout_lstm_hidden = arguments.dropout
    dropout_mlp = arguments.dropout

    # [Hyperparameters for optimizer]
    learning_rate = 2e-3
    decay = 0.75
    decay_steps = 5000
    beta_1 = 0.9
    beta_2 = 0.9
    epsilon = 1e-12

    # [Run]
    batch_scale = 5000  # for scaled batching
    n_epochs = arguments.epochs

    vocab = Vocabulary()
    vocab = vocab.fit(train_file, arguments.emb, min_occur_count)
    embs = vocab.load_embedding(True) if arguments.emb else None

    vocab.save(vocab_destination)

    model = DozatManning(
        vocab,
        word_dims,
        tag_dims,
        dropout_emb,
        lstm_layers,
        lstm_hiddens,
        dropout_lstm_input,
        dropout_lstm_hidden,
        mlp_arc_size,
        mlp_rel_size,
        dropout_mlp,
        pretrained_embeddings=embs,
    )

    optimizer = dy.AdamTrainer(
        model.parameter_collection, learning_rate, beta_1, beta_2, epsilon
    )

    # Callbacks
    custom_learning_update_callback = UpdateParamsCallback(
        optimizer, learning_rate, decay, decay_steps
    )
    save_callback = ModelSaveCallback(model_destination)
    callbacks = [custom_learning_update_callback, save_callback]

    parser = Model(
        model,
        decoder="cle",
        loss="crossentropy",
        optimizer=optimizer,
        strategy="scaled_batch",
        vocab=vocab,
    )

    # Prep data
    training_data = vocab.tokenize_conll(arguments.train)
    dev_data = vocab.tokenize_conll(arguments.dev)
    test_data = vocab.tokenize_conll(arguments.test)

    parser.train(
        training_data,
        dev_file,
        dev_data,
        epochs=n_epochs,
        batch_size=batch_scale,
        callbacks=callbacks,
    )

    parser.load_from_file(model_destination)

    metrics = parser.evaluate(arguments.test, test_data, batch_size=batch_scale)
    test_uas = metrics["nopunct_uas"]
    test_las = metrics["nopunct_las"]

    print()
    print(metrics)
    print(">> Test score:", test_uas, test_las)


if __name__ == "__main__":
    main()
