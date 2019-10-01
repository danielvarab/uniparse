"""Run script for a proof of concept MST parser using UniParse."""
import os
import time

import argparse

import numpy as np
import sklearn.utils

from uniparse import Vocabulary
from uniparse.util import write_predictions_to_file
from uniparse.evaluate import evaluate_files

from uniparse.models.mst import MST
from uniparse.models.mst_encode import BetaEncodeHandler


def pre_encode(encoder, samples, accumulate_vocab=False):
    """Preencode dataset."""
    if accumulate_vocab:
        encoder.unlock_feature_space()

    encoded_dataset = []
    for _step, (words, _, tags, heads, rels, _) in enumerate(samples):
        words = np.array(words, dtype=np.uint64)
        tags = np.array(tags, dtype=np.uint64)
        target_arcs = np.array(heads, dtype=np.int64)
        target_rels = np.array(rels, dtype=np.int64)

        # we do this to initialize the encoders vocab.
        _, _ = encoder(words, tags, target_arcs, target_rels)

        # this is needed
        encoded_samples = (words, tags, target_arcs, target_rels)
        encoded_dataset.append(encoded_samples)

    # gotta do this to ensure not adding more stuff
    if accumulate_vocab:
        encoder.lock_feature_space()

    return encoded_dataset


def evaluate(encoder, parser, vocab, gold_file, data, name):
    """Evaluate model."""
    predictions = []
    for step, (forms, tags, _, _) in enumerate(data):
        dev_arc_edges, dev_rel_edges = encoder(forms, tags)
        dev_pred_arcs, dev_pred_rels, _, _ = parser(
            dev_arc_edges, dev_rel_edges, None, None
        )
        ns = [step] * len(dev_pred_arcs[1:])
        prediction_tuple = (ns, dev_pred_arcs[1:], dev_pred_rels[1:])
        predictions.append(prediction_tuple)

    output_file = "dev_epoch_%s.txt" % name
    write_predictions_to_file(
        predictions, reference_file=gold_file, output_file=output_file, vocab=vocab
    )

    metrics = evaluate_files(output_file, gold_file)
    os.system("rm %s" % output_file)

    return predictions, metrics


def train(
    encoder, params, vocab, training_data, dev_data_file, dev_data, epochs, param_file
):
    """Training procedure."""
    start = step_time = time.time()
    max_uas = 0.0
    uas, ras = [], []
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        print("=======================")

        # shuffle the dataset on each epoch
        training_data = sklearn.utils.shuffle(training_data)

        for step, (forms, tags, target_arcs, target_rels) in enumerate(training_data):
            arc_edges, rel_edges = encoder(forms, tags)
            _, _, sample_uas, sample_ras = params(
                arc_edges, rel_edges, target_arcs, target_rels
            )

            uas.append(sample_uas)
            ras.append(sample_ras)
            if step % 500 == 0 and step > 0:
                mean_uas = np.mean(uas)
                mean_ras = np.mean(ras)

                time_spent = round(time.time() - step_time, 3)
                val_tuple = (step, round(mean_uas, 3), round(mean_ras, 3), time_spent)
                print("> Step %d UAS: %f LAS %f Time %f" % val_tuple)

                step_time = time.time()
                uas, ras = [], []

        # time to evaluate
        print(">> Done with epoch %d. Evaluating on dev..." % epoch)
        _, metrics = evaluate(encoder, params, vocab, dev_data_file, dev_data, epoch)
        print(">> dev epoch %d" % epoch)
        print(metrics)
        print()

        nopunct_uas = metrics["nopunct_uas"]
        if nopunct_uas > max_uas:
            np.save(param_file, params.W)
            print(">> saved to", param_file)
            max_uas = nopunct_uas

    print(">> Finished. Time spent", time.time() - start)


if __name__ == "__main__":
    ARGPARSER = argparse.ArgumentParser()
    ARGPARSER.add_argument("--train", required=True)
    ARGPARSER.add_argument("--dev", required=True)
    ARGPARSER.add_argument("--test", required=True)
    ARGPARSER.add_argument("--model", required=True)

    ARGUMENTS, UNK = ARGPARSER.parse_known_args()

    TRAIN_FILE = ARGUMENTS.train
    DEV_FILE = ARGUMENTS.dev
    TEST_FILE = ARGUMENTS.test
    MODEL_FILE = ARGUMENTS.model
    N_EPOCHS = 5

    VOCAB = Vocabulary()
    VOCAB.fit(TRAIN_FILE)

    print("> Loading in data")
    TRAIN = VOCAB.tokenize_conll(ARGUMENTS.train)
    DEV = VOCAB.tokenize_conll(ARGUMENTS.dev)
    TEST = VOCAB.tokenize_conll(ARGUMENTS.test)

    ENCODER = BetaEncodeHandler()
    print("> Pre-encoding edges")
    START_TIME = time.time()
    TRAIN = pre_encode(ENCODER, TRAIN, accumulate_vocab=True)
    DEV = pre_encode(ENCODER, DEV)
    TEST = pre_encode(ENCODER, TEST)
    print(">> Done pre-encoding edges", time.time() - START_TIME)

    # 5m is completely arbitrary but fits all features for PTB.
    # TODO: Infer this from the encoder by letting it grow
    PARAMS = MST(5_000_000)

    # Train model
    train(ENCODER, PARAMS, VOCAB, TRAIN, DEV_FILE, DEV, N_EPOCHS, MODEL_FILE)

    # populate with best parameters
    PARAMS.W = np.load("%s.npy" % MODEL_FILE)

    print(">> Time to evaluate on test set")
    _, TEST_METRICS = evaluate(ENCODER, PARAMS, VOCAB, TEST_FILE, TEST, "test")
    print(TEST_METRICS)
