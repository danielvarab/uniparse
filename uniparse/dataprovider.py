""""""

from collections import defaultdict, namedtuple

import numpy as np
import sklearn.utils

from sklearn.cluster import KMeans


def gen_pad_3d(x, padding_token):
    """Pad a 3D token matrix."""
    if not isinstance(x, list):
        x = list(x)

    max_word_len = max(len(word) for sentence in x for word in sentence)
    max_seq_len = max(len(s) for s in x)

    b, s = len(x), max_seq_len
    buff = np.empty((b, s, max_word_len), dtype=np.int32)
    buff.fill(padding_token)
    for i, sentence in enumerate(x):
        for j, char_ids in enumerate(sentence):
            buff[i, j, : len(char_ids)] = char_ids

    return buff


def gen_pad_2d(x, padding_token):
    """Pad a 2D matrix."""
    if not isinstance(x, list):
        x = list(x)

    batch_size = len(x)
    max_sentence_length = max(len(sentence) for sentence in x)
    buff = np.empty((batch_size, max_sentence_length))
    buff.fill(padding_token)
    for i, sentence in enumerate(x):
        buff[i, : len(sentence)] = sentence

    return buff


def split(l: list, batch_length: int):
    return [l[i : i + batch_length] for i in range(0, len(l), batch_length)]


Batch = namedtuple("Batch", ["x", "y"])


def build_from_clusters(
    scale_f, index_clusters, feature_clusters, padding_token, shuffle=True
):

    batches, indicies = [], []
    for len_key in feature_clusters.keys():
        values = feature_clusters[len_key]
        indices = index_clusters[len_key]

        # shuffle the entire cluster
        if shuffle:
            indices, values = sklearn.utils.shuffle(indices, values)

        batch_size = scale_f(values)
        indices = split(indices, batch_size)
        data_splits = split(values, batch_size)

        for batch in data_splits:
            # get the first four features
            # (conventionally set to :: form, lemma, tag, head, rel and take
            # advantage of them being well formed)
            features = map(lambda t: t[:5], batch)
            padded_features = gen_pad_3d(features, padding_token)

            # extract the character token ids
            chars = map(lambda t: t[5], batch)
            padded_chars = gen_pad_3d(chars, padding_token)

            # this purely assumes the order of how vocabulary provides the data
            # which in the format of ::
            # Tuple[List(b,n), List(b,n), List(b,n), List(b,n), List(b,n), List(b,n,d)]
            words, lemmas, tags, heads, rels = [
                padded_features[:, i, :] for i in range(5)
            ]

            # chars :: (batch_size, seq_len, longet_word)
            # words+ :: (batch_size, seq_len)
            _batch = Batch(x=[words, lemmas, tags, padded_chars], y=[heads, rels])
            batches.append(_batch)

        indicies.extend(indices)

    if shuffle:
        indices, batches = sklearn.utils.shuffle(indicies, batches)

    return indicies, batches


class BucketBatcher:
    def __init__(self, samples, padding_token):
        self._dataset = samples
        self._padding_token = padding_token

    def get_data(self, max_bucket_size, shuffle: bool = True):
        data = self._dataset

        len_clusters = defaultdict(list)
        index_clusters = defaultdict(list)

        # scan over dataset and add them to buckets of same length
        for sample_id, sample in enumerate(data):
            words, *_ = sample
            n = len(words)

            len_clusters[n].append(sample)
            index_clusters[n].append(sample_id)

        def batch_size_f(_):
            return max_bucket_size

        batch_indicies, batches = build_from_clusters(
            batch_size_f, index_clusters, len_clusters, self._padding_token, shuffle
        )

        return batch_indicies, batches


class ScaledBatcher:
    def __init__(self, samples, cluster_count, padding_token):
        self._dataset = samples
        self._padding_token = padding_token

        # map sentences to their lengths
        lengths = np.array([len(s[0]) for s in samples]).reshape(-1, 1)
        kmeans = KMeans(cluster_count)
        labels = kmeans.fit_predict(lengths)

        len_clusters = defaultdict(list)
        index_clusters = defaultdict(list)

        for sample_id, (sample, n) in enumerate(zip(samples, labels)):
            len_clusters[n].append(sample)
            index_clusters[n].append(sample_id)

        self._clusters = len_clusters
        self._indicies = index_clusters

    def get_data(self, scale: int, shuffle=True):
        def compute_batch_size(x):
            cluster_size = len(x)
            n = max(len(t[0]) for t in x)
            n_tokens = cluster_size * n

            num_of_splits = max(n_tokens // scale, 1)  # weighted batching
            return max(cluster_size // num_of_splits, 1)

        indicies, batches = build_from_clusters(
            compute_batch_size,
            self._indicies,
            self._clusters,
            self._padding_token,
            shuffle,
        )

        return indicies, batches


class VanillaBatcher(object):
    def __init__(self):
        # do we want this ? Runtime becomes somewhat unreliable because large sentences
        # scale. To use this one hasto be very conservative about batch size_
        # I think we do... but lets w8
        pass

    def get_data(self, batch_size, shuffle=True):
        raise NotImplementedError()
