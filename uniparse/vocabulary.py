"""Vocabulary module containing the Vocabulary class."""

from collections import Counter

import re
import pickle

import numpy as np

from .util import validate_line


class Vocabulary:
    """
    Class responsible for reading in files, preprocessing and managing tokens.
    """

    # Reserved token mappings
    PAD = 0
    ROOT = 1
    OOV = UNK = 2

    def __init__(self):
        self._id2word = None
        self._word2id = None
        self._lemma2id = None
        self._id2lemma = None
        self._tag2id = None
        self._id2tag = None
        self._rel2id = None
        self._id2rel = None
        self._id2freq = None
        self._id2char = None
        self._char2id = None
        self._words_in_train_data = None
        self._pret_file = None

    @staticmethod
    def _normalize_word(word):
        match = re.match("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+", word)
        return "NUM" if match else word.lower()

    def fit(self, input_file, pretrained_embeddings=None, min_occur_count=0):
        """Fit vocabulary on CONLLU file."""
        word_counter, lemma_set, tag_set, rel_set, char_set = self._collect_tokens(
            input_file
        )

        self._id2word = ["<pad>", "<root>", "<unk>"]
        self._id2lemma = ["<pad>", "<root>", "<unk>"]
        self._id2tag = ["<pad>", "<root>", "<unk>"]
        self._id2rel = ["<pad>", "root"]
        self._id2char = ["<pad>", "root", "<unk>"]
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)

        # add dataset tokens
        self._id2char += list(char_set)
        self._id2lemma += list(lemma_set)
        self._id2tag += list(tag_set)
        self._id2rel += list(rel_set)

        def reverse(x):
            """Map value to index."""
            return dict(zip(x, range(len(x))))

        self._word2id = reverse(self._id2word)
        self._lemma2id = reverse(self._id2lemma)
        self._tag2id = reverse(self._id2tag)
        self._rel2id = reverse(self._id2rel)
        self._char2id = reverse(self._id2char)
        self._id2freq = {wid: word_counter[word] for word, wid in self._word2id.items()}
        self._words_in_train_data = len(self._id2word)

        self._pret_file = None
        if pretrained_embeddings:
            self._pret_file = pretrained_embeddings
            self._add_pret_words(pretrained_embeddings)

        return self

    def load(self, filename):
        """Load pickled vocabulary from disk."""
        with open(filename, "rb") as fh:
            tmp_dict = pickle.load(fh)

            self.__dict__.update(tmp_dict)
        return self

    def save(self, filename):
        """Save vocabulary to disk."""
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f, 2)
        return self

    def load_dataset(self, input_file, tokenize=True):
        word_root, lemma_root, tag_root, rel_root = [self.ROOT] * 4
        char_root = [self.ROOT]
        root_head = -1

        sents = []
        words, lemmas, tags, heads, rels, chars = (
            [word_root],
            [lemma_root],
            [tag_root],
            [root_head],
            [rel_root],
            [char_root],
        )

        with open(input_file, encoding="UTF-8") as dataset_handler:
            for line in dataset_handler.readlines():
                line = line.rstrip()
                if not line:  # empty line
                    sent = (words, lemmas, tags, heads, rels, chars)
                    sents.append(sent)

                    words, lemmas, tags, heads, rels, chars = (
                        [word_root],
                        [lemma_root],
                        [tag_root],
                        [root_head],
                        [rel_root],
                        [char_root],
                    )
                else:
                    # parse line and map to word/token and add to sentence
                    line_feats = self._parse_conll_line(line, tokenize=tokenize)
                    sentence_features = [words, lemmas, tags, heads, rels, chars]
                    for s_features, w_feature in zip(sentence_features, line_feats):
                        s_features.append(w_feature)
        return sents

    def _collect_tokens(self, input_file):
        word_counter = Counter()
        lemma_set = set()
        tag_set = set()
        rel_set = set()
        char_set = set()
        with open(input_file, encoding="UTF-8") as fh:
            for line in fh.readlines():
                if not validate_line(line):
                    continue

                info = line.strip().split()
                if len(info) == 10:
                    word, lemma, tag, _, rel, chars = self._parse_conll_line(
                        info, tokenize=False
                    )
                    word_counter[word] += 1
                    lemma_set.add(lemma)
                    tag_set.add(tag)
                    char_set.update(chars)
                    if rel != "root":
                        rel_set.add(rel)

        return word_counter, lemma_set, tag_set, rel_set, char_set

    def _add_pret_words(self, embedding_file):
        words_in_train_data = set(self._word2id.keys())
        offset = max(self._word2id.values()) + 1
        counter = 0

        with open(embedding_file, encoding="UTF-8") as f:
            for line in f.readlines():
                line = line.strip().split()
                if not line:
                    continue

                word = line[0]
                if word in words_in_train_data:
                    continue
                self._word2id[word] = offset + counter
                counter += 1

    def load_embedding(self, variance_normalize=False):
        """ load embeddings """

        assert self._pret_file is not None, "no embedding to load...."

        embs = [[]] * len(self._word2id.keys())
        vector = None
        with open(self._pret_file, encoding="UTF-8") as f:
            print(">> Loading embedding vectors")
            for i, line in enumerate(f.readlines(), start=1):
                line = line.strip().split()
                if not line:
                    continue

                word, vector = line[0], line[1:]
                # word_id = self._word2id[word]

            print(f">> Done loading embeddings ({i})")

        emb_size = len(vector)
        for idx, emb in enumerate(embs):
            if not emb:
                embs[idx] = np.zeros(emb_size)
        pret_embs = np.array(embs, dtype=np.float32)

        if variance_normalize:
            pret_embs /= np.std(pret_embs)

        return pret_embs

    def tokenize_conll(self, file):
        """Maps string dataset to integer token lookups."""
        # TODO: Rename to load_conllu
        return self._read_conll(file, tokenize=True)

    def _parse_conll_line(self, info, tokenize):
        word = info[1].lower()
        lemma = info[2].lower()
        tag = info[3]
        head = int(info[6])
        rel = info[7]
        chars = list(info[1])

        word = self._normalize_word(word)
        if tokenize:
            word = self._word2id.get(word, self.OOV)
            lemma = self._lemma2id.get(lemma, self.OOV)
            tag = self._tag2id[tag]
            rel = self._rel2id.get(rel, self.OOV)

            chars = [self.char2id(c) for c in chars]

        return word, lemma, tag, head, rel, chars

    def _read_conll(self, input_file, tokenize=True):
        word_root = self.ROOT
        lemma_root = self.ROOT
        tag_root = self.ROOT
        rel_root = self.ROOT
        char_root = [self.ROOT]
        root_head = -1

        sents = []
        words, lemmas, tags, heads, rels, chars = (
            [word_root],
            [lemma_root],
            [tag_root],
            [root_head],
            [rel_root],
            [char_root],
        )

        with open(input_file, encoding="UTF-8") as fh:
            for line in fh.readlines():
                if not validate_line(line):
                    continue

                info = line.strip().split("\t")
                if len(info) == 10:
                    word, lemma, tag, head, rel, characters = self._parse_conll_line(
                        info, tokenize=tokenize
                    )
                    words.append(word)
                    lemmas.append(lemma)
                    tags.append(tag)
                    heads.append(head)
                    rels.append(rel)
                    chars.append(characters)
                else:
                    sent = (words, lemmas, tags, heads, rels, chars)
                    sents.append(sent)

                    words, lemmas, tags, heads, rels, chars = (
                        [word_root],
                        [lemma_root],
                        [tag_root],
                        [root_head],
                        [rel_root],
                        [char_root],
                    )

        return sents

    def word2id(self, x):
        return self._word2id.get(x, self.OOV)

    def id2word(self, x):
        return self._id2word[x]

    def rel2id(self, x):
        return self._rel2id[x]

    def id2rel(self, x):
        return self._id2rel[x]

    def tag2id(self, x):
        return self._tag2id.get(x, self.OOV)

    def char2id(self, x):
        return self._char2id.get(x, self.OOV)

    @property
    def wordid2freq(self):
        return self._id2freq

    @property
    def PUNCT(self):
        return self._tag2id["PUNCT"]

    @property
    def words_in_train(self):
        return self._words_in_train_data

    @property
    def vocab_size(self):
        return len(self._word2id)

    @property
    def upos_size(self):
        return len(self._tag2id)

    @property
    def tag_size(self):
        return len(self._tag2id)

    @property
    def char_size(self):
        return len(self._char2id)

    @property
    def label_count(self):
        return len(self._rel2id)

    @property
    def rel_size(self):
        return len(self._rel2id)
