# UniParse - A framework for graph-based dependency parsing.
UniParse is a universal graph-based modular parsing framework, for quick prototyping and comparison of parsers and parser components. With this framework we provide a collection of helpful tools and implementations that can assist the process of developing graph based dependency parsers.


## Installation
Since UniParse contains cython code that depends on numpy, installation must be done in two steps.
```bash
# (1)
pip install -r requirements.txt

# (2)
pip install [-e] .  # include -e if you'd like to be able to modify framework code
```

### [Neural Models]
UniParse includes a small collection of state-of-the-art neural models that are implemented using a high level model wrapper that should reduce development time significantly. This high-level model wrapper component currently supports two neural backends, namely: DyNet and PyTorch. One of these libraries are required to use the model wrapper.

```bash
# uncomment desired (if any) backend
# pip install dynet>=2.1
# pip install torch>=1.0
```

## Components
### Blazing-fast decoders

| Algorithm         |     en_ud     |    en_ptb    |  sentences/s | % faster |
| ----------------- | ------------- | ------------ | ------------ | -------- |
| CLE    (Generic)  |     19.12     |     93.8     | ~ 404        |   -      |
| Eisner (Generic)  |     96.35     |     479.1    | ~ 80         |   -      |
| CLE    (UniParse) |     1.764     |     6.98     | ~ 5436       |   1345%  |
| Eisner (UniParse) |     1.49      |     6.31     | ~ 6009       |   7500%  |

```python
import numpy as np
from uniparse.decoders import eisner, cle

score_matrix = np.eye(10, k=-1)
eisner(score_matrix)
# > array([-1,  2,  3,  4,  5,  6,  7,  8,  9,  0])

cle(score_matrix)
# > array([-1,  2,  3,  4,  5,  6,  7,  8,  9,  0])
```

### Evaluation
UniParse includes an evaluation script that works from within the framework, as well as by itself. For the former:

```python
from uniparse.evaluate import conll17_eval  # Wrapped UD evaluation script
from uniparse.evaluate import perl_eval  # Wrapped CONLL2006/2007 perl script. Ignores unicode punctuations (used for SOTA reports)
from uniparse.evaluate import evaluate_files  # UniParse rewritten evaluation. Provides scores with and without punctuation.


conll17_eval(test_file, gold_reference)
# > {"uas": ..., "las": ...)
metrics2 = perl_eval(test_file, gold_reference)
# > {"uas": ..., "las": ...)
metrics3 = evaluate_files(test_file, gold_reference)
# > {
#   "uas": .., 
#   "las": .., 
#   "nopunct_uas": .., 
#   "nopunct_las": .., 
# }
```

... and for the latter, please copy the following path to a desired location `uniparse/evaluate/uniparse_evaluate.py` and use by running 
```bash
python uniparse_evaluate.py --test [FILENAME.CONLLU] --gold [GOLD_REFERENCE.CONLLU]
```

### Vocabulary
```python
from uniparse import Vocabulary

vocab = Vocabulary().fit(CONLL_FILE, EMBEDING_FILE)
data = vocab.tokenize_conll(TRAIN_FILE)
word_ids, lemma_ids, upos_ids, gold_arcs, gold_rels, chars_ids = data[0]
```

### Model Wrapper
```python
vocab = ... # as above
params = ... # included or custom model 
parser = Model(
    PARAMETERS,
    decoder="eisner",
    loss="kiperwasser",
    optimizer="adam",
    strategy="bucket",
    vocab=vocab,
)
parser.train(
    train_data,
    dev_filename,
    dev_data,
    epochs=epochs,
    batch_size=32,
    patience=3,
)

predictions = parser.run(test_file, test_data)
metrics = parser.evaluate(test_file, test_data)
```

### Batching
```python
from uniparse.dataprovider import ScaledBatcher, BucketBatcher

dataprovider1 = BucketBatcher(data, padding_token=vocab.PAD)
idx, batches = dataprovider.get_data(scale, shuffle=True)

dataprovider2 = ScaledBatcher(data, padding_token=vocab.PAD)
idx, batches = dataprovider.get_data(shuffle=True)
```

## Included models
With Uniparse we include a set of state-of-the-art neural models composed entirely by UniParse components, as well as training scripts. We invite you to use these models as a starting point and freely rerun, modify and extend the models for further development or evaluation.

You'll find all the training scripts under `/scripts`

```bash
# example 
python scripts/run_dynet_kiperwasser.py --train TRAIN_FILE --dev DEV_FILE --test TEST_FILE --model_dest MODEL --epochs 30
```


| Model                          |   Language    |   UAS w.p.   |   LAS w.p.   |   UAS n.p.   |   LAS n.p.  |
| ------------------------------ | ------------- | ------------ | ------------ | ------------ | ----------- |
| Kiperwasser & Goldberg (2016)  |               |              |              |              |             |
|                                |  Danish (UD)  | 83.18%       | 79.57%       | 83.67%       | 79.47%      |
|                                |  English (UD) | 87.06%       | 84.68%       | 88.08%       | 85.43%      |
|                                | English (PTB) | 92.56%       | 91.17%       | 93.14%       | 91.57%      |
| Dozat & Manning (2017)         |    -          |              |              |              |             |
|                                |  Danish (UD)  | 87.42%       | 84.98%       | 87.84%       | 84.99%      |
|                                |  English (UD) | 90.74        | 89.01%       | 91.47        | 89.38       |
|                                | English (PTB) | 94.91%       | 93.70%       | 95.43%       | 94.06%      |
| Nguyen and Verspoor (2018)     | -             |              |              |              |             |
|                                |  Danish (UD)  | TBA          | TBA          | TBA          | TBA         |
|                                |  English (UD) | TBA          | TBA          | TBA          | TBA         |
|                                | English (PTB) | TBA          | TBA          | TBA          | TBA         |
| MST (non-neural)               | -             |              |              |              |             |
|                                |  Danish (UD)  | 67.17        | 55.52        | 68.80        | 55.30       |
|                                |  English (UD) | 73.47        | 65.20        | 75.55        | 66.25       |
|                                | English (PTB) | 74.00        | 63.60        | 76.07        | 64.67       |

with `w.p.` and `n.p.` denoting 'with punctuation', 'no punctuation'. No punctuation follows the rule of excluding modifier tokens consisting entirely of unicode punctuation characters; this definition is standard in current research.

*Note that these models must be trained. We are actively working on providing downloadable pretrained models.*


### PTB split
Since the splitting of Penn treebank files is not fully standerdised we indicate the split used in experiments from [our paper](https://www.aclweb.org/anthology/W19-6149), as well as supporting literature.
Note that published model performances for systems we re-implement and distribute with UniParse may use different splits, which have a observerable impact on performance. Specifically, we note that [Dozat and Manning](https://arxiv.org/pdf/1611.01734.pdf)'s parser performs differently even using under splits than reported in their paper.

|   Train   |  Dev   |  Test  | Discard |
|:---------:|:------:|:------:|:-------:|
| `{02-21}` | `{22}` | `{23}` | `{00}`  | 


# Citation
If you use UniParse, please cite our [paper](https://www.aclweb.org/anthology/W19-6149).

```
@inproceedings{varab-schluter-2019-uniparse,
    title = "{U}ni{P}arse: A universal graph-based parsing toolkit",
    author = "Varab, Daniel  and Schluter, Natalie",
    booktitle = "Proceedings of the 22nd Nordic Conference on Computational Linguistics",
    month = "30 " # sep # " {--} 2 " # oct,
    year = "2019",
    address = "Turku, Finland",
    publisher = {Link{\"o}ping University Electronic Press},
    url = "https://www.aclweb.org/anthology/W19-6149",
    pages = "406--410",
    abstract = "This paper describes the design and use of the graph-based parsing framework and toolkit UniParse, released as an open-source python software package. UniParse as a framework novelly streamlines research prototyping, development and evaluation of graph-based dependency parsing architectures. UniParse does this by enabling highly efficient, sufficiently independent, easily readable, and easily extensible implementations for all dependency parser components. We distribute the toolkit with ready-made configurations as re-implementations of all current state-of-the-art first-order graph-based parsers, including even more efficient Cython implementations of both encoders and decoders, as well as the required specialised loss functions.",
}
```
