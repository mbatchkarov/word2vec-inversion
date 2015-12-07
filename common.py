import logging
from zipfile import ZipFile
import json
import re

import numpy as np
from sklearn.datasets import fetch_20newsgroups


def _get_yelp_reviews_as_dict(label):
    """
    Yield yelp reviews as a dict {x: list of tokenised sentences, y: label}
    :param label: 'training' or 'test'
    :return: :raise StopIteration:
    """

    def _clean_yelp(text):
        # cleaner (order matters)
        contractions = re.compile(r"'|-|\"")
        # all non alphanumeric
        symbols = re.compile(r'(\W+)', re.U)
        # single character removal
        singles = re.compile(r'(\s\S\s)', re.I | re.U)
        # separators (any whitespace)
        seps = re.compile(r'\s+')

        text = text.lower()
        text = contractions.sub('', text)
        text = symbols.sub(r' \1 ', text)
        text = singles.sub(' ', text)
        text = seps.sub(' ', text)
        return text

    # sentence splitter
    alteos = re.compile(r'([!\?])')

    def _yelp_sentences(l):
        l = alteos.sub(r' \1 .', l).rstrip("(\.)*\n")
        return l.split(".")

    with ZipFile("data/yelp_%s_set.zip" % label, 'r') as zf:
        with zf.open("yelp_%s_set/yelp_%s_set_review.json" % (label, label)) as f:
            for i, line in enumerate(f):
                if i > 1000:
                    raise StopIteration
                rev = json.loads(line.decode())
                yield {'y': rev['stars'], \
                       'x': [_clean_yelp(s).split() for s in _yelp_sentences(rev['text'])]}


def _get_yelp_data(three_way=False):
    """
    Return yelp data as a tuple of
        - list of training documents (untokenised, unsegmented)
        - np.array of training labels
        - list of test documents
        - np.array of test documents
    :rtype : tuple
    """

    def _read(setname):
        X, y = [], []
        for doc in _get_yelp_reviews_as_dict(setname):
            words = []
            for sent in doc['x']:
                words.extend(sent)
            X.append(' '.join(words))
            y.append(doc['y'])

        return X, np.array(y)

    tr_text, ytr = _read('training')
    ev_text, yev = _read('test')
    if three_way:
        for y in [ytr, yev]:
            y[y <= 2] = 1
            y[y == 2] = 2
            y[y >= 4] = 4
    return tr_text, ytr, ev_text, yev


def _get_20ng_data():
    cats = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'talk.religion.misc']
    train = fetch_20newsgroups(data_home='data', subset='train', categories=cats)
    test = fetch_20newsgroups(data_home='data', subset='test', categories=cats)

    return train.data, train.target, test.data, test.target


def get_data(corpus, **kwargs):
    logging.info('Reading labelled corpus')
    loaders = {
        'yelp': _get_yelp_data,
        '20ng': _get_20ng_data
    }

    return loaders[corpus]()
