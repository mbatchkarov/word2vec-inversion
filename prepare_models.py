import argparse
import glob
import json
import logging
import multiprocessing
from operator import itemgetter
import pickle
from copy import deepcopy

import numpy as np

from gensim.models import Word2Vec
from common import get_data


def _train(X, y, basemodel, taddy__specialised_pkl=None, **kwargs):
    labels = list(sorted(set(y)))
    logging.info('Training specialised models for classes %r', labels)
    if not taddy__specialised_pkl:
        raise ValueError('This needs to be provided yo')

    class_specific_models = [deepcopy(basemodel) for _ in range(len(labels))]
    for i, label in enumerate(labels):
        indices = np.where(y == label)[0]
        slist = itemgetter(*indices)(X)
        class_specific_models[i].train(slist, total_examples=len(indices))
        logging.info('Did %d', i)

    with open(taddy__specialised_pkl, 'wb') as outfile:
        logging.info('Dumping to disk')
        pickle.dump(class_specific_models, outfile)


def _new_model(data, iters=3, **kwargs):
    # create a w2v learner
    wtv = Word2Vec(workers=multiprocessing.cpu_count(),  # use them cores
                   iter=iters,  # sweeps of SGD through the data; more is better
                   min_count=15)
    wtv.build_vocab(data)
    return wtv


def _pretrained_model(iters=3, taddy__pretrained_pkl=None, **kwargs):
    logging.info('Reading pretrained wtv pickle')
    with open(taddy__pretrained_pkl, 'rb') as infile:
        basemodel = pickle.load(infile)
    basemodel.iter = iters
    return basemodel


def _run_single(conf_file):
    logging.info('---------------\nLoaded %s', conf_file)
    with open(conf_file) as inf:
        conf = json.load(inf)

    X, y, _, _ = get_data(**conf)
    if conf['taddy__pretrained_pkl']:
        _train(X, y, _pretrained_model(**conf), **conf)
    else:
        _train(X, y, _new_model(X, **conf), **conf)


def _run_all():
    for conf_file in sorted(glob.glob('results/**/conf.txt')):
        _run_single(conf_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', default=False,
                       help='Whether to run ALL available jobs')

    group.add_argument('--id', type=int,
                       help='Run only this experiment')

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s : %(message)s",
                        datefmt='%m-%d %H:%M')

    parameters = parser.parse_args()
    if parameters.all:
        _run_all()
    else:
        _run_single('results/exp%d/conf.txt' % parameters.id)
