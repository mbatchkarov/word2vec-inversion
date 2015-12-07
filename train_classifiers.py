import argparse
import base64
import glob
import json
import logging
import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from common import get_data
from taddy_classifier import TaddyClassifier


def get_classifier(clf__class=None, **kwargs):
    models = {
        'MultinomialNB': MultinomialNB,
        'LinearSVC': LinearSVC,
        'TaddyClassifier': TaddyClassifier
    }

    if 'taddy' in clf__class.lower():
        model = TaddyClassifier(**kwargs)
        return model
    else:
        pipe = Pipeline([('vectorizer', CountVectorizer()),
                         ('clf', models[clf__class]())])
        return pipe


def get_classifier_params_for_grid_search(clf__class=None, **kwargs):
    params = {
        'MultinomialNB': {'clf__alpha': np.logspace(-5, 0, num=6, base=10)},
        'LinearSVC': {'clf__C': np.logspace(-5, 7, num=13, base=10)},
        'TaddyClassifier': {}
    }
    return params[clf__class]


def _run_single(conf_file):
    logging.info('---------------\nLoaded %s', conf_file)
    with open(conf_file) as inf:
        conf = json.load(inf)

    Xtr, ytr, Xev, yev = get_data(**conf)
    clf = get_classifier(**conf)
    # cd = RandomizedSearchCV(clf, param_distributions=get_classifier_params_for_grid_search(**conf),
    #                         n_iter=10, n_jobs=4, cv=3, verbose=5)
    cd = GridSearchCV(clf, get_classifier_params_for_grid_search(**conf),
                      n_jobs=5, cv=5, verbose=2)
    cd.fit(Xtr, ytr)
    preds = cd.best_estimator_.predict(Xev)

    d = dict()
    d['predictions'] = base64.b64encode(preds).decode('utf8')
    # to decode:
    # r = base64.decodebytes(s)
    # q = np.frombuffer(r, dtype=np.float64)
    d['best_cv_score'] = cd.best_score_
    d.update(cd.best_params_)

    with open(os.path.join(conf['output_dir'], 'output.txt'), 'w') as outf:
        json.dump(d, outf, indent=1)


def _run_all():
    conf_files = sorted(glob.glob('results/**/conf.txt'))
    logging.info('Running all %d classification experiments', len(conf_files))
    for conf_file in conf_files:
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
        logging.info('Running a single classification experiment')
        _run_single('results/exp%d/conf.txt' % parameters.id)
