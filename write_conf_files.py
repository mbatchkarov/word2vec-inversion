from copy import deepcopy
import json
import os

__author__ = 'mmb28'


def mkdirs_if_not_exists(dir):
    """
    Creates a directory (and all intermediate directories) if it doesn't exists.
    Behaves like mkdir -p, and is prone to race conditions

    Source: http://stackoverflow.com/q/273192/419338
    :param dir:
    :return:
    """
    if not (os.path.exists(dir) and os.path.isdir(dir)):
        os.makedirs(dir)


template = {
    'output_dir': None,
    'corpus': None,
    'taddy__iters': None,
    'taddy__pretrained_pkl': None,
    'taddy__specialised_pkl': None,
    'yelp__three_way': None,
    'clf__class': None,
}


def traditional_classifier_experiments():
    for corpus in ['yelp', '20ng']:
        for three_way in [True, False, None]:
            for classifier in ['MultinomialNB', 'LinearSVC']:
                if (three_way is not None) and corpus != 'yelp':
                    continue

                conf = deepcopy(template)
                conf['corpus'] = corpus
                conf['yelp__three_way'] = three_way
                conf['clf__class'] = classifier

                yield conf


def taddy_experiments(first_id):
    i = first_id
    pretrained = 'models/wtv_model_cwiki_50perc.pkl'
    classifier = 'TaddyClassifier'
    for corpus in ['yelp', '20ng']:
        for iters in [1, 10, 50]:
            for three_way in [True, False, None]:
                if (three_way is not None) and corpus != 'yelp':
                    continue

                conf = deepcopy(template)
                conf['corpus'] = corpus
                conf['taddy__iters'] = iters
                conf['taddy__pretrained_pkl'] = pretrained
                conf['taddy__specialised_pkl'] = 'models/specialised_wtv_%d.pkl' % i
                conf['yelp__three_way'] = three_way
                conf['clf__class'] = classifier
                i += 1
                yield conf


def all_experiments():
    experiments = []
    experiments.extend(traditional_classifier_experiments())
    experiments.extend(taddy_experiments(len(experiments) + 1))
    return experiments

def write_conf(conf, output_dir):
    mkdirs_if_not_exists(output_dir)
    with open(os.path.join(output_dir, 'conf.txt'), 'w') as outf:
        conf['output_dir'] = output_dir
        json.dump(conf, outf, indent=1)


if __name__ == '__main__':
    mkdirs_if_not_exists('results')

    for i, conf in enumerate(all_experiments()):
        output_dir = os.path.join('results', 'exp%d' % (i + 1))
        write_conf(conf, output_dir)
