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

if __name__ == '__main__':
    mkdirs_if_not_exists('results')
    i = 1
    for corpus in ['yelp', '20ng']:
        for iters in [1, 5]:
            for pretrained in ['models/wtv_model_cwiki_50perc.pkl', None]:
                for three_way in [True, False, None]:
                    for classifier in ['MultinomialNB', 'TaddyClassifier']:
                        if (three_way is not None) and corpus != 'yelp':
                            continue

                        output_dir = os.path.join('results', 'exp%d' % i)
                        mkdirs_if_not_exists(output_dir)

                        conf = deepcopy(template)
                        conf['output_dir'] = output_dir
                        conf['corpus'] = corpus
                        conf['taddy__iters'] = iters
                        conf['taddy__pretrained_pkl'] = pretrained
                        conf['taddy__specialised_pkl'] = 'models/specialised_wtv_%d.pkl' % i
                        conf['yelp__three_way'] = three_way
                        conf['clf__class'] = classifier

                        with open(os.path.join(output_dir, 'conf.txt'), 'w') as outf:
                            json.dump(conf, outf, indent=1)
                            i += 1
