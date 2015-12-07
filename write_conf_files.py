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
    'iters': None,
    'pretrained_pkl': None,
    'specialised_pkl': None,
}

if __name__ == '__main__':
    mkdirs_if_not_exists('results')
    i = 1
    for corpus in ['yelp', '20ng']:
        for iters in [1, 5, 10, 20]:
            for pretrained in ['models/wtv_model_cwiki_50perc.pkl', None]:
                output_dir = os.path.join('results', 'exp%d' % i)
                mkdirs_if_not_exists(output_dir)

                conf = deepcopy(template)
                conf['output_dir'] = output_dir
                conf['corpus'] = corpus
                conf['iters'] = iters
                conf['pretrained_pkl'] = pretrained
                conf['specialised_pkl'] = 'models/specialised_wtv_%d.pkl' % i

                with open(os.path.join(output_dir, 'conf.txt'), 'w') as outf:
                    json.dump(conf, outf, indent=1)
                    i += 1
