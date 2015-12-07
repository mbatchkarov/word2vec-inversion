import pickle

import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin, BaseEstimator


class TaddyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, taddy__specialised_pkl=None, **kwargs):
        self.taddy__specialised_pkl = taddy__specialised_pkl

    def fit(self, *args, **kwargs):
        if self.taddy__specialised_pkl is None:
            raise ValueError('I need this')

        with open(self.taddy__specialised_pkl, 'rb') as infile:
            self.models = pickle.load(infile)

    def predict(self, X):
        probs = docprob(X, self.models)
        return probs.values.argmax(axis=1) + 1  # gold is 1..5, predictions are 0..4


def docprob(docs, mods):
    """
    docprob takes two lists
    * docs: a list of documents, each of which is a list of sentences
    * models: the candidate word2vec models (each potential class)

    it returns the array of class probabilities.  Everything is done in-memory.
    """
    # score() takes a list [s] of sentences here; could also be a sentence generator
    sentlist = [s for d in docs for s in d]
    # the log likelihood of each sentence in this review under each w2v representation
    llhd = np.array([m.score(sentlist, len(sentlist)) for m in mods])
    # now exponentiate to get likelihoods,
    lhd = np.exp(llhd - llhd.max(axis=0))  # subtract row max to avoid numeric overload
    # normalize across models (stars) to get sentence-star probabilities
    prob = pd.DataFrame((lhd / lhd.sum(axis=0)).transpose())
    # and finally average the sentence probabilities to get the review probability
    prob["doc"] = [i for i, d in enumerate(docs) for s in d]
    prob = prob.groupby("doc").mean()
    return prob


if __name__ == '__main__':
    pass
