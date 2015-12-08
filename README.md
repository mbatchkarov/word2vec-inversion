# Document classification by inversion of word vectors

I recently read Matt Taddy's paper "Document classification by inversion of distributed language representations" (available [here](http://arxiv.org/abs/1504.07295)). The paper presents an neat method of making use of distributional information for document classification. However, my intuition is that the method would not work well for small labelled, so I decided to check this experimentally.

# Requirements
Implementation of method is based on example provided by [gensim](https://github.com/piskvorky/gensim). You will need a bleeding edge copy. Also, standard Python 3 scientific stack, e.g. `numpy` and `scikit-learn`. Legacy Python (<3.0) not supported.

# Current features

 - Taddy, Naive Bayes and SVM classifiers with grid search for parameter settings.
 - Yelp reviews and 20 Newsgroups data sets

# Disclaimer
This is a weekend hack, very much work in progress. Haven't had a chance to run an extensive evaluation. Preliminary results suggest Naive Bayes < Taddy < SVM (with grid search) for Yelp data.

# Todo

 - More classifiers
 - More labelled data sets
 - Grid search for `word2vec` parameters (currently using half of [cwiki](https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/) with default settings)