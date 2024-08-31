#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Deep learning via word2vec's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_ [2]_.

The training algorithms were originally ported from the C package https://code.google.com/p/word2vec/
and extended with additional functionality.

For a blog tutorial on gensim word2vec, with an interactive web app trained on GoogleNews, visit http://radimrehurek.com/2014/02/word2vec-tutorial/

**Install Cython with `pip install cython` to use optimized word2vec training** (70x speedup [3]_).

Initialize a model with e.g.::

>>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Word2Vec.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the word2vec C format::

  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

You can perform various syntactic/semantic NLP word tasks with the model. Some of them
are already built-in::

  >>> model.most_similar(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.50882536), ...]

  >>> model.doesnt_match("breakfast cereal dinner lunch".split())
  'cereal'

  >>> model.similarity('woman', 'man')
  0.73723527

  >>> model['computer']  # raw numpy vector of a word
  array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

and so on.

If you're finished training a model (=no more updates, only querying), you can do

  >>> model.init_sims(replace=True)

to trim unneeded model memory = use (much) less RAM.

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [3] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""

import logging
import sys
import os
import heapq
import time
from copy import deepcopy
import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum,\
    prod

logger = logging.getLogger("gensim.models.word2vec")


from gensim_my import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange

from gensim_my.models import Word2Vec
from gensim_my.models.word2vec import Vocab

import pyximport
models_dir = os.path.dirname(__file__) or os.getcwd()
print("MODEL_DIR",models_dir)
pyximport.install(setup_args={"include_dirs": [models_dir, get_include()]})
# import pyximport
# pyximport.install()
# from word2vec_inner import train_sentence_topic, train_sentence_topic2, train_sentence_topic4

def predict_topic(model, context_embedding, word2):
    beta = 0.5
    prob = []
    exp_list = []
    for topic in range(model.cmty_num):
        exp_list.append(exp( dot(context_embedding, model.syn0_topic[topic]) ) )
    # sum_exp = sum(exp_list)
    for topic in range(model.cmty_num):
        p1 = exp_list[topic]
        p2 = (model.cmty_count[word2.index][topic]+beta)/(model.nwsum[topic]+len(model.vocab) * beta)
        prob.append(p1*p2)
    tot_p = sum(prob)
    norm_prob = [k/tot_p for k in prob]
    for k in range(1, len(norm_prob)):
        norm_prob[k] += norm_prob[k-1]
    predict_list = []
    tt = random.rand()
    for k in range(len(norm_prob)):
        if tt < norm_prob[k]:
            break
    model.nwsum[k] += 1
    model.cmty_count[word2.index][k] += 1
    return k

def train_sentence_topic3(model, sentence, alpha, work=None):
    """
    Update skip-gram model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.0

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        reduced_window = random.randint(model.window)  # `b` in the original word2vec code

        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        context_embedding = zeros(model.layer1_size)
        for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
            context_embedding += model.syn0[word2.index]

        for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
            # don't train on OOV words and on the `word` itself
            if word2 and not (pos2 == pos):
                topic = predict_topic(model, context_embedding, word2)
                # update(word, topic, model, alpha, work)
                l1 = model.syn0_topic[topic]
                neu1e = zeros(l1.shape)

                if model.hs:
                    # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
                    l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
                    fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
                    ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                    # model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
                    neu1e += dot(ga, l2a) # save error

                if model.negative:
                    # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                    word_indices = [word.index]
                    while len(word_indices) < model.negative + 1:
                        w = model.table[random.randint(model.table.shape[0])]
                        if w != word.index:
                            word_indices.append(w)
                    l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
                    fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
                    gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
                    model.syn1neg[word_indices] += outer(gb, l1) # learn hidden -> output
                    neu1e += dot(gb, l2b) # save error

                model.syn0_topic[topic] += neu1e  # learn input -> hidden

    return len([word for word in sentence if word is not None])



def assign(model, sentence, alpha, work=None):
    """
    Update skip-gram model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    # if model.negative:
    #     # precompute negative labels
    #     labels = zeros(model.negative + 1)
    #     labels[0] = 1.0

    for pos, pairword in enumerate(sentence):
        word = pairword[0]
        if pairword[0] is None:
            continue  # OOV word in the input sentence => skip
        reduced_window = random.randint(model.window)  # `b` in the original word2vec code

        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, pairword2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
            # don't train on OOV words and on the `word` itself
            word2 = pairword2[0]
            topic = pairword2[1]
            model.cmty_count[word2.index][topic] += 1
            # if word2 and not (pos2 == pos):
            #     l1 = model.syn0_topic[topic]
            #     neu1e = zeros(l1.shape)

            #     if model.hs:
            #         # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
            #         l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
            #         fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
            #         ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
            #         # model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
            #         neu1e += dot(ga, l2a) # save error

            #     if model.negative:
            #         # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
            #         word_indices = [word.index]
            #         while len(word_indices) < model.negative + 1:
            #             w = model.table[random.randint(model.table.shape[0])]
            #             if w != word.index:
            #                 word_indices.append(w)
            #         l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
            #         fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
            #         gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
            #         model.syn1neg[word_indices] += outer(gb, l1) # learn hidden -> output
            #         neu1e += dot(gb, l2b) # save error

            #     model.syn0_topic[topic] += neu1e  # learn input -> hidden

    return len([word for word in sentence if word is not None])

class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


class TopicModel(Word2Vec):
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2vec/

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original word2vec implementation via `save_word2vec_format()` and `load_word2vec_format()`.

    """
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
        sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0, cmty_num=100, simu=False):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `sg` defines the training algorithm. By default (`sg=1`), skip-gram is used. Otherwise, `cbow` is employed.
        `size` is the dimensionality of the feature vectors.
        `window` is the maximum distance between the current and predicted word within a sentence.
        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).
        `seed` = for the random number generator.
        `min_count` = ignore all words with total frequency lower than this.
        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
                default is 0 (off), useful value is 1e-5.
        `workers` = use this many worker threads to train the model (=faster training with multicore machines)
        `hs` = if 1 (default), hierarchical sampling will be used for model training (else set to 0)
        `negative` = if > 0, negative sampling will be used, the int for negative
                specifies how many "noise words" should be drawn (usually between 5-20)
        `cbow_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
                Only applies when cbow is used.
        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.sg = int(sg)
        self.table = None # for negative sampling --> this needs a lot of RAM! consider setting back to None before saving
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.window = int(window)
        self.seed = seed
        self.min_count = min_count
        self.sample = sample
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.cmty_num = cmty_num

        if sentences is not None:
            self.build_vocab(sentences)
            if simu == False:
                print("train deepwalk"  )
                self.train(sentences)




    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        random.seed(self.seed)
        self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
        if self.hs:
            self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)

        # self.cmty_count = zeros((len(self.vocab), self.cmty_num), dtype=REAL)
        # self.nwsum = zeros(self.cmty_num, dtype=REAL)

        self.syn0norm = None

    def reset_weights_topic(self, topic_number):
        self.nwsum = zeros(self.cmty_num, dtype=REAL)
        self.syn0_topic = empty((topic_number, self.layer1_size), dtype=REAL)
        for i in xrange(topic_number):
            self.syn0_topic[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
        self.cmty_count = zeros((len(self.vocab), self.cmty_num), dtype=REAL)
        for i in range(len(self.cmty_count)):
            sample_cmty = random.randint(self.cmty_num)
            self.cmty_count[i][sample_cmty] += 1.
            self.nwsum[sample_cmty] += 1.
        self.old_cmty_count = deepcopy(self.cmty_count)
        self.mulp = zeros(self.cmty_num, dtype=REAL)
        self.sentence_embedding = zeros(self.layer1_size, dtype=REAL)
        self.context_embedding = zeros(self.layer1_size, dtype=REAL)

        # self.exp_array = zeros(self.layer1_size, dtype=REAL)


    def train_topic(self, topic_number, sentences, total_words=None, word_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.reset_weights_topic(topic_number)
        print("train topic")
        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        total_words = total_words or int(sum(v.count * v.sample_probability for v in itervalues(self.vocab)))
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count
                if self.sg:
                    job_words = sum(train_sentence_topic3(self, sentence, alpha, work) for sentence in job)
                else:
                    job_words = sum(train_sentence_cbow(self, sentence, alpha, work, neu1) for sentence in job)
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        print("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %(100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for sentence in sentences:
                # avoid calling random_sample() where prob >= 1, to speed things up a little:
                sampled = [(self.vocab[word[0]],word[1]) for word in sentence
                    if word[0] in self.vocab and (self.vocab[word[0]].sample_probability >= 1.0 or self.vocab[word[0]].sample_probability >= random.random_sample())]
                yield sampled

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

        return word_count[0]

    def train_assign(self, topic_number, sentences, total_words=None, word_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        # self.reset_weights_topic(topic_number)
        print("train assign")
        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        total_words = total_words or int(sum(v.count * v.sample_probability for v in itervalues(self.vocab)))
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count
                if self.sg:
                    job_words = sum(assign(self, sentence, alpha, work) for sentence in job)
                else:
                    job_words = sum(train_sentence_cbow(self, sentence, alpha, work, neu1) for sentence in job)
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        print("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %(100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for sentence in sentences:
                # avoid calling random_sample() where prob >= 1, to speed things up a little:
                sampled = [(self.vocab[word[0]],word[1]) for word in sentence
                    if word[0] in self.vocab and (self.vocab[word[0]].sample_probability >= 1.0 or self.vocab[word[0]].sample_probability >= random.random_sample())]
                yield sampled

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

        return word_count[0]
    def train_topic2(self, topic_number, sentences, total_words=None, word_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        total_words = total_words or int(sum(v.count * v.sample_probability for v in itervalues(self.vocab)))
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count
                if self.sg:
                    job_words = sum(train_sentence_topic3(self, sentence, alpha, work) for sentence in job)
                else:
                    job_words = sum(train_sentence_cbow(self, sentence, alpha, work, neu1) for sentence in job)
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
                            (100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for sentence in sentences:
                # avoid calling random_sample() where prob >= 1, to speed things up a little:
                sampled = [self.vocab[word] for word in sentence
                    if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or self.vocab[word].sample_probability >= random.random_sample())]
                yield sampled

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))
        return word_count[0]


    def train_topic4(self, topic_number, sentences, total_words=None, word_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.reset_weights_topic(topic_number)
        for i in range(len(self.cmty_count)):
            sample_cmty = random.randint(self.cmty_num)
            self.cmty_count[i][sample_cmty] += 1.
            self.nwsum[sample_cmty] += 1
        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        total_words = total_words or int(sum(v.count * v.sample_probability for v in itervalues(self.vocab)))
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count
                if self.sg:
                    job_words = sum(train_sentence_topic3(self, sentence, alpha, work) for sentence in job)
                else:
                    job_words = sum(train_sentence_cbow(self, sentence, alpha, work, neu1) for sentence in job)
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
                            (100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for sentence in sentences:
                # avoid calling random_sample() where prob >= 1, to speed things up a little:
                sampled = [self.vocab[word] for word in sentence
                    if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or self.vocab[word].sample_probability >= random.random_sample())]
                yield sampled

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))
        return word_count[0]

    #get the topic's most similar words in the vocab
    def check_topic(self, topic_number, frequency):
        vector1 = matutils.unitvec(self.syn0_topic[topic_number])
        res = []
        for w in self.vocab:
            v = self.vocab[w].count
            if v <=frequency:
                continue
            vector2 = matutils.unitvec(self[w])
            tmp = dot(vector1, vector2)
            res.append((w,tmp))
        res = sorted(res, cmp = lambda x,y :-cmp(x[1],y[1]))[:40]
        print("=================================================================================")
        for (word, result) in res:
            try:
                print(word, result)
            except:
                pass
    def most_similar_topic(self, word):
        vector1 = matutils.unitvec(self[word])
        res = []
        for i in range(len(self.syn0_topic)):
            vector2 = matutils.unitvec(self.syn0_topic[i])
            tmp =dot(vector1, vector2)
            res.append((i, tmp))
        res = sorted(res, cmp=lambda x,y : -cmp(x[1], y[1]))
        print("================================================================================")
        for (word, reuslt) in res:
            print(word, reuslt)

    def save_topic(self, filename):
        topic_num = len(self.syn0_topic)
        with open(filename, "w") as f:
            for i in range(topic_num):
                for j in range(len(self.syn0_topic[0])):
                    f.write(str(self.syn0_topic[i][j])+" ")
                f.write("\n")
    
    def save_wordvector(self, filename):
        with open(filename,"w") as f:
            for w in self.vocab:
                v = self[w]
                now_line = str(w)
                for i in range(self.layer1_size):
                    now_line = now_line + " " + str(v[i])
                print(now_line, file=f)

    def generate_cmty(self):
            # if prob > 0.1 then set this node in this cmty

        self.node_cmty = {}
        self.cmtys = [[] for k in range(self.cmty_num)]
        self.ave_cmty = zeros((len(self.vocab), self.layer1_size), dtype=REAL)

        for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
            self.node_cmty[vocab.index] = []
            total_count = float(sum(self.cmty_count[vocab.index]))
            for cmty_index in range(len(self.cmty_count[vocab.index])):
                self.ave_cmty[vocab.index] += self.cmty_count[vocab.index][cmty_index]/total_count * self.syn0_topic[cmty_index]
                if self.cmty_count[vocab.index][cmty_index]/total_count > 0.05:
                    self.node_cmty[vocab.index].append(cmty_index)
                    self.cmtys[cmty_index].append(vocab.index)
        cmty_num_list = [len(self.node_cmty[key]) for key in self.node_cmty]
        no_verts_cmty_num = 0
        for i in range(len(self.cmtys)):
            if len(self.cmtys[i]) == 0:
                no_verts_cmty_num += 1
        print('len(self.cmtys)', len(self.cmtys))
        print('no_verts_cmty_num', no_verts_cmty_num)

        f = open('cmty before', 'w')
        for i in range(len(self.old_cmty_count)):
            for j in range(len(self.old_cmty_count[i])):
                f.write(str(self.old_cmty_count[i][j])+' ')
            f.write('\n')
        f.close()
        f = open('cmty after', 'w')
        for i in range(len(self.cmty_count)):
            for j in range(len(self.cmty_count[i])):
                f.write(str(self.cmty_count[i][j])+' ')
            f.write('\n')
        f.close()


    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        """

        if fvocab is not None:
            logger.info("Storing vocabulary in %s" % (fvocab))
            with utils.smart_open(fvocab, 'wb') as vout:
                for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                    vout.write(utils.to_utf8("%s %s\n" % (word, vocab.count)))
        logger.info("storing %sx%s projection weights into %s" % (len(self.vocab), self.layer1_size, fname))
        assert (len(self.vocab), self.layer1_size) == self.syn0.shape
        with utils.smart_open(fname+'cmtyembed.txt', 'wb') as fcmtyembed:
            fcmtyembed.write(utils.to_utf8("%s %s\n" % (self.cmty_num, self.layer1_size)))
            for cmty_index in range(len(self.syn0_topic)):
                row = self.syn0_topic[cmty_index]
                fcmtyembed.write(utils.to_utf8("%s %s\n" % (str(cmty_index), ' '.join("%f" % val for val in row))))

        with utils.smart_open(fname, 'wb') as fout, utils.smart_open(fname+'_multi', 'wb') as fmulti, utils.smart_open(fname+'_cmty', 'wb') as fcmty:
            fout.write(utils.to_utf8("%s %s\n" % (len(self.vocab), self.layer1_size*2)))
            fmulti.write(utils.to_utf8("%s %s\n" % (len(self.vocab), self.layer1_size*2)))

            # store in sorted order: most frequent words at the top
            for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):

                row0 = self.syn0[vocab.index]
                row1 = self.ave_cmty[vocab.index]
                row2 = self.node_cmty[vocab.index]
                fcmty.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%d" % val for val in row2))))
                if binary:
                    fout.write(utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s %s\n" % (word, ' '.join("%f" % val for val in row0), ' '.join("%f" % val for val in row1)    ) ) )
                word2_cmty = []
                # print word, str(vocab.index)
                if vocab.index in self.node_cmty:
                    word2_cmty = deepcopy(self.node_cmty[vocab.index])
                if len(word2_cmty) <= 0:
                    print("hahahahha")
                for cmty_index in word2_cmty:
                    row1 = self.syn0_topic[cmty_index]
                    # print row1
                    if binary:
                        fmulti.write(utils.to_utf8(word) + b" " + row.tostring())
                    else:
                        fmulti.write(utils.to_utf8("%s %s %s\n" % (word, ' '.join("%f" % val for val in row0), ' '.join("%f" % val for val in row1)    ) ) )                 

                
class BrownCorpus(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                line = utils.to_unicode(line)
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield words


class Text8Corpus(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest, max_sentence_length = [], b'', 1000
        with utils.smart_open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split()) # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # the last token may have been split in two... keep it for the next iteration
                words, rest = (utils.to_unicode(text[:last_token]).split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= max_sentence_length:
                    yield sentence[:max_sentence_length]
                    sentence = sentence[max_sentence_length:]


class LineSentence(object):
    """Simple format: one sentence = one line; words already preprocessed and separated by whitespace."""
    def __init__(self, source):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in self.source:
                yield utils.to_unicode(line).split()
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for line in fin:
                    #yield utils.to_unicode(line).split()
                    yield line.strip().split()

class CombinedSentence(object):
    def __init__(self, word_filename, topic_filename):
        self.topic_filename = topic_filename
        self.word_filename = word_filename
    def __iter__(self):
        with utils.smart_open(self.topic_filename) as topic, utils.smart_open(self.word_filename) as word:
            for line1, line2 in zip(word, topic):
                #line1 = line1.decode('utf8',errors='ignore').encode('utf8')
                #line2 = line2.decode('utf8',errors='ignore').encode('utf8')
                #words = utils.to_unicode(line1).split()
                #topics =utils.to_unicode(line2).split()
                words = line1.strip().split()
                topics = line2.strip().split()
                yield [(it1, int(it2)) for (it1, it2) in zip(words, topics)]
            


# Example: ./word2vec.py ~/workspace/word2vec/text8 ~/workspace/word2vec/questions-words.txt ./text8
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))
    logging.info("using optimization %s" % FAST_VERSION)

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]
    from gensim.models.word2vec import Word2Vec  # avoid referencing __main__ in pickle

    seterr(all='raise')  # don't ignore numpy errors

    # model = Word2Vec(LineSentence(infile), size=200, min_count=5, workers=4)
    model = Word2Vec(Text8Corpus(infile), size=200, min_count=5, workers=1)

    if len(sys.argv) > 3:
        outfile = sys.argv[3]
        model.save(outfile + '.model')
        model.save_word2vec_format(outfile + '.model.bin', binary=True)
        model.save_word2vec_format(outfile + '.model.txt', binary=False)

    if len(sys.argv) > 2:
        questions_file = sys.argv[2]
        model.accuracy(sys.argv[2])

    logging.info("finished running %s" % program)
