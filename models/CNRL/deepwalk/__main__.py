#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import subprocess
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import node2vec
import logging

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

# from deepwalk import graph
import graph
import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram
from six import text_type as unicode
from six import iteritems
from six.moves import range
from utils import loadData
import psutil
from multiprocessing import cpu_count
from numpy import random as rd

# p = psutil.Process(os.getpid())
# p.set_cpu_affinity(list(range(cpu_count())))

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()


def sentence_to_window(walk_file, window_size):
    walks = loadData.load_walks(walk_file)
    f = open(walk_file, 'w')

    for sentence in walks:
        for pos, word in enumerate(sentence):
            reduced_window = rd.randint(window_size)  # `b` in the original word2vec code

            # now go over all words from the (reduced) window, predicting each one in turn
            start = max(0, pos - window_size + reduced_window)
            for pos2, word2 in enumerate(sentence[start: pos + window_size + 1 - reduced_window], start):
                f.write(unicode(str(sentence[pos2]) + ' ', 'utf-8'))
            f.write(unicode('\n'))
    f.close()


def loadLabel(labelFile, outputfile):
    infile = open(labelFile)
    lines = infile.readlines()
    labelList = [0 for k in range(len(lines))]
    labels = []
    for line in lines:
        lineArr = line.strip().split()
        newLabel = int(lineArr[1])
        if newLabel not in labels:
            labels.append(newLabel)
            labelList[int(lineArr[0])] = len(labels) - 1
        else:
            labelIndex = labels.index(newLabel)
            labelList[int(lineArr[0])] = labelIndex
    print("labels")
    print(labels, len(labels))
    f = open(outputfile, 'wb')
    for k in labels:
        f.write(str(k) + ' ')
    # print labelList, len(labelList)
    f.close()
    return labelList, len(labels)


def process(args):
    if args.format == "adjlist":
        G = graph.load_adjacencylist(args.input, undirected=args.undirected)
        print('no adj list!')
        exit(-1)
    elif args.format == "edgelist":
        our_G = graph.load_edgelist(args.input, undirected=args.undirected)
        nx_G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in nx_G.edges():
            nx_G[edge[0]][edge[1]]['weight'] = 1
        nx_G = nx_G.to_undirected()
    elif args.format == "mat":
        G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

    labelList = []
    totalLabels = 0
    if args.labelFile != '':
        labelList, totalLabels = loadLabel(args.labelFile, args.output + str("_labelIndex.txt"))

    print("Number of nodes: {}".format(len(our_G.nodes())))

    num_walks = len(our_G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))
    # walks_filebase = args.output + ".walks"
    # walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
    #                                      path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
    #                                      num_workers=args.workers)

    # rw_method = 0,
    dataMat = []
    if args.vectorFile != '':
        dataMat, len_of_vector = loadData.loadVec(args.vectorFile, args.total_verts)

    cmty = []
    if args.cmtyFile != '':
        cmty = loadData.load_cmty(args.cmtyFile)

    if data_size < args.max_memory_data_size:
        print("Walking...")

        # arg ppr to choose ppr random walk or not
        if args.CNE == 1:
            cmtys = loadData.load_cmty(args.cmtyFile)
            cmty = []
        G = node2vec.Graph(nx_G, False, args.p, args.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(args.number_walks, args.walk_length)
        walks = [map(str, walk) for walk in walks]
        print("Training...")
        # walks_filebase = args.output + ".walks"
        # walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
        #                                      path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
        #                                      num_workers=args.workers, rw_method=args.rw_method, vector=dataMat, cmty=cmty)
        if args.CNE == 3:
            print('UseLDA')
            # use degree distribution for frequency in tree
            walks_filebase = args.output + ".node2vecwalks"
            walk_file = open(walks_filebase,"w")
            walk_write=list()
            for walk in walks:
                walk_write.append(u"%s" %(" ".join(walk)+"\n"))
            walk_file.writelines(walk_write)
            print(walks_filebase)
            tassign = walks_filebase + '_tassign'
            lda_command = 'java -Xmx16384m LdaGibbsSampler ' + walks_filebase + ' ' + tassign + ' ' + \
                          str(args.maxIndex) + ' ' + str(args.cmty_num)
            print(lda_command)
            subprocess.call(lda_command, shell=True)
            import topic_model
            sentence_word = topic_model.LineSentence(walks_filebase)
            model = topic_model.TopicModel(sentence_word, size=args.representation_size, workers=args.workers, sg=1, hs=1, window=args.window_size, cmty_num=args.cmty_num)
            sentence = topic_model.CombinedSentence(walks_filebase, tassign)
            print("Training the topic vector...")
            model.train_topic(args.cmty_num, sentence)
            model.generate_cmty()

        elif args.CNE == 4:
            print('halfLDA+Embedding')
            import topic_model
            model = topic_model.TopicModel(walks, size=args.representation_size, workers=args.workers, sg=1,hs=1, window=args.window_size, cmty_num=args.cmty_num)
            model.reset_weights_topic(args.cmty_num)
            model.train_topic2(topic_number=args.cmty_num, sentences=walks)
            model.generate_cmty()

        else:
            print('Word2Vec')
            model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, workers=args.workers, sg=1)


    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
        print("Walking...")

        walks_filebase = args.output + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                                          path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
                                                          num_workers=args.workers, rw_method=args.rw_method, vector=dataMat, cmty=cmty)

        print("Counting vertex frequency...")
        if not args.vertex_freq_degree:
            vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        model = Skipgram(sentences=serialized_walks.combine_files_iter(walk_files), vocabulary_counts=vertex_counts,
                         size=args.representation_size,
                         window=args.window_size, min_count=0, workers=args.workers)
        # print model.syn0[0]

    model.save_word2vec_format(args.output)


def main():
    parser = ArgumentParser("deepwalk",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument('--format', default='adjlist',
                        help='File format of input file')

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                        help='Size to start dumping walks to disk, instead of keeping them in memory.')

    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    parser.add_argument('--mssg', default=0, dest='mssg', type=int,
                        help='Use mssg to train')

    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. This option is faster than '
                             'calculating the vocabulary.')

    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')

    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')

    parser.add_argument('--trainMethod', default=0, type=int,
                        help='different training method for mssg, 0 for ordinary, 1 for new1, 2 for new2')

    parser.add_argument('--gauss', default=0, type=int,
                        help='0-Not use gaussian, 1 for gaussian, 2 for learning gaussian')

    parser.add_argument('--diagonal', default=0, type=int,
                        help='diagonal 0 or spherical 1')

    parser.add_argument('--covm', default=3, type=float, help='min covariance')

    parser.add_argument('--covM', default=5, type=float, help='max covariance')

    parser.add_argument('--labelFile', default='', help='label file for gaussian')

    parser.add_argument('--adjlistFile', default='', help='adjlist to generate negative sample')

    parser.add_argument('--vectorFile', default='', help='if use vector to random walk')

    parser.add_argument('--cmtyFile', default='', help='if use cmty random walk')

    parser.add_argument('--rw_method', default=0, type=int, help='0 for normal rw, 1 for cond rw, 2 for memory rw, 3 for cmty rw')

    parser.add_argument('--total_verts', default=0, type=int, help='if use memory rw, when load vector file need this arg')

    parser.add_argument('--CNE', default=0, type=int, help='if use cne model, CNE = 1, else = 0 ')

    parser.add_argument('--cmty-num', default=100, type=int, help='CNE=2, will train 100 cmty')

    parser.add_argument('--maxIndex', default=-1, type=int, help='max index')

    parser.add_argument('--window-sentence', default=False, type=bool)
    #node2vec
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
    logger.setLevel(numeric_level)

    if args.debug:
        sys.excepthook = debug

    process(args)


if __name__ == "__main__":
    sys.exit(main())
