#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import math
import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count

logger = logging.getLogger("deepwalk")


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return order()

  def cos_sim(self, beginVertexVec, endVertexVec):
    dotSum = 0
    beginSqr = 0
    endSqr = 0
    for i in range(len(beginVertexVec)):
        dotSum += beginVertexVec[i] * endVertexVec[i]
        beginSqr += beginVertexVec[i] * beginVertexVec[i]
        endSqr += endVertexVec[i] * endVertexVec[i]
    return dotSum/(math.sqrt(beginSqr)*math.sqrt(endSqr))

  def get_next_walk(self, cur, before_vector, vector):
    G = self
    # max_sim = self.cos_sim(before_vector, vector[G[cur][0]])
    # choose = G[cur][0]
    adj = G[cur]
    dis_list = [self.cos_sim(before_vector, vector[next]) for next in G[cur]]
    choose = 0
    # print 'dis_dict', dis_list
    ord_pro_list = [math.exp(i) for i in dis_list]
    sum_pro = sum(ord_pro_list)
    pro_list = [i/sum_pro for i in ord_pro_list]
    # print 'pro_list', pro_list
    # aggr_pro_list = [pro_list[0]]
    # for i in range(1, len(pro_list)):
    #   aggr_pro_list.append(pro_list[i] + aggr_pro_list[-1])
    # print 'aggr_pro_list', aggr_pro_list
    p = random.random()
    # print 'p', p
    choose = adj[-1]
    for i in range(len(adj)):
      if p < pro_list[i]:
        choose = adj[i]
        break
      else:
        p -= pro_list[i]   
    # print 'i', i 
    # for next in G[cur]:
    #   dis = self.cos_sim(before_vector, vector[next])
    #   # print dis
    #   agg_dis_dict.append()
    #   if dis > max_sim:
    #     choose = next
    return choose

  def vector_random_walk(self, path_length, alpha=0, rand=random.Random(), start=None, vector=[]):
    """ Returns a weighted random walk. 
        weight is the 1/conductance
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    # print 'vector in vector_random_walk'
    # print start
    G = self
    eps = 1e-5
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]
    vector_sum = deepcopy(vector[0])
    # vert_set save the vert in the path
    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          choose = self.get_next_walk(cur, vector_sum, vector)
          path.append(choose)
          for i in range(len(vector[choose])):
            vector_sum[i] += vector[choose][i]
        else:
          path.append(path[0])
      else:
        break
    return path

  #use 1/conductance as the weight to generate next walk
  def conductance_random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a weighted random walk. 
        weight is the 1/conductance
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    eps = 1e-5
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]

    # vert_set save the vert in the path
    vert_set = set()
    vert_set.add(path[0])
    cut = float(len(G[path[0]]))
    vol = float(len(G[path[0]]))
    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        a = rand.random()
        if a >= alpha:
          adj = []
          cond = []
          prob = []
          # save
          for next_walk in G[cur]:
            adj.append(next_walk)

            if next_walk in vert_set:
              cut_temp = cut
              vol_temp = vol
            else:
              cut_temp = cut + len(G[next_walk])
              vol_temp = vol + len(G[next_walk])

              for i in vert_set:
                if i in G[next_walk]:
                  cut_temp -= 2
            # cond.append(float(cut_temp)/float(vol_temp))
            cond.append(cut_temp/vol_temp)
            prob.append(math.exp(-(cut_temp/vol_temp)))
            # prob.append(float(vol_temp)/(float(cut_temp)+eps))
          # regulize 1/cond, and calculate accumulate prob
          sum_prob = sum(prob)
          # print sum_prob
          for i in range(len(prob)):
            prob[i] = prob[i]/sum_prob
          # print 'adjl', adj
          # for i in range(len(prob)):
          #   if i == 0:
          #     acc_prob.append(prob[i]/sum_prob)
          #   else:
          #     acc_prob.append(prob[i]/sum_prob + acc_prob[-1])
          # print acc_prob

          # weighted random walk
          p = rand.random()
          chosen = adj[-1]
          for i in range(len(adj)):
            if p < prob[i]:
              chosen = adj[i]
              break
            else:
              p -= prob[i]

          if chosen not in vert_set:
            vol += len(G[chosen])
            cut += len(G[chosen])
            for i in vert_set:
              if i in G[chosen]:
                cut -= 2
          
          vert_set.add(chosen)
          if chosen == path[-1]:
            print("why this?")
          path.append(chosen)
          # print path
          # path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
          vert_set.add(path[0])
      else:
        break
    if start == 7:
      print(path)
    return path
     
  # based on personized page rank algo.
  def ppr_random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]
    finalPath = []
    cut = float(len(G[path[0]]))
    vol = float(len(G[path[0]]))

    while len(path) < (path_length - len(finalPath)):
      cur = path[-1]
      if len(G[cur]) > 0:
        conductance = cut/vol
        if rand.random() < conductance:
          # continue random walk
          path.append(rand.choice(G[cur]))
          vol += float(len(G[path[-1]]))
          cut += float(len(G[path[-1]]))
          for i in range(len(path)-1):
            if path[i] in G[path[-1]]:
              cut -= 2
        else:
          finalPath += path
          # break
          # restart from a node 
          if len(path) > 1:
            newStart = path[-2]
          else:
            newStart = path[-1]
          path = [newStart]
          cut = float(len(G[path[0]]))
          vol = float(len(G[path[0]]))
        # if rand.random() >= alpha:
        #   path.append(rand.choice(G[cur]))
        # else:
        #   path.append(path[0])
      else:
        finalPath += path
        break
    # print finalPath
    return finalPath

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return path

  def cmty_random_walk(self, path_length, alpha=0, rand=random.Random(), start=None, cmty=[], node_cmty={}, beta=0):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
        cmty: community[i] -- verts list in community_i
        beta: the probability walk out of the community
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]

    # TODO: use different prob to random choose 
    if path[0] not in node_cmty:
      # print path[0]
      return self.random_walk(path_length=path_length, alpha=alpha, rand=rand, start=start)
    rand_cmty = rand.choice(node_cmty[path[0]])
    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          next_walks = []
          for i in G[cur]:
            if i in cmty[rand_cmty]:
              next_walks.append(i)
            # beta = 0 now
            elif rand.random() <= beta:
              next_walks.append(i)
          if next_walks == []:
            print("len path when next_walks = []", len(path))
            path.append(rand.choice(G[cur]))
          else:
            path.append(rand.choice(next_walks))
        else:
          path.append(path[0])
      else:
        break
    return path

# generate every vert's cmtys
def build_node_cmty(cmty):
  print("build node cmty")
  node_cmty = {}
  for i in range(len(cmty)):
    for vert in cmty[i]:
      if vert in node_cmty:
        node_cmty[vert].append(i)
      else:
        node_cmty[vert] = [i]
  # print node_cmty
  return node_cmty

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0), rw_method=0, vector=[], cmty = []):
  walks = []

  node_cmty = build_node_cmty(cmty)
  nodes = list(G.nodes())
  print('rw_method', rw_method)
  for cnt in range(num_paths):
    print(cnt)
    rand.shuffle(nodes)
    for node in nodes:
      if rw_method == 0:
        walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
      elif rw_method == 1:
        # print "ppr"
        walks.append(G.conductance_random_walk(path_length, rand=rand, alpha=alpha, start=node))
      elif rw_method == 2:
        walks.append(G.vector_random_walk(path_length, rand=rand, alpha=alpha, start=node, vector=vector))
      elif rw_method == 3:
        walks.append(G.cmty_random_walk(path_length, rand=rand, alpha=alpha, start=node, cmty=cmty, node_cmty=node_cmty))
      
  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0), rw_method=0, vector=[], cmty=[]):
  walks = []
  node_cmty = build_node_cmty(cmty)

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    print(cnt)
    rand.shuffle(nodes)
    for node in nodes:
      if rw_method == 0:
        yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)
      elif rw_method == 1:
        yield G.conductance_random_walk(path_length, rand=rand, alpha=alpha, start=node)
      elif rw_method == 2:
        yield G.vector_random_walk(path_length, rand=rand, alpha=alpha, start=node, vector=vector)
      elif rw_method == 3:
        yield G.cmty_random_walk(path_length, rand=rand, alpha=alpha, start=node, cmty=cmty, node_cmty=node_cmty)

  


def clique(size):
    return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])
  
  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()

  with open(file_) as f:
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
      total = 0 
      for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
          adjlist.extend(adj_chunk)
          total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 


def load_edgelist(file_, undirected=True):
  G = Graph()
  with open(file_) as f:
    for l in f:
      x, y = l.strip().split()[:2]
      x = int(x)
      y = int(y)
      G[x].append(y)
      if undirected:
        G[y].append(x)
  
  G.make_consistent()
  return G


def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G


