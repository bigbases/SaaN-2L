from utils import loadData
import sys
import random
import math
import datetime
import os
import subprocess
import threading
import time
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from numpy import sum as np_sum, dot, transpose, array
from utils.transform import adjList_to_edgeList, edgeList_to_adjList
from copy import deepcopy
from utils.train_test import gen_train_test, gen_train_test_from_walks
ranVecNum = 0
result = []

def cosDis(beginVertexVec, endVertexVec):
    dotSum = 0
    beginSqr = 0
    endSqr = 0
    for i in range(len(beginVertexVec)):
        dotSum += beginVertexVec[i] * endVertexVec[i]
        beginSqr += beginVertexVec[i] * beginVertexVec[i]
        endSqr += endVertexVec[i] * endVertexVec[i]
    return dotSum/(math.sqrt(beginSqr)*math.sqrt(endSqr))

def ranVec(len_of_vector):
    global ranVecNum
    vec = []
    for i in range(len_of_vector):
        vec.append(random.random()*2-1)
    ranVecNum += 1
    return vec

def join(dataMat1, dataMat2, len_of_vector_1, len_of_vector_2, num_of_vertex):
    dataMat = []
    for i in range(num_of_vertex):
        dataMat.append(dataMat1[i] + dataMat2[i])
        # print dataMat1[i] + dataMat2[i]
    print(len(dataMat[0]))
    return dataMat, len_of_vector_1 + len_of_vector_2

def calAvgE(ui, uj, covi, covj):
    logMulti = 0
    eps = 1e-6
    global result
    # for i in range(len(covi)):
    #     logMulti += math.log(covj[i]/covi[i])
    oneResult = [-np_sum(covj/(covi+eps))-np_sum(covi/(covj+eps)), -dot(((uj-ui)/(covj+eps)),(uj-ui).reshape(len(uj), 1))-dot(((uj-ui)/(covj+eps)), (uj-ui).reshape(len(uj), 1))]
    result.append(oneResult)
    return oneResult[0] + oneResult[1]


def similarity(begin, end, strategy, adjList, len_of_vector, dataMat, freqMat, covVec, matrixU):
    #common neighbors
    # if len(adjList[begin]) == 0 or len(adjList[end]) == 0:
    #   return 0
    eps = 1e-6
    if strategy == 'LINE12' or strategy == 'LINE1' or strategy == 'LINE2':
        # deep copy
        # beginVertexVec = dataMat[begin][:]
        # endVertexVec = dataMat[end][:]
        if len(dataMat[begin]) < 1:
            print(random)
            beginVertexVec = ranVec(len_of_vector)
        else:
            beginVertexVec = dataMat[begin]
        if len(dataMat[end]) < 1:
            endVertexVec = ranVec(len_of_vector)
        else:
            endVertexVec = dataMat[end]
        return (cosDis(beginVertexVec, endVertexVec) + 1)/2.0

    if strategy == 'DKL':
        ui = array(dataMat[begin])
        uj = array(dataMat[end])
        covi = array(covVec[begin])
        covj = array(covVec[end])
        negDkl = calAvgE(ui, uj, covi, covj)
        return negDkl

    if strategy == 'DeepWalk':
        # changed for One Dim Deepwalk
        if len(dataMat[begin]) < 1:
            beginVertexVec = ranVec(len_of_vector)
            print(begin)
        else:
            beginVertexVec = dataMat[begin]

        if len(dataMat[end]) < 1:
            endVertexVec = ranVec(len_of_vector)
        else:
            endVertexVec = dataMat[end]

        return (cosDis(beginVertexVec, endVertexVec) + 1)/2.0

    if strategy == 'MssgAvgSim':
        # TODO: when vertex isn't trained, the random vec will not be added to the mssgVec
        list_of_beginVertexVec = dataMat[begin][:]
        list_of_endVertexVec = dataMat[end][:]
        if len(list_of_beginVertexVec) < 1:
            beginVertexVec = ranVec(len_of_vector)
            list_of_beginVertexVec.append(beginVertexVec)
        if len(list_of_endVertexVec) < 1:
            endVertexVec = ranVec(len_of_vector)
            list_of_endVertexVec.append(endVertexVec)
        totalSim = 0.0
        count = 0.0
        for i, beginVec in enumerate(list_of_beginVertexVec):
            for j, endVec in enumerate(list_of_endVertexVec):
                totalSim += cosDis(beginVec, endVec) 
                count += 1
        return (totalSim/count + 1)/2.0

    if strategy == 'MssgAvgSimWeighted':
        list_of_beginVertexVec = dataMat[begin][:]
        list_of_endVertexVec = dataMat[end][:]
        totalSim = 0.0
        for i, beginVec in enumerate(list_of_beginVertexVec):
            for j, endVec in enumerate(list_of_endVertexVec):
                totalSim += cosDis(beginVec, endVec) * freqMat[begin][i] * freqMat[end][j]
        return totalSim

    if strategy == 'MssgLocalSim':
        list_of_beginVertexVec = dataMat[begin][:]
        list_of_endVertexVec = dataMat[end][:]
        if len(list_of_beginVertexVec) < 1:
            beginVertexVec = ranVec(len_of_vector)
            list_of_beginVertexVec.append(beginVertexVec)
        if len(list_of_endVertexVec) < 1:
            endVertexVec = ranVec(len_of_vector)
            list_of_endVertexVec.append(endVertexVec)
            # print len(list_of_endVertexVec) - len(dataMat[end])

        maxSim = cosDis(list_of_beginVertexVec[0], list_of_endVertexVec[0])
        sense1 = 0
        sense2 = 0
        for i, beginVec in enumerate(list_of_beginVertexVec):
            for j, endVec in enumerate(list_of_endVertexVec):
                cs = cosDis(beginVec, endVec)
                if cs > maxSim:
                    maxSim = cs
        return (maxSim + 1)/2.0

    # if strategy == 'MssgGlobalSim':
    #   beginVertexVec = globalVec[begin]
    #   endVertexVec = globalVec[end]
    #   if len(beginVertexVec) < 1:
    #       beginVertexVec = ranVec(len_of_vector)
    #   if len(endVertexVec) < 1:
    #       endVertexVec = ranVec(len_of_vector)
    #   return cosDis(beginVertexVec, endVertexVec)
    if strategy == 'DeepNet':
        # print matrixU
        vx = array(dataMat[begin])
        vy = array(dataMat[end])
        # xuy = dot(dot(vx, matrixU), vy)
        return dot(dot(vx, matrixU), vy)

    cn = 0
    for i in adjList[begin]:
        if i in adjList[end]:
            cn += 1
    if strategy == 'Common Neighbors':
        return cn
    if strategy == 'Salton Index':
        return cn/math.sqrt(eps + len(adjList[begin])*len(adjList[end]))
    if strategy == 'Jaccard Index':
        return float(cn)/float(eps + len(adjList[begin]) + len(adjList[end]) - cn)
    if strategy == 'Sorensen Index':
        return 2*float(cn)/float(eps + len(adjList[begin]) + len(adjList[end]))
    if strategy == 'Resource Allocation':
        k = 0
        for i in adjList[begin]:
            if i in adjList[end]:
                # if len(adjList[i]) != 0:
                k += 1.0/float(len(adjList[i]) + eps)
        return k


def genTrainedVertex(edgeList_train, num_of_vertex):
    trainedVertex = []
    for i in range(num_of_vertex):
        trainedVertex.append(0)
    for edge in edgeList_train:
        trainedVertex[edge[0]] = 1
        trainedVertex[edge[1]] = 1
    return trainedVertex

def ranGenNegEdge(adjList, trainedVertex):
    while True:
        begin2 = random.randint(0, len(adjList)-1)
        end2 = random.randint(0, len(adjList)-1)
        if (end2 not in adjList[begin2]) and trainedVertex[begin2] == 1 and trainedVertex[end2] == 1:
            break   
    return [begin2, end2]
# def foundEnd2(begin1, end, adjList, trainedVertex):
#   for node in adjList[end]:

def genNegEdgeNorm(edge, adjList, trainedVertex):
    while True:
        begin = random.randint(0, len(adjList)-1)
        end = random.randint(0, len(adjList)-1)
        if (end not in adjList[begin]) and trainedVertex[begin] == 1 and trainedVertex[end] == 1 and begin != end:
            break
    return begin, end, False

def genNegEdge(edge, adjList, trainedVertex):
    begin1 = edge[0]
    end1 = edge[1]
    count = 0
    adjList1 = adjList[begin1][:]
    random.shuffle(adjList1)
    # shuffle the adjList
    for next in adjList1:
        adjListNext = adjList[next][:]
        random.shuffle(adjListNext)
        for end2 in adjListNext:
            if end2 not in adjList[begin1] and (trainedVertex[end2] == 1) and (end2 != begin1):
                return begin1, end2, False

    while True:
        end2 = random.randint(0, len(adjList)-1)
        if end2 not in adjList[begin1] and (trainedVertex[end2] == 1) and (end2 != begin1):
            return begin1, end2, True

    # while True:
    #   found = False
    #   for node in adjList[next]:
    #       if node not in adjList[begin1] and (trainedVertex[node] == 1) and (node != begin1):
    #           end2 = node
    #           found = True
    #           break
    #   if found:
    #       if count > 0:
    #           print 'count', count
    #       break
    #   else:
    #       next = node
    #       count += 1
    # return [begin1, end2]

# extract edgeList which have at least one multiSenseVertex
def genMultiSenseEdgeTest(multiSenseVertex, trained_edgeList_test, edgeList_neg, sava_file_name):
    MultiSense_trained_EdgeList_test = []
    MultiSense_edgeList_neg = []
    f = open(sava_file_name, 'w')
    for i in range(len(trained_edgeList_test)):
        if (trained_edgeList_test[i][0] in multiSenseVertex) or (trained_edgeList_test[i][1] in multiSenseVertex):
            MultiSense_trained_EdgeList_test.append(trained_edgeList_test[i])
            MultiSense_edgeList_neg.append(edgeList_neg[i])
            f.write(str(trained_edgeList_test[i][0]) + ' ' + str(trained_edgeList_test[i][1]) + ' '\
                + str(edgeList_neg[i][0]) + ' ' + str(edgeList_neg[i][1]) + '\n')
    return MultiSense_trained_EdgeList_test, MultiSense_edgeList_neg


def genEdgeTest(adjList, edgeList_test, trainedVertex, AUCmethod):

    trained_edgeList_test = []
    edgeList_neg = []
    randNum = 0.0
    for edge in edgeList_test:
        if trainedVertex[edge[0]] == 1 and trainedVertex[edge[1]] == 1:
            if AUCmethod == 0:
                begin2, end2, randOrNot = genNegEdgeNorm(edge, adjList, trainedVertex)
            else:
                begin2, end2, randOrNot = genNegEdge(edge, adjList, trainedVertex)
            if randOrNot:
                randNum += 1.0
            else:
                trained_edgeList_test.append(edge)
                edgeList_neg.append([begin2, end2])

            if AUCmethod == 0:
                begin2, end2, randOrNot = genNegEdgeNorm([edge[1], edge[0]], adjList, trainedVertex)
            else:
                begin2, end2, randOrNot = genNegEdge([edge[1], edge[0]], adjList, trainedVertex)
            if randOrNot:
                randNum += 1.0
            else:
                trained_edgeList_test.append([edge[1], edge[0]])
                edgeList_neg.append([begin2, end2])

    print(randNum, len(trained_edgeList_test))
    return trained_edgeList_test, edgeList_neg

# def genTrainedEdgeTest(edgeList_test, mssgVec):
#   trained_edgeList_test = []
#   for edge in edgeList_test:
#       if len(mssgVec[edge[0]]) > 0 and len(mssgVec[edge[1]]) > 0:
#           trained_edgeList_test.append(edge)
#   return trained_edgeList_test

def choose_edge_pair(edgeList, adjList):
    missingIndex = random.randint(0, len(edgeList)-1)
    begin1 = edgeList[missingIndex][0]
    end1 = edgeList[missingIndex][1]

    # begin1 = begin2 and random choose end2
    # while True:
    #   end2 = random.randint(0, len(adjList)-1)
    #   if end2 not in adjList[begin1]:
    #       break

    # return begin1, end1, begin1, end2


    next = end1
    count = 0
    while True:
        # begin2 = random.randint(0, len(adjList)-1)
        # end2 = random.randint(0, len(adjList)-1)
        # if end2 not in adjList[begin2]:
        #   break
        found = False
        for node in adjList[next]:
            if node not in adjList[begin1]:
                end2 = node
                found = True
                break
        if found:
            if count > 0:
                print('count', count)
            break
        else:
            next = node
            count += 1

    return begin1, end1, begin1, end2


def AUCtest(strategy, adjList_train, len_of_vector, adjList, trained_edgeList_test, edgeList_neg, mssgVec, freqMat, covVec, matrixU):
    # circle = 10000
    nn = 0.0
    i = 0
    eps = 1e-6
    index_of_wrong_predict = []
    for i in range(len(trained_edgeList_test)):
        s1 = similarity(trained_edgeList_test[i][0], trained_edgeList_test[i][1], strategy, adjList_train, len_of_vector, mssgVec, freqMat, covVec, matrixU)
        s2 = similarity(edgeList_neg[i][0], edgeList_neg[i][1], strategy, adjList_train, len_of_vector, mssgVec, freqMat, covVec, matrixU)
        
        if float(s1) - float(s2) > eps:
            nn += 1
        elif math.fabs(float(s1) - float(s2)) <= eps:
            nn += 0.5
        if float(s1) - float(s2) <= eps:
          index_of_wrong_predict.append([i, trained_edgeList_test[i][0], trained_edgeList_test[i][1], edgeList_neg[i][0], edgeList_neg[i][1]])

    # f = open("log", 'w')
    # time_str = time.strftime('%Y-%m-%d %A %X %Z',time.localtime(time.time()))  
    print(strategy, "AUC:", nn, '/', len(trained_edgeList_test), '=', nn/float(len(trained_edgeList_test)))
    # f.write(time_str + '\n')
    # f.write(strategy + "AUC:" + str(nn) + '/' + str(len(trained_edgeList_test)) + '=' + str( nn/float( len(trained_edgeList_test) ) ) )
    # f.close()
    return nn/float(len(trained_edgeList_test)), index_of_wrong_predict
    # ignore = 0
    # while i < circle:
    #   # begin1, end1, begin2, end2 = choose_edge_pair(edgeList_test, adjList)


class Edge:

    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.sim = 0

    def setSim(self, sim):
        self.sim = sim

    def show(self):
        print('(' + str(self.begin) + ',' + str(self.end) + ')', self.sim)

def PRECtest(strategy, dataMat, adjList_train, len_of_vector, adjList, edgeList_test, mssgVec):
    all_edge = []
    for test_edge in edgeList_test:
        begin = test_edge[0]
        end = test_edge[1]
        if len(mssgVec[begin]) < 1 or len(mssgVec[end]) < 1:
            continue
        e = Edge(test_edge[0], test_edge[1])
        e.setSim(similarity(test_edge[0], test_edge[1], strategy, dataMat, adjList_train, len_of_vector, mssgVec))
        all_edge.append(e)
    L = len(all_edge)
    for i in range(L):
        while True:
            begin2 = random.randint(0, len(adjList)-1)
            end2 = random.randint(0, len(adjList)-1)
            if len(mssgVec[begin2]) < 1 or len(mssgVec[end2]) < 1:
                continue
            if end2 not in adjList[begin2]:
                break
        e = Edge(begin2, end2)
        e.setSim(similarity(test_edge[0], test_edge[1], strategy, dataMat, adjList_train, len_of_vector, mssgVec))
        all_edge.append(e)

    # for i in range(len(adjList)-1):
    #   for j in range(1, len(adjList)):
    #       if i < j and (j not in adjList_train[i]):
    #           if len(dataMat[i]) > 0 and len(dataMat[j]) > 0:
    #               # edge_dict[[i,j]] = similarity(i, j, strategy, dataMat, adjList_train, len_of_vector, mssgVec)
    #               e = Edge(i, j)
    #               e.setSim(similarity(i, j, strategy, dataMat, adjList_train, len_of_vector, mssgVec))
    #               all_edge.append(e)
    # TODO:use dict not class to save similarity of an edge
    def cmp(x, y):
        eps = 1e-6
        if math.fabs(x.sim - y.sim) < eps:
            return 0
        if x.sim - y.sim > eps:
            return 1
        return -1

    all_edge.sort(cmp, reverse = True)

    # if strategy == 'DeepWalk':
    #   for e in all_edge:
    #       e.show()

    count = 0.0
    for i in range(len(edgeList_test)):
        if all_edge[i].end in adjList[all_edge[i].begin]:
            count += 1.0

    print(strategy, "precision:", count, '/', L, '=', count / L)

    return count / L

def sphericalCov(covVec):
    for i in range(len(covVec)):
        total = 0.0
        if len(covVec[i]) > 0:
            for j in range(len(covVec[i])):
                total += covVec[i][j]
            avg = total / len(covVec[i])
            covVec[i] = [avg for k in range(len(covVec[i]))]

def genEdgeListTestAndTrain(adjList, edgeNum):
    adjList_test = [[] for k in range(len(adjList))]
    adjList_train = deepcopy(adjList)
    testNum = 0
    while testNum < edgeNum:
        i = random.randint(0, len(adjList)-1)
        if len(adjList_train[i]) > 1:
            j_index = random.randint(0, len(adjList_train[i])-1)
            j = adjList_train[i][j_index]
            if len(adjList_train[j]) > 1:
                adjList_test[i].append(j)
                adjList_test[j].append(i)
                # del adjList_train[i][j_index]
                adjList_train[i].remove(j)
                adjList_train[j].remove(i)

                testNum += 1
    print('testNum', testNum)
    # print adjList_test
    # print adjList_train
    edgeList_test = adjList_to_edgeList(adjList_test)
    edgeList_train = adjList_to_edgeList(adjList_train)
    return edgeList_test, edgeList_train


def main():
    random.seed(1)
    parser = ArgumentParser(description = 'link prediction demo for GNE')
    parser.add_argument('--input', required = True, help = 'Name of adjlist file.')
    parser.add_argument('--fold', default = 5, type = int, help = 'Number of folds in N-folds test.')
    parser.add_argument('--veclen', default = 64, type = int, help = 'Length of the vertex vector')
    parser.add_argument('--metric', default = 'AUC', help = 'Evaluation metrics: AUC or PRECISION')
    parser.add_argument('--reset', required = True, default = 1, type = int, help = 'reRun the Deepwalk or not')
    parser.add_argument('--mssg', default = 0, type = int, help = 'the sense Used in MssgWalk')
    parser.add_argument('--method', default = 0, type = int, help = 'deepwalk = 0, LINE = 1')
    parser.add_argument('--number-walks', default = 10, type = int, help = 'Number of random walks to start at each node')
    parser.add_argument('--walk-length', default = 40, type = int, help = 'Length of the random walk started at each node')
    parser.add_argument('--window-size', default = 5, type = int, help = 'Window size of skipgram model.')
    parser.add_argument('--workers', default = 1, type = int, help = 'Number of parallel processes.')
    parser.add_argument('--trainMethod', default = 0, type = int, help = '...')
    parser.add_argument('--fileTest', default = '', type = str, help = 'the test file name')
    parser.add_argument('--gauss', default = 0, type = int, help = 'Gauss or not in Deepwalk')
    parser.add_argument('--diagonal', default = 0, type = int, help = 'diagonal 0 or spherical 1')
    parser.add_argument('--covm', default = 0.5, type = float, help = 'minimum covariance')
    parser.add_argument('--covM', default = 1, type = float, help = 'maximum covariance')
    parser.add_argument('--AUCmethod', default = 0, type = int, help = '0 for normal AUC and 1 for difficult AUC')
    parser.add_argument('--labelFile', default = '', type = str, help = 'label file for gaussian')
    parser.add_argument('--walk-file', default='', type=str, help='the walks file')
    parser.add_argument('--full-graph', default=False, type=bool, help='use whole graph to train, to measure different test set')
    parser.add_argument('--rw_method', default=0, type=int, help='0 for normal rw, 1 for cond rw, 2 for memory rw')
    parser.add_argument('--cne', default=0, type=int, help='1 for cne model')
    parser.add_argument('--dn', default=0, type=int, help='1 for deepnet model')
    parser.add_argument('--resetdn', default=1, type=int, help='1 for retraining dn')
    parser.add_argument('--model', default=0, type=int, help='need add')
    parser.add_argument('--testtrain', default=0, type=int, help='use train set as test set')
    parser.add_argument('--cmty', default=-1, type=int, help='cmtys to train')
    parser.add_argument('--cene', default=0, type=int, help='use community enhanced model')
    parser.add_argument('--line', default=1, type=int, help='line algo order')
    parser.add_argument('--cenecmty', default=100, type=int, help='community enhanced cmty num')
    parser.add_argument('--cenemssg', default=0, type=int, help='if use multi sense for community enhanced representation')
    parser.add_argument('--rev', default=0, type=int, help='if 1 then will trans test and train(if we want train-ration = 0.2 when fold = 5)')
    #node2vec
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    # parser.add_argument('')
    args = parser.parse_args()

    name_of_strategy = ['DeepWalk', 'Common Neighbors', 'Salton Index', 'Jaccard Index', 'Resource Allocation']

    in_file_adjList = args.input

    fold = int(args.fold)
    len_of_vec = int(args.veclen)
    metric = args.metric
    print(metric)

    adjList = loadData.loadAdjList(in_file_adjList)
    for i in range(len(adjList)):
        if i in adjList[i]:
            print('have self loop')
            adjList[i].remove(i)

    print(adjList[:10])
    for i in range(len(adjList)):
        for j in adjList[i]:
            if i not in adjList[j]:
                adjList[j].append(i)
            #    print('fuck')
    print(adjList[:10])
    num_of_vertex = len(adjList)
    edgeList = adjList_to_edgeList(adjList)
    # edgeList = loadData.loadEdgeList(in_file_edgeList)
    # edgeList = loadData.loadEdgeList(in_file_adjList + '_toEdgeList')
    # print edgeList[0:10]
    # print 'init train and test edge'
    # starttime = datetime.datetime.now()
    # random.shuffle(edgeList)
    # endtime = datetime.datetime.now()
    # print 'shuffle use time', (endtime - starttime).seconds
    # print edgeList[0:10]
    # for i in range(len(edgeList)):
    #     if edgeList[i][0] == edgeList[i][1]:

    unit = int(len(edgeList)/fold)
    precision = []

    # TODO: load senseCount and calculate sense weight
    senseCount = []
    # TODO: use range(fold) to N-fold test

    # model = GaussEmbeddingNetwork(adjlist = adjList, veclen = args.veclen, batch_size = 300, diagonal = True)
    # model.save(in_file_adjList+"_toEmbedding")

    for cross in range(1):
        print('--'*20)
        print('test', str(cross))
        a = random.randint(0, 100)
        print('unit', unit)
        if args.walk_file == '':
            if args.rev == 1:
                edgeList_train, edgeList_test = gen_train_test(adjList, unit)
            else:
                edgeList_test, edgeList_train = gen_train_test(adjList, unit)
        else:
            walks = loadData.load_walks(args.walk_file)
            edgeList_test, edgeList_train = \
            gen_train_test_from_walks(adjList, unit, walks)

        if args.full_graph == True:
            print("use full graph to train")
            edgeList_train = edgeList
        print(edgeList_test[0])
        print(edgeList_train[0])
        print('edgeList_test', len(edgeList_test))
        print('edgeList_train', len(edgeList_train))
        # edgeList_test  = edgeList[cross*unit:(cross+1)*unit]
        # edgeList_train = edgeList[:cross*unit] + edgeList[(cross+1)*unit:]

        trainedVertex = genTrainedVertex(edgeList_train, num_of_vertex)
        edgeList_train_file_name = in_file_adjList + '_toEdgeList_train' + str(cross)
        edgeList_test_file_name = in_file_adjList + '_toEdgeList_test' + str(cross)

        # # Training method for Deepwalk and more
        f = open(edgeList_train_file_name, 'w')
        for k in range(len(edgeList_train)):
            #the input format just for DeepWalk
            # LINE input
            if args.method == 1:
                f.write(str(edgeList_train[k][0]) + ' ' + str(edgeList_train[k][1]) + ' 1 \n')
                f.write(str(edgeList_train[k][1]) + ' ' + str(edgeList_train[k][0]) + ' 1 \n')
            # dw input
            else:
                f.write(str(edgeList_train[k][0]) + ' ' + str(edgeList_train[k][1]) + '\n')
        f.close()
        ftest = open(edgeList_test_file_name, 'w')
        for k in range(len(edgeList_test)):
            ftest.write(str(edgeList_test[k][0]) + ' ' + str(edgeList_test[k][1]) + '\n')
        ftest.close()
        in_file_vertexVec_new = in_file_adjList + '_toEmbedding' + str(cross)
        adjList_train = edgeList_to_adjList(edgeList_train, num_of_vertex)
        if args.method == 0:
            # gen cmty file
            if args.rw_method == 3 or args.cne > 0:#??
                f_un = open(edgeList_train_file_name+'_un', 'w')
                for k in range(len(edgeList_train)):
                  f_un.write(str(edgeList_train[k][0]+1) + '\t' + str(edgeList_train[k][1]+1) + '\n')
                  f_un.write(str(edgeList_train[k][1]+1) + '\t' + str(edgeList_train[k][0]+1) + '\n')
                f_un.close()
                bigclam_command = './bigclam -i:' + edgeList_train_file_name + '_un ' + '-o:' + edgeList_train_file_name + ' -c:' + args.cmty
                print(bigclam_command)
                subprocess.call(bigclam_command, shell=True)
                
            # use deepwalk or gauss_dw
            if args.gauss == 0 or args.gauss == 2:
                if args.mssg > 0 or args.cenemssg > 0:
                    name_of_strategy = ['MssgAvgSim', 'MssgLocalSim', 'Common Neighbors', 'Salton Index', 'Jaccard Index', 'Resource Allocation']
                if args.cne > 0:                
                    name_of_strategy = ['MssgAvgSim', 'MssgLocalSim', 'Common Neighbors', 'Salton Index', 'Jaccard Index', 'Resource Allocation']
                    dw_command = 'python3 deepwalk/__main__.py --format edgelist --input ' + edgeList_train_file_name \
                    + ' --representation-size ' + str(len_of_vec/2) + ' --output ' + in_file_vertexVec_new\
                    + ' --mssg ' + str(args.mssg) + ' --window-size ' + str(args.window_size) \
                    + ' --walk-length ' + str(args.walk_length) + ' --number-walks ' + str(args.number_walks) \
                    + ' --workers ' + str(args.workers) + ' --trainMethod ' + str(args.trainMethod) \
                    + ' --diagonal ' + str(args.diagonal) \
                    + ' --covm ' + str(args.covm) + ' --covM ' + str(args.covM) + ' --gauss ' + str(args.gauss) + ' --CNE 1'\
                    +  ' --cmtyFile ' + edgeList_train_file_name + 'cmtyvv.txt'
                    f.close()
                    print(dw_command, 'cne > 0 and gauss == 0 or gauss 2')
                    if args.reset == 1:
                        if args.rw_method == 3:
                            dw_command += ' --rw_method 3 ' + ' --cmtyFile ' + edgeList_train_file_name + 'cmtyvv.txt'
                        starttime = datetime.datetime.now()
                        subprocess.call(dw_command, shell = True)
                        endtime = datetime.datetime.now()
                        print('deepwalk running time', (endtime - starttime).seconds)
                else:

                    dw_command = 'python3 deepwalk/__main__.py --format edgelist --input ' + edgeList_train_file_name \
                    + ' --representation-size ' + str(len_of_vec) + ' --output ' + in_file_vertexVec_new\
                     + ' --mssg ' + str(args.mssg) + ' --window-size ' + str(args.window_size) \
                     + ' --walk-length ' + str(args.walk_length) + ' --number-walks ' + str(args.number_walks) \
                     + ' --workers ' + str(args.workers) + ' --trainMethod ' + str(args.trainMethod) \
                     + ' --diagonal ' + str(args.diagonal) \
                     + ' --covm ' + str(args.covm) + ' --covM ' + str(args.covM) + ' --gauss ' + str(args.gauss) + \
                    ' --p ' + str(args.p) + ' --q ' + str(args.q)
                    if args.labelFile != '':
                      dw_command += ' --labelFile ' + args.labelFile
                    f.close()
                    print(dw_command, 'normal deepwalk')

                    if args.reset == 1:
                        if args.rw_method == 3:
                            dw_command += ' --rw_method 3 ' + ' --cmtyFile ' + edgeList_train_file_name + 'cmtyvv.txt'
                        if args.cene > 0:
                            if args.cene == 1:
                                dw_command = 'python3 deepwalk/__main__.py --format edgelist --input ' + edgeList_train_file_name + \
                                ' --window-size ' + str(args.window_size) + ' --walk-length ' + str(args.walk_length) + ' --number-walks ' + str(args.number_walks) + \
                                ' --output ' + in_file_vertexVec_new + ' --gauss ' + str(args.gauss) + ' --mssg ' + str(args.mssg) + \
                                ' --trainMethod ' + str(args.trainMethod) + ' --representation-size ' + str(len_of_vec) + \
                                ' --covm ' + str(args.covm) + ' --covM ' + str(args.covM) + ' --CNE 3' + ' --cmty-num ' + str(args.cenecmty) + ' --maxIndex ' + str(num_of_vertex) + ' --workers ' + str(args.workers)+ \
                                ' --p ' + str(args.p) + ' --q ' + str(args.q)
                            if args.cene == 2:
                                dw_command = 'python3 deepwalk/__main__.py --format edgelist --input ' + edgeList_train_file_name + \
                                ' --window-size ' + str(args.window_size) + ' --walk-length ' + str(args.walk_length) + ' --number-walks ' + str(args.number_walks) + \
                                ' --output ' + in_file_vertexVec_new + ' --gauss ' + str(args.gauss) + ' --mssg ' + str(args.mssg) + \
                                ' --trainMethod ' + str(args.trainMethod) + ' --representation-size ' + str(len_of_vec) + \
                                ' --covm ' + str(args.covm) + ' --covM ' + str(args.covM) + ' --CNE 4' + ' --cmty-num ' + str(args.cenecmty) + ' --maxIndex ' + str(num_of_vertex) + ' --workers ' + str(args.workers)+ \
                                ' --p ' + str(args.p) + ' --q ' + str(args.q)
                        print(dw_command)
                        starttime = datetime.datetime.now()
                        subprocess.call(dw_command, shell = True)
                        endtime = datetime.datetime.now()
                        print('deepwalk running time', (endtime - starttime).seconds)
                        # community enhanced model!
                # memory random walk
                if args.rw_method == 2:
                    dw_command += ' --rw_method 2 ' + ' --vectorFile ' + in_file_vertexVec_new + ' --total_verts ' + str(len(adjList))
                    print(dw_command)
                    if args.reset == 1:
                        starttime = datetime.datetime.now()
                        subprocess.call(dw_command, shell = True)
                        endtime = datetime.datetime.now()
                        print('deepwalk running time', (endtime - starttime).seconds)
                
                freqMat = []
                mssgVec = []
                covVec = []
                matrixU = []
                if args.mssg > 0 or args.cne > 0 or args.cenemssg > 0:
                    if args.cenemssg > 0:
                        mssgVec, len_of_vector = loadData.loadMultiSenseVec(in_file_vertexVec_new+'_multi', num_of_vertex)
                    else:
                        mssgVec, len_of_vector = loadData.loadMultiSenseVec(in_file_vertexVec_new, num_of_vertex)
                else:
                    # will be used by DeepNet
                    mssgVec, len_of_vector = loadData.loadVec(in_file_vertexVec_new, num_of_vertex)
                if args.gauss == 2:
                    name_of_strategy = ['DKL', 'DeepWalk', 'Common Neighbors', 'Salton Index', 'Jaccard Index', 'Resource Allocation']
                    covVec, len_of_vector = loadData.loadVec(in_file_vertexVec_new + '_cov', num_of_vertex)
                
                # deepnet have done deepwalk
                if args.dn == 1:
                    name_of_strategy = ['DeepNet', 'DeepWalk', 'Common Neighbors', 'Salton Index', 'Jaccard Index', 'Resource Allocation']
                    if args.resetdn == 1:
                        model = DeepNet(adjlist=adjList_train, embedding=mssgVec, veclen=len_of_vector)
                        model.save(in_file_vertexVec_new+'_u')
                    matrixU = loadData.load_matrixU(in_file_vertexVec_new+'_u', len_of_vector)

        # for LINE
        else:
            output = args.input + '_embed.txt'
            freqMat = []
            mssgVec = []
            covVec = []
            matrixU = []
            name_of_strategy = ['DeepWalk', 'Common Neighbors', 'Salton Index', 'Jaccard Index', 'Resource Allocation']

            # edgeList_train_file_name = args.input + '_un'
            # f = open(edgeList_train_file_name, 'w')
            # for edge in edgelist:
            #     f.write(str(edge[0])+' '+str(edge[1])+' 1'+'\n')
            #     f.write(str(edge[1])+' '+str(edge[0])+' 1'+'\n')
            # f.close()
            starttime = datetime.datetime.now()
            line_command1 = './line -train ' + edgeList_train_file_name + \
            ' -output ' + in_file_vertexVec_new + ' -size '+ str(args.veclen) + ' -order ' + str(args.line) + ' -negative 5 -samples 10 -rho 0.025 -threads 20'
            endtime = datetime.datetime.now()
            if args.reset == 1:
                subprocess.call(line_command1, shell=True)
            print('line running time', (endtime-starttime).seconds)
            mssgVec, len_of_vector = loadData.loadVec(in_file_vertexVec_new, num_of_vertex)
            # print mssgVec

        precision_once = []
        # generate Test Set
        trained_edgeList_test, edgeList_neg = genEdgeTest(adjList, edgeList_test, trainedVertex, args.AUCmethod)
        if args.testtrain == 1:
            trained_edgeList_test, edgeList_neg = genEdgeTest(adjList, edgeList_train, trainedVertex, args.AUCmethod)

        for strategy in name_of_strategy:
            # # deepwalk
            # if args.method == 0:
            if metric == 'AUC':
                AUCtest_result, index_of_wrong_predict = AUCtest(strategy, adjList_train, len_of_vector, adjList, trained_edgeList_test, edgeList_neg, mssgVec, freqMat, covVec, matrixU)
                f_wrong = open(in_file_adjList + '_' + strategy + '_wrong_' + str(cross), 'w')
                for i in index_of_wrong_predict:
                  for j in i:
                    f_wrong.write(str(j) + ' ')
                  f_wrong.write('\n')
                f_wrong.close()
                precision_once.append(AUCtest_result)
            else:
                precision_once.append(PRECtest(strategy, dataMat, adjList_train, len_of_vector, adjList, edgeList_test, mssgVec))
            # # LINE
            # else:
            #     dataMat = []
            #     len_of_vector = 0
            #     if strategy == 'LINE12':
            #         dataMat = dataMat0
            #         len_of_vector = len_of_vector0
            #     if strategy == 'LINE1':
            #         dataMat = dataMat1
            #         len_of_vector = len_of_vector1
            #     if strategy == 'LINE2':
            #         dataMat = dataMat2
            #         len_of_vector = len_of_vector2
            #     precision_once.append(AUCtest(strategy, adjList_train, len_of_vector, adjList, trained_edgeList_test, edgeList_neg, dataMat, [], []))
        precision.append(precision_once)
        
    total = [0]*len(precision[0])
    print(precision)
    for k in precision:
        for l in range(len(k)):
            total[l] += k[l]
    print(total)
    for k in range(len(total)):
        total[k] /= float(fold)
    for i, strategy1 in enumerate(name_of_strategy):
        print(strategy1, total[i] )

    global result
    print(result[0:10])
if __name__ == '__main__':
    main()
