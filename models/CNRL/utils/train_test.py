import random
#from transform import adjList_to_edgeList, edgeList_to_adjList
from copy import deepcopy

def edgeList_to_adjList(edgeList, num_of_vertex):
    adjList = []
    for i in range(num_of_vertex):
        adjList.append([])
    for edge in edgeList:
        # if edge[0] >= num_of_vertex:
        #   print edge[0]
        # if edge[1] >= num_of_vertex:
        #   print edge[1]
        adjList[edge[0]].append(edge[1])
        adjList[edge[1]].append(edge[0])
    # print 'edgeList to adjList'
    # for i in range(len(adjList)):
    #   print len(adjList[i])
    return adjList
    
def adjList_to_edgeList(adjList):
    edgeList = []
    for i in range(len(adjList)):
        for j in range(len(adjList[i])):
            if i < adjList[i][j]:
                edgeList.append([i, adjList[i][j]])
    return edgeList

def gen_train_test(adjList, edgeNum):
    '''generate Train edges and
    '''
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

def gen_train_test_from_walks(adjList, edgeNum, walks):
    '''generate Train/Test edges from walk 
    '''
    adjList_test = [[] for k in range(len(adjList))]
    adjList_train = deepcopy(adjList)
    testNum = 0
    while testNum < edgeNum:
        walk = random.choice(walks)
        i_index = random.randint(0, len(walk)-2)
        i = walk[i_index]
        j = walk[i_index+1]
        if len(adjList_train[i]) > 1 and len(adjList_train[j]) > 1 and j in adjList_train[i]:
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

def edgeList_to_adjList(edgeList, num_of_vertex):
    adjList = []
    for i in range(num_of_vertex):
        adjList.append([])
    for edge in edgeList:
        # if edge[0] >= num_of_vertex:
        #   print edge[0]
        # if edge[1] >= num_of_vertex:
        #   print edge[1]
        adjList[edge[0]].append(edge[1])
        adjList[edge[1]].append(edge[0])
    # print 'edgeList to adjList'
    # for i in range(len(adjList)):
    #   print len(adjList[i])
    return adjList
    
def adjList_to_edgeList(adjList):
    edgeList = []
    for i in range(len(adjList)):
        for j in range(len(adjList[i])):
            if i < adjList[i][j]:
                edgeList.append([i, adjList[i][j]])
    return edgeList

    