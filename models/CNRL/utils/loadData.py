def loadVec(in_file_vertexVec, num_of_vertex):
    # f = open(in_file_vertexVec)
    # dataMat = []
    # for line in f.readlines():
    #   lineArr = line.strip().split()
    #   dataLine = []
    #   for i in range(len(lineArr)):
    #       dataLine.append(float(lineArr[i]))
    #   dataMat.append(dataLine)
    # f.close()
    infile = open(in_file_vertexVec)
    line0 = infile.readline()
    line0arr = line0.strip().split()
    len_of_vector = int(line0arr[1])
    # dataMat = [[]]*num_of_vertex
    dataMat = []
    for i in range(num_of_vertex):
        dataMat.append([])
    for line in infile.readlines():
        lineArr = line.strip().split()
        index = int(float(lineArr[0]))
        data = lineArr[1:]
        dataMat[index] = [float(k) for k in data]
    return dataMat, len_of_vector

def load_matrixU(in_file, veclen):
    matrixU = []
    infile = open(in_file)
    for line in infile.readlines():
        lineArr = line.strip().split()
        arr = [float(k) for k in lineArr]
        matrixU.append(arr)

    return matrixU

def loadMultiSenseVec(in_file_vertexVec, num_of_vertex):
    infile = open(in_file_vertexVec)
    line0 = infile.readline()
    line0arr = line0.strip().split()
    len_of_vector = int(line0arr[1])
    # dataMat = [[]]*num_of_vertex
    dataMat = []
    for i in range(num_of_vertex):
        dataMat.append([])
    for line in infile.readlines():
        lineArr = line.strip().split()
        index = int(float(lineArr[0]))
        data = lineArr[1:]
        dataMat[index].append([float(k) for k in data])
    return dataMat, len_of_vector
    
def loadAdjList(in_file_adjList):
    f = open(in_file_adjList)
    adjList = []
    for line in f.readlines():
        lineArr = line.strip().split()
        dataLine = []
        if len(adjList) != int(lineArr[0]):
            print("lack vertex", str(len(adjList)))
        for i in range(1, len(lineArr)):
            # avoid self loop
            if (len(adjList)!=int(lineArr[i])):
                dataLine.append(int(lineArr[i]))
        adjList.append(dataLine)
    f.close()
    return adjList

def loadEdgeList(in_file_edgeList):
    f = open(in_file_edgeList)
    edgeList = []
    for line in f.readlines():
        lineArr = line.strip().split()
        dataLine = []
        if int(lineArr[0])!= int(lineArr[1]):
            edgeList.append([int(lineArr[0]), int(lineArr[1])])
        for i in range(2):
            dataLine.append(int(lineArr[i]))
        # edgeList.append(dataLine)
    f.close()
    return edgeList


def loadSenseCount(file_name, num_of_vertex):
    threshold = 0.1
    infile = open(file_name)
    # dataMat = [[]]*num_of_vertex
    dataMat = []
    for i in range(num_of_vertex):
        dataMat.append([])
    for line in infile.readlines():
        lineArr = line.strip().split()
        index = int(lineArr[0])
        data = lineArr[1:]
        dataMat[index] = [int(k) for k in data]
    freqMat = []
    multiSenseVertex = []
    for i in range(num_of_vertex):
        freqMat.append([])
    for i in range(num_of_vertex):
        if len(dataMat[i]) > 0:
            total = 0.0
            for j in range(len(dataMat[i])):
                total += dataMat[i][j]
            for j in range(len(dataMat[i])):
                freqMat[i].append(float(dataMat[i][j])/float(total))
            senses = 0
            for j in range(len(freqMat[i])):
                if freqMat[i][j] > threshold:
                    senses += 1
            if senses >= 2:
                multiSenseVertex.append(i)

    return freqMat, multiSenseVertex

def loadEdgeTest(file_name):
    infile = open(file_name)
    trained_edgeList_test = []
    edgeList_neg = []
    for line in infile.readlines():
        lineArr = line.strip().split()
        trained_edge = [int(lineArr[0]), int(lineArr[1])]
        edge_neg = [int(lineArr[2]), int(lineArr[3])]
        trained_edgeList_test.append(trained_edge)
        edgeList_neg.append(edge_neg)
    return trained_edgeList_test, edgeList_neg

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
    # for i in xrange(len(labelList)):
    #     print i, labelList[i]
    # print labelList, len(labelList)

    f.close()
    return labelList, len(labels)

def load_walks(in_file_walk):
    f = open(in_file_walk)
    walks = []
    for line in f.readlines():
        lineArr = line.strip().split()
        walk = []
        for node in lineArr:
            walk.append(int(node))
        walks.append(walk)
    f.close()
    return walks

def load_cmty(in_file_cmty):
    f = open(in_file_cmty)
    cmtys = []
    for line in f.readlines():
        lineArr = line.strip().split()
        cmty = []
        for node in lineArr:
            # adjust for bigclam algo
            cmty.append(int(node)-1)
        cmtys.append(cmty)
    f.close()
    return cmtys

def load_label(in_file_label):
    label = {}
    f = open(in_file_label)
    for line in f.readlines():
        lineArr = line.strip().split()
        label[(int)(lineArr[0])] = (int)(lineArr[1])
    return label

def load_multilabel(in_file_label):
    label = {}
    f = open(in_file_label)
    for line in f.readlines():
        lineArr = line.strip().split()
        if int(lineArr[0]) not in label:
            label[int(lineArr[0])]=list()
        label[int(lineArr[0])].append(int(lineArr[1]))
    return label

def load_nw(in_file):
    f = open(in_file)
    nw = []
    for line in f.readlines():
        lineArr = line.strip().split()
        nw.append([int(t) for t in lineArr])
    return nw


