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