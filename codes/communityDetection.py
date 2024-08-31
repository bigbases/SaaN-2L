from igraph import Graph

def communityDetector(ig, algo_str):
    ig = Graph.from_networkx(ig.to_networkx())
    start = time.time()
    if algo_str == 'LP':
        cd_algo = Graph.community_label_propagation(ig)
    
    elif algo_str == 'MM':
        cd_algo = Graph.community_multilevel(ig)

    elif algo_str == 'EG':
        cd_algo = Graph.community_leading_eigenvector(ig)
    
    elif algo_str == 'RW':
        cd_algo = Graph.community_walktrap(ig)
        cd_algo = cd_algo.as_clustering()

    elif algo == 'IM':
        cd_algo = Graph.community_infomap(ig)

    print("####### %s Community Detection Complete (time elapsed: %.2fs) #######" %(dataset_str, time.time() - start))
    print()
    print(cd_algo.summary())
    return cd_algo