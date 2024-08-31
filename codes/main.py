import random
random.seed(0)

import numpy as np
np.random.seed(0)

import argument

def main():
    args, unknown = argument.parse_args()
    
    # Graph Construction 
    G, node_subjects = graphloader(args.dataset)

    # Convert the data structure from StellarGraph -> NetworkX -> Igraph
    ig = Graph.from_networkx(G.to_networkx())

    # Community Detection
    cd_algo = communityDetector(ig, args.cd_algo)

    fea_mat = pd.DataFrame(G.node_features(), index= G.nodes())

    # Preserve the connections for original node information of super nodes
    original_edges = ig.get_edgelist()
    super_node_id = defaultdict(list)
    super_node_edges = defaultdict(set) # { super_node_id: [edge_list] }

    for commu in range(len(cd_algo)):
        if len(cd_algo[commu]) >= size_thresh: # If major community
            for node in cd_algo[commu]: # for each node 
                super_node_id[commu].append(node) # the index of nodes which should be converted to super node
                for edges in original_edges: # check edges
                    if edges[0] == node: # If it includes current node
                        super_node_edges[commu].add(edges[1]) # add the neighborhood node index number
                    elif edges[1] == node: 
                        super_node_edges[commu].add(edges[0])
    
    # create a mapping dictionary (node id: nx_name) from igraph.
    node_dict = {}
    for idx in range(ig.vcount()):
        node_dict[idx] = ig.vs['_nx_name'][idx]

    # CaaN Graph Construction
    CaaN = ig.copy()
    membership = cd_algo.membership
    counter = Counter(membership).most_common()

    minor_commuID = []
    # If the community size is less than threshold, add to minor community ID list
    for c in counter:
        if c[1] < size_thresh:
            minor_commuID.append(c[0])

    # Re-assign membership for minor community nodes
    new_id = len(cd_algo)
    for i in range(len(membership)):
        if membership[i] in minor_commuID:
            membership[i] = new_id
            new_id += 1
            
    idx_map = {}
    n = 0
    for i in sorted(dict(Counter(membership))):
        idx_map[i] = n
        n+=1
        
    # Initialization by index
    new_idx = 0
    for i in range(len(membership)):
        membership[i] = idx_map[membership[i]]

    CaaN.contract_vertices(membership, combine_attrs="first")
    CaaN.simplify(combine_edges="ignore") 
    print("Caan Graph Information : ")
    print(CaaN.summary())
    
    # Remove minor nodes from CaaN except for super nodes
    superG = CaaN.copy()
    cnt = 0
    deleted = []
    for v in range(superG.vcount()):
        if superG.vs['_nx_name'][v] in minor_nodes_features:
            deleted.append(v)
            cnt += 1
    print(cnt)
    superG.delete_vertices(deleted)

    # Global GRL 
    CaaN_fea_mat = fea_mat[fea_mat.index.isin(CaaN.vs['_nx_name'])] # extract features of subgraphs
    CaaN_sg = sg.StellarGraph.from_networkx(CaaN.to_networkx(), node_features = CaaN_fea_mat.reset_index(drop=True))
    CaaN_subjects = node_subjects[node_subjects.index.isin(CaaN.vs['_nx_name'])].reset_index(drop=True) # Label of subgraphs

    # Base model and Global GRL Training 
    if args.model == 'gcn':
        from models import GCN
        if args.task == 'node': 
            X_base = GCN.node_classification(G, node_subjects, args) # Base model 
            X_CaaN = GCN.node_classification(CaaN_sg, CaaN_subjects, args) # Global GRL
        elif args.task == 'link':
            X_base = GCN.link_prediction(G, args)
            X_CaaN = GCN.link_prediction(CaaN_sg, args)

    elif args.model == 'graphsage':
        from models import GraphSAGE
        if args.task == 'node': 
            X_base = GraphSAGE.node_classification(G, node_subjects, args)
            X_CaaN = GraphSAGE.node_classification(CaaN_sg, CaaN_subjects, args)
        elif args.task == 'link':
            X_base = GraphSAGE.link_prediction(G, args)
            X_CaaN = GraphSAGE.link_prediction(CaaN_sg, args)

    elif args.model == 'gat':
        from models import GAT
        if args.task == 'node': 
            X_base = GAT.node_classification(G, node_subjects, args)
            X_CaaN = GAT.node_classification(CaaN_sg, CaaN_subjects, args)
        elif args.task == 'link':
            X_base = GAT.link_prediction(G, args)
            X_CaaN = GAT.link_prediction(CaaN_sg, args)

    elif args.model == 'node2vec':
        from models import Node2Vec
        if args.task == 'node': 
            X_base = Node2Vec.node_classification(G, node_subjects, args)
            X_CaaN = Node2Vec.node_classification(CaaN_sg, CaaN_subjects, args)
        elif args.task == 'link':
            X_base = Node2Vec.link_prediction(G, args)
            X_CaaN = Node2Vec.link_prediction(CaaN_sg, args)

    start = time.time()
    node_embeddings = X 

    # Local GRL 
    for commu in range(len(cd_algo)):
        if len(cd_algo[commu]) >= size_thresh: # If Major Community
            sub_node_embeddings = subgraph_learning(cd_algo[commu], args)
            
            # Overwrite from subgraph embedding
            for i in cd_algo[commu]:
                node_embeddings[i] = sub_node_embeddings[i]

    print("Local GRL Time : ",time.time() - start)


if __name__ == "__main__":
    main()
