import pickle
import dgl
from dgl.distgnn.partition import libra_partition as libra
import time
import argparse
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser("Multi-Level GRL")

    parser.add_argument("--output_path", type=str, default="./libra")
    parser.add_argument("--partition", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="flickr")
    
    args = parser.parse_args()

    partitions = args.partition
    dataset = args.dataset
    output_path = args.output_path
    if dataset == "flickr":
        with open('./dataset/flickr_dgl.pickle','rb') as f:
            g = pickle.load(f)


    start = time.time()
    result = libra.partition_graph(partitions, g, output_path)
    end = time.time() - end
    print(end)
    
    print("="*10)
    print("Making SaaN graph")
    t1 = time.time()
    import pandas as pd
    community_len = partitions # argu1
    dataset_path = output_path # argu2
    path_original_data = g # argu3
    folder = dataset_path + "/"+str(community_len)+"Communities"
    df_replica = pd.read_csv(folder+"/replicationlist.csv")
    df_replica = df_replica.drop_duplicates(ignore_index=True)
    print("the # of replication : ",len(df_replica))
    import os
    file_list = os.listdir(folder)
    file_list = [file for file in file_list if file.endswith(".txt")]
    part_num = len(file_list)
    communities = []
    for i in range(part_num):
        community = pd.read_csv(folder+'/community'+str(i)+'.txt', sep=",", header=None)
        community.columns = ["src", "dst", "y"]
        communities.append(community)
    nodes_tmp = list(set(communities[0]["src"].tolist() + communities[0]["dst"].tolist()))
    li = []
    li2 = []
    for i in tqdm(df_replica["id"]):
        num = ""
        num1 = 0
        for j in range(part_num):
            if i in communities[j]["src"].values or i in communities[j]["dst"].values:
                num += str(j) + ","
                num1 += 1
        num = num.rstrip(",")
        li.append(num)
        li2.append(num1)
    df_replica["community"] = li
    df_replica["community_num"] = li2
    import random
    li3 = []
    li4 = []
    for i in tqdm(df_replica["community"]):
        commu_li = i.split(",")
        random_element = random.choice(commu_li)
        li3.append(random_element)
        commu_li.remove(random_element)
        li4.append(commu_li)
    df_replica["selected_community"] = li3
    df_replica["deleted_community"] = li4
    for j in range(part_num):
        df_replica["community"+str(j)] = -1
        
        
    lilili = []
    for index,row in df_replica.iterrows():
        number = row["id"]
        deleted_commu = row["deleted_community"]
        selected = row["selected_community"]
        # print(number, " ",deleted_commu)
        tmp = list(set(communities[int(selected)]["src"].tolist() + communities[int(selected)]["dst"].tolist()))
        tmp.sort()
        idx = tmp.index(int(number))
        df_replica["community"+str(selected)][index] = idx
        #communities[int(selected)].loc[communities[int(selected)]["src"]==number,"src"] = -1 # selected 도 -1로 바꿔야함 -> 슈퍼노드로 만들기 위함
        #communities[int(selected)].loc[communities[int(selected)]["dst"]==number,"dst"] = -1
        for i in deleted_commu:
            for j in range(part_num):
                if i==str(j):
                    lilili.append(j)
                    nodes_tmp = list(set(communities[j]["src"].tolist() + communities[j]["dst"].tolist()))
                    nodes_tmp.sort()
                    idx = nodes_tmp.index(int(number))
                    df_replica["community"+str(j)][index] = idx 
                    #communities[j].loc[communities[j]["src"]==number,"src"] = -1 # 없앨거면 다 없애는 게 맞지 않음..? 
                    #지금 방식에서는 replicaion이 존재하는 노드들을 따로 빼서 슈퍼노드로 만드는 것이므로 선택된 커뮤니티에 있는 애들도 없애야 함
                    # grl 이전에 삭제하는 경우에는 서브 그래프 만드는 과정에서 deletec commu만 -1하면 됨
                    #communities[j].loc[communities[j]["dst"]==number,"dst"] = -1
                else:
                    pass
    df_replica.to_csv(output_path+"/selected_replica.csv",index=False)
    
    for k in range(part_num):
        a = list(set(communities[k]["src"].tolist() + communities[k]["dst"].tolist()))
        a.sort()
        if (a[0]!=-1):
            print("community",k," : ",a[0])
    # load original data
    G = g
    feat_all = G.ndata['feat']
    label_all = G.ndata['label']

    # find unique
    all_member_dict = {}
    for j in range(part_num):
        nodes0 = list(set(communities[j]["src"].tolist() + communities[j]["dst"].tolist()))
        nodes0.sort()
        #nodes0.remove(-1)
        li0 = [j for i in range(len(nodes0))]
        member0 = dict(map(lambda i,j : (i,j) , nodes0,li0))
        all_member_dict.update(member0)

    #node_replicated = df_replica["id"].tolist()
    #idx_replicated = list(range(part_num,part_num + len(node_replicated)))
    #member_replicated = dict(map(lambda i,j : (i,j) , node_replicated,idx_replicated))
    #all_member_dict.update(member_replicated)
    li_test = list(range(0,G.ndata["label"].shape[0])) # 노드 수
    no_li = []

    for i in li_test:
        if i not in all_member_dict.keys():
            no_li.append(i)
    #no_li = [x for x in no_li if x not in node_replicated]
    lili = list(range(part_num,part_num+1+len(no_li)))
    member_none = dict(map(lambda i,j : (i,j) , no_li,lili))
    all_member_dict.update(member_none)
    all_member_dict = dict(sorted(all_member_dict.items()))

    membership = list(all_member_dict.values()) # GRL 이후에 선택하는 경우를 위한 멤버십 리스트!

    import dgl
    nx_g = dgl.to_networkx(G, node_attrs=['x','label'])
    import igraph as ig
    ig_g = ig.Graph.from_networkx(nx_g)
    clust = ig.VertexClustering(ig_g, membership=membership)

    import pickle
    # save
    with open(output_path+'/clustering.pickle', 'wb') as f:
        pickle.dump(clust, f, pickle.HIGHEST_PROTOCOL)

    import numpy as np
    import torch
    res_li = []
    nn = 0
    for i in clust.subgraphs():
        vs11 = np.array(torch.stack(i.vs["x"]))
        res = np.mean(vs11, axis=0)
        res_li.append(res)
        nn += 1
    def set_new_attr(ver_seq,res_attr):
        for i in range(len(ver_seq.vs)):
            ver_seq.vs[i]['feat'] = res_attr[i]
        return ver_seq
    ig_g.contract_vertices(membership, combine_attrs="first")
    final = set_new_attr(ig_g,res_li)
    final = final.simplify()
    SaaN = final
    # save
    with open(output_path+'/SaaN_revised_including_nodes.pickle', 'wb') as f:
        pickle.dump(SaaN, f, pickle.HIGHEST_PROTOCOL)
    t2 = time.time() - t1
    print(f"Building SaaN graph : ",t2)

    
