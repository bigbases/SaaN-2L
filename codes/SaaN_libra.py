import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.datasets import *
from torch_geometric.data import NeighborSampler as RawNeighborSampler
from torch_geometric.utils.convert import from_networkx as fn
from torch_geometric.utils.convert import to_networkx
from torch import Tensor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import json
from networkx.readwrite import json_graph
import socket
import pickle
import threading
import argparse
import time
from pssh.clients.ssh.parallel import ParallelSSHClient
import threading
import select
import argparse
import copy
import tqdm
import numpy as np
import pandas as pd
import dgl
import igraph as ig
import math
class NodeClassifier(torch.nn.Module):
    def __init__(self,input_dim,classes):
        super(NodeClassifier, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32,classes)
        )

    def forward(self, x):
        output = self.layers(x)
        return output
def set_new_attr(ver_seq,res_attr):
    for i in range(len(ver_seq.vs)):
        ver_seq.vs[i]['feat'] = res_attr[i]
    return ver_seq
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_channels) # convolution layer
        self.conv2 = GATConv(hidden_channels, hidden_channels) # 일반 GCN 레이어로 바꿔보기
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
def run_gat(graph, epoch, batch,community_num,num_partition):
    global embedding_list
    global embedding_dit
    model = GAT(graph.x.shape[1],128) # 512로 바꾸기
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = epoch
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(graph.x.to(torch.float), graph.edge_index)
        loss = loss = criterion(out, graph.label)
        loss.backward()
        optimizer.step()
        #print('epoch {}, loss {}'.format(epoch, loss.item()))
        
    if(community_num=="global"):
        embedding = model(graph.x.to(torch.float),graph.edge_index)
        embedding = embedding.detach()
        embedding = embedding[num_partition:]
        label_global = graph.label[num_partition:]
        embedding_list.append(embedding)
        y_list.append(label_global)
        embedding_dit[str(community_num)] = embedding
        y_dit[str(community_num)] = label_global
    else:
        embedding = model(graph.x.to(torch.float),graph.edge_index)
        embedding = embedding.detach()
        embedding = embedding[:-(num_partition-1)]
        embedding_list.append(embedding)
        y_list.append(graph.label[:-(num_partition-1)])
        embedding_dit[str(community_num)] = embedding
        y_dit[str(community_num)] = graph.label
    return embedding
if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser("Multi-Level GRL")

    parser.add_argument("--file_path", type=str, default="./libra")
    parser.add_argument("--partition", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="flickr")
    args = parser.parse_args()

    num_partition = args.partition
    dataset = args.dataset
    file_path = args.file_path
    epoch  = args.epoch
    batch = args.batch
    
    if dataset == "flickr":
        with open('./dataset/flickr_dgl.pickle','rb') as f:
            G_origin = pickle.load(f)

    
    df_replica = pd.read_csv(file_path + "/selected_replica_most_edges.csv")
    with open(file_path + '/SaaN_revised_including_nodes.pickle', 'rb') as f:
        SaaN = pickle.load(f)
    with open(file_path + '/clustering.pickle', 'rb') as f:
        clust = pickle.load(f)
    SaaN_tmp = SaaN
    
    nx_origin = dgl.to_networkx(G_origin, node_attrs=['x','label'])
    ig_origin = ig.Graph.from_networkx(nx_origin)
    #clust = ig.VertexClustering(ig_g, membership=membership)
    
    # 클러스터링 된 그래프, 원본 그래프

    graph_list = []
    for j in tqdm.tqdm(range(num_partition)): # 커뮤니티 0번부터 순회
        ig_origin = ig.Graph.from_networkx(nx_origin)
        clust_tmp = clust
        membership = clust_tmp.membership

        for idx, row in df_replica.iterrows(): # 중복되는 노드 처리 -metis에서는 안봐도 되는 코드
            number = row['id']
            comcom = row["community"].split(",")
            if str(j) in comcom:
                membership[number] = j

        com_len = len([x for x in membership if x == j]) # 커뮤니티 n번에 속하는 노드들의 개수

        # 다른 커뮤니티 노드 추가하기

        part_membership = list(range(0,com_len)) # 커뮤니티 n번에 속하는 노드들을 개별 파티션으로 할당
        for aa in range(len(membership)):
            if membership[aa] < j:
                membership[aa] += com_len
            elif membership[aa] > j:
                membership[aa] += com_len - 1
        tmp_num = 0
        for idx in range(len(membership)):
            if membership[idx] == j:
                membership[idx] = part_membership[tmp_num]
                tmp_num += 1
                if tmp_num == com_len:
                    break

        clust_new = ig.VertexClustering(ig_origin, membership=membership)
        res_li = []
        nn2 = 0
        for i in clust_new.subgraphs():
            vs11 = np.array(torch.stack(i.vs["x"]))
            res = np.mean(vs11, axis=0)
            res_li.append(res)
            nn2 += 1


        ig_origin.contract_vertices(membership, combine_attrs="first")

        final = set_new_attr(ig_origin,res_li)
        final = final.simplify()
        final = fn(final.to_networkx()) # igraph -> pyG
        final["x"] = final["feat"]
        graph_list.append(final)

    global embedding_list
    global embedding_dit
    global y_list
    embedding_list = []
    y_list = []
    embedding_dit = {}
    y_dit = {}
    tmp = math.ceil(num_partition/20) # the number of graph / the number of cores
    print(tmp)
    threads = []
    start = time.time()
    for i in range(tmp):
        graph_lili = graph_list[i*20:(i+1)*20]
        for graph in graph_lili:
            idx = graph_list.index(graph)
            t= threading.Thread(target=run_gat,args=[graph,epoch,batch,idx,num_partition])
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
    end = time.time() - start
    print(f"GRL time : {end} secs")
    tmp_li = list(range(num_partition))
    tmp_dit = dict(map(lambda i : (i,[]) , tmp_li)) # for replicated nodes
    repli_dit = {}
    repli_y = {}

    for index,row in df_replica.iterrows():
        node_id = row["id"]
        select = row["selected_community"]
        deleted_commu = row["deleted_community"]
        commu = eval(deleted_commu)
        commu.append(select)
        lili = []
        #repli_dit[str(node_id)] = []
        for i in commu:
            tmp_dit[int(i)].append(row["community"+str(i)]) # for 
            lili.append(embedding_dit[str(i)][row["community"+str(i)]])
            yy = y_dit[str(i)][row["community"+str(i)]]
        averaging = torch.mean(torch.stack(lili), 0,True)

        repli_dit[str(node_id)] = averaging
        repli_y[str(node_id)] = yy.reshape(-1)

    new_dit = {}
    for i in tmp_dit.keys():
        overall_len = embedding_dit[str(i)].shape[0]
        tmp_li2 = list(range(overall_len))
        new_list = [x for x in tmp_li2 if x not in tmp_dit[i]] # excluding replications
        new_dit[i] = new_list
    new_embed = {}
    new_y = {}
    #new_embed["global"] = embedding_dit["global"]
    #new_y["global"] = y_dit["global"]
    for i in new_dit.keys():
        new_embed[str(i)] = embedding_dit[str(i)][new_dit[i]]
        new_y[str(i)] = y_dit[str(i)][new_dit[i]]
    for i in repli_dit.keys():
        new_embed["node"+i] = repli_dit[i]
        new_y["node"+i] = repli_y[i]
    embed = torch.cat(list(new_embed.values()),0)
    label = torch.cat(list(new_y.values()),0)
    model = NodeClassifier(embed.shape[1],label.unique().shape[0]+1) # +1 is when using flickr
    x_train, x_test, y_train, y_test = train_test_split(embed, label, test_size=0.2, shuffle=True, random_state=45) 

    #train set 을 한번 더 나누어서 validation set 또한 만들어 낼 수 있습니다. 
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=True, random_state=45)
    #x_train, x_test, y_train, y_test = train_test_split(embed, label, test_size=0.2, random_state=42)
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    #n_epochs = args.num_epochs
    n_epochs = epoch
    batch_size = batch
    batches_per_epoch = len(x_train) // batch_size

    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                X_batch = x_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # compute and store metrics
                acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(x_val)
        ce = loss_fn(y_pred, y_val)
        acc = (torch.argmax(y_pred, 1) == y_val).float().mean()
        ce = float(ce)
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)

        print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")

    model.load_state_dict(best_weights)

    pred = model(x_test).argmax(dim=1)
    correct = (pred == y_test).sum()
    acc = int(correct) / int(x_test.shape[0])
    print(f'Accuracy: {acc}')