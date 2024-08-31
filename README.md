# SaaN 2L-GRL: Two-Level Graph Representation Learning Empowered with Subgraph-as-a-Node
This paper has been submitted for publication in [TKDE 2024](https://ieeexplore.ieee.org/document/10595395).

## Abstract

> In this study, we propose a novel graph representation learning (GRL) model, called Two-Level GRL with Subgraphas-a-Node (SaaN 2L-GRL in short), that partitions input graphs into smaller subgraphs for effective and scalable GRL in two levels: 1) local GRL and 2) global GRL. To realize the two-level GRL in an efficient manner, we propose an abstracted graph, called Subgraph-as-a-Node Graph (SaaN in short), to effectively maintain the high-level graph topology while significantly reducing the size of the graph. By applying the SaaN graph to both local and global GRL, SaaN 2L-GRL can effectively preserve the overall structure of the entire graph while precisely representing the nodes within each subgraph. Through time complexity analysis, we confirm that SaaN 2L-GRL significantly reduces the learning time of existing GRL models by using the SaaN graph for global GRL, instead of using the original graph and processing local GRL on subgraphs in parallel. Our extensive experiments show that SaaN 2L-GRL outperforms existing GRL models in both accuracy and efficiency. In addition, we show the effectiveness of SaaN 2L-GRL using diverse kinds of graph partitioning methods including five community detection algorithms and representative edge- and vertex-cut algorithms.

## Requirements
- python==3.6.13
- gensim==3.8.3
- igraph==0.9.11
- keras==2.6.0
- matplotlib==3.3.4
- networkx==2.5.1
- numpy==1.19.5
- pandas==1.1.5
- scikit-learn==0.24.2
- scipy==1.5.4
- stellargraph==1.2.1
- tensorflow==2.6.2
- torch_geometric==2.1
- pymetis
- dgl==2.0.0

## Implementation

In experiments, the source code for the three baseline models: [CNRL](https://arxiv.org/abs/1611.06645), [AnECI](https://ieeexplore.ieee.org/document/9835662), [GCA](https://dl.acm.org/doi/abs/10.1145/3442381.3449802)

is in the folders `models/CNRL`, `models/AnECI`, and `models/GCA`, respectively.


For a detailed description of running the model, see the original GitHub repository.

1. [CNRL](http://nlp.csai.tsinghua.edu.cn/%7Etcc/datasets/simplified_CNRL.zip) (TKDE 2018)

2. [AnECI](https://github.com/Gmrylbx/AnECI) (ICDE 2022)

3. [GCA](https://github.com/CRIPAC-DIG/GCA) (WWW 2021)

All other source codes in the paper are in the `codes` folder.

## Dataset

Dataset used in this study is provided in `datasets`

1. [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)

2. [Flickr](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

Other datasets (PubMed and CiteSeer) can be obtained from [stellargraph library](https://stellargraph.readthedocs.io/en/v0.9.0/_modules/stellargraph/datasets/datasets.html). 

## Run
- Metis partitioning
```python ./codes/metis.py```
- Libra partitioning
```python ./codes/libra_partitioning.py```
- Community detection
```python ./codes/communityDetection.py```
- Train GRL model
```python ./codes/main.py```
