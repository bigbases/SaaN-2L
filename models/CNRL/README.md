# CNRL
## Requirements
1. `gensim 0.10.2`
2. `Cython 0.25.1`

## Preparation for Training and Testing
1. Run `make all`

## Elavuation
### Link prediction (refer to `lda_link.py`)
1. --input input_graph (default form: adjacency list)
2. --cene 1 (SCNRL), --cene 2 (ECNRL)
3. --fold n_folds
4. --reset 1 (restart training)
6. --cenecmty 20 (community number)


### Vertex classification (refer to `classify.py`)
1. --input input_graph (default form: edge list)
2. --label label_file
3. --metric 7 (SCNRL) --metric 8 (ECNRL) --metric 0 (node2vec)
4. --reset 1 (restart training)
5. --cmty 20 (community number)

## Demo
1. link prediction: `python lda_link.py --input lp/wiki/graph.txt_adjList --reset 1 --cene 2 --fold 20 --cenecmty 100`
2. vertex classification: `python classify.py --input data_classify/citeseer/graph.txt --label data_classify/citeseer/group.txt --reset 1 --metric 8 --cmty 100`