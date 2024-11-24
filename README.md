This repository is the implementation of paper [Heterogeneous Graph Self Attention Networks(HGSAN)] submitted to Pattern Recognition.

## Installation

Install [pytorch](https://pytorch.org/get-started/locally/)

Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Data Preprocessing
We used datasets  provided by GTN(https://github.com/seongjunyun/Graph_Transformer_Networks).

Take DBLP as an example to show the formats of input data:

`node_features.pkl` is a numpy array whose shape is (num_of_nodes, num_of_features). It contains input node features.

`edges.pkl` is a list of scipy sparse matrices. Each matrix has a shape of (num_of_nodes, num_of_nodes) and is formed by edges of a certain edge type.

`labels.pkl` is a list of lists. labels[0] is a list containing training labels and each item in it has the form [node_id, target]. labels[1] and labels[2] are validation labels and test labels respectively with the same format.

`node_types.npy` is generated by `preprocess.py`. It is a numpy array which contains node type information and has a shape of (num_of_nodes,). Each value in the array is an integer and lies in [0, num_of_node_types).

Note that the inputs of our method are only raw information of a heterogeneous network (network topology, node types, edge types, and node attributes if applicable). We do not need to manually design any meta path or meta graph.

## Running the code
``` 
$ mkdir data
$ cd data
```
Download datasets (DBLP, ACM, IMDB) from this [link](https://drive.google.com/file/d/1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5/view?usp=sharing) and extract data.zip into data folder.
```
$ cd ..
```
- DBLP  --GlobalAttention：dropout=0.0 lr=0.0005 head=8; GraphTrans：dropout=0.0 lr=0.002 head=8; other: lr=0.002 
```
python DBLP_run.py --dataset DBLP  --norm false --adaptive_lr True
```
- ACM  --GlobalAttention：dropout=0.5 lr=0.0005 head=1; GraphTrans：dropout=0.5 lr=0.001 head=1; other: lr=1e-4
```
python ACM_run.py --dataset ACM  --norm true --adaptive_lr True
```
- IMDB  --GlobalAttention：dropout=0.5 lr=0.0004 head=8; GraphTrans：dropout=0.5 lr=0.002 head=8; other: lr=1e-4
```
python IMDB_run.py --dataset IMDB  --norm false --adaptive_lr True
```


