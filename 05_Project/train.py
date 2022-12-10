# Imports
from datasets import list_datasets, load_dataset
import json
import dgl
import torch as th
from dgl.data.utils import load_graphs
from lib.receipt_net import ReceiptNet
import os
import pickle
from lib.graph_modeler import load_normalized_example, graph_modeling, normalize_graph
import sys

import warnings


def train():
    
    print("LOADING DATASET")
    dataset = load_dataset("naver-clova-ix/cord-v2")
    
    graph_data = []

    for key in ['train', 'test']:

        for receipt in dataset[key]:

            word_areas, labels, width, height, ocr_text, features = load_normalized_example(receipt)
            graph = graph_modeling(word_areas, labels, width, height, features)
            graph_data.append(graph)

        big_graph = dgl.batch(graph_data)
        big_graph = dgl.add_self_loop(big_graph)

        num_nodes = big_graph.number_of_nodes()

        test_mask = th.zeros(num_nodes)
        train_mask = th.zeros(num_nodes)

        test_mask = test_mask.to(th.bool)
        train_mask = train_mask.to(th.bool)

        train_mask[:len(dataset['train'])] = 1
        test_mask[len(dataset['train']):] = 1

        big_graph.ndata['test_mask'] = test_mask
        big_graph.ndata['train_mask'] = train_mask
        big_graph.ndata['label'] = big_graph.ndata['label'].to(th.long)

        big_graph = normalize_graph(big_graph)

    dgl.save_graphs('data/graph_training.bin', big_graph)
    training_graph = load_graphs('data/graph_training.bin')[0][0]
    
    validation_graphs = []

    # Load validation graphs
    for receipt in dataset['validation']:

        word_areas, labels, width, height, ocr_text, features = load_normalized_example(receipt)
        graph = graph_modeling(word_areas, labels, width, height, features)
        graph = normalize_graph(graph)
        graph = dgl.add_self_loop(graph)

        validation_graphs.append(graph)

    # train the model
    scores = []
    convs = ['sage', 'graph', 'gat', 'tag','cheby']
    for c in convs:
        clf = ReceiptNet(training_graph, validation_graphs, conv=c)
        scores.append(clf.score())

    # notify user that training is complete and print results
    print('Training complete.')
    print("RESULTS")
    for i in range(len(convs)):
        print(convs[i] + ": " + str(scores[i]))


if __name__ == '__main__':
    train()
    sys.exit(0)