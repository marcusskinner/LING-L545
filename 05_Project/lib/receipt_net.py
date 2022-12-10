# Relevant Imports
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import dgl
import sys

from lib.gnn import GATNodeClassifier, GCNNodeClassifier, ChebNetNodeClassifier, TAGNodeClassifier, SAGENodeClassifier
from lib.graph_modeler import create_graph

class ReceiptNet():
    """
    
    """
    def __init__(self,
                 training_graph,
                 validation_graphs = None,
                 conv = 'sage',
                 seed = None):
        
        # Initialize the training and validation graphs
        self.training_graph = training_graph
        if validation_graphs is not None:
            self.validation_graphs = validation_graphs
        
        # setting the features
        price_ft = self.training_graph.ndata['feat']
        
        # find the labels and masks
        labels = self.training_graph.ndata['label']
        train_mask = self.training_graph.ndata['train_mask']
        test_mask = self.training_graph.ndata['test_mask']
        
        # initialize the models
        if conv == 'sage':
            self.model = SAGENodeClassifier(seed=seed, feat_drop=0.125)
        elif conv == 'gat':
            self.model = GATNodeClassifier()
        elif conv == 'graph':
            self.model = GCNNodeClassifier()
        elif conv == 'tag':
            self.model = TAGNodeClassifier()
        elif conv == 'cheby':
            self.model = ChebNetNodeClassifier()
        
        # train the models with dataset
        self.model.fit(self.training_graph, price_ft, labels, train_mask, test_mask)
        
        
    def get_seed():
        return self.model.seed
        
    def predict_most_probable(self, model, target_class, graph, features):
        """
        Predicts the node that has the highest value of belonging to the target_class. For example, if we have three nodes and
        a target_class = 3, with each node having probabilities 0.02, 0.04, 0.41, then the third node with be predicted.
        
        Parameters
        ----------
        model : obj
            the model used for prediction
        target_class : int
            the class to predict on
        graph : obj
            a dgl graph
        features : tensor
            a tensor of all node features used for prediction
        
        Returns
        ----------
        """
        logits = model.forward(graph, features)
        pred = logits.softmax(1)
        pred = torch.transpose(pred, 0, 1)
        return torch.argmax(pred[target_class]).item()
    
        
    def predict(self, img):
        """
        Predicts the price total on a receipt
        
        Parameters
        ----------
        img : obj
            A PIL Image of a receipt
        
        Returns : dict
            A dictionary containing the text for the price, date, and name
        """
        # Create graph from json file
        graph, ocr_text = create_graph(response)
        
        # setting the features
        price_ft = graph.ndata['feat']
        
        # make a prediction for price
        price_index = self.predict_most_probable(self.model, 1, graph, price_ft)
        
        # match the ocr text with the indices
        price = ocr_text[price_index]
        name = []
        for i in name_index:
            name += [ocr_text[i]]
        
        line_items = []
        for i in lineitem_index:
            line_items += [ocr_text[i]]
        
        # return the ocr text
        return {'Price-Total' : price}
            
    def score(self):
        """
        Calculates the percent of labels guessed correctly (true positives). Does not take into account the number of labels guessed 
        incorrectly (false positives).
        
        Parameters
        ----------
        validation_graphs : list obj
            A list of dgl graphs for validation
        """
        # if the validation graphs empty, return none
        if self.validation_graphs == None or len(self.validation_graphs) == 0:
            print("There are no validation graphs!")
            return None
        
        num_samples = len(self.validation_graphs)
        
        # Variables for True Positives (TP)
        TP_price = 0
        
        # Variables for False Positives (FP)
        FP_price = 0
        
        # Varialbes for True Negative (TN)
        TN_price = 0
        
        # Variables for False Negatives (FN)
        FN_price = 0
        
        # variables for other computation
        num_nodes = 0
        num_prices = 0
            
        for graph in self.validation_graphs:
            # setting the features
            price_ft = graph.ndata['feat']
            
            labels = graph.ndata['label'].tolist()
            
            price_index = self.predict_most_probable(self.model, 1, graph, price_ft)
            
            num_nodes += graph.num_nodes()
            num_prices += labels.count(1)
            
            # compute true positives and false positives
            if labels[price_index] == 1: TP_price += 1 
            else: FP_price += 1
                   
         # compute true negatives
        TN_price = (num_nodes - num_prices) - FP_price
            
        # compute false negatives
        FN_price = num_prices - TP_price
        
        # Compute Sensitivity
        sens_price = TP_price/(TP_price+FN_price)
        
        # Compute Specificity
        spec_price = TN_price/(TN_price+FP_price)
        
        # Compute Precision
        prec_price = TP_price/(TP_price+FP_price)
        
        # Compute F1 Score
        f1_price = 2 * ((prec_price*sens_price)/(prec_price+sens_price))
        
        # Compute Accuracy
        acc_price = (TP_price + TN_price)/(num_nodes)
        
        # comment out for more metrics
        """
        print('---PRICE EVALUATION---')
        print(f'Sensitivity: ' + f'{sens_price}')
        print(f'Specificity: ' + f'{spec_price}')
        print(f'Precision: ' + f'{prec_price}')
        print(f'F1-Score: ' + f'{f1_price}')
        print(f'Accuracy: ' + f'{acc_price}')
        """
        return prec_price