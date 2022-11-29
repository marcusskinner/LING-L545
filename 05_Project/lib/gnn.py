import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import dgl
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)


class GATNodeClassifier(nn.Module):
    """
    A class that implements the Graph Attention convolutions from dgl. 
    
    Attributes
    ----------
    num_heads : int
        Number of heads in Multi-Head Attention
    num_layers : int
        The number of layers in the neural network; each layer includes a convolutional layer, hidden layers, followed by another convolutional layer.
    num_out_heads : int
        Number of output heads in Multi-Head Attention
    num_hidden : int
        Number of hidden layers in each layer of the network
    in_drop : float
        Dropout rate on input features
    attn_drop : float
        Dropout rate on attention weight
    negative_slope : float
        LeakyReLU angle of negative slope
    residual : bloolean
        If true, use residual connection
    lr : float
        the learning rate of the network
    weight_decay : float
        weight_decay
    n_epochs : int
        the number of epochs during training
    early_stop : boolean
        If true, use early stopping
    activation : function
        the activation function in the network
        
    Methods
    ----------
    forward()
        feeds the input through the network
    fit()
        trains the model on a graph
    predict()
        predicts the nodes on an input graph
    score()
        computes the accuracy on the input mask
    """
    def __init__(self,
                 num_heads = 8,
                 num_layers = 1,
                 num_out_heads = 1,
                 num_hidden = 8,
                 in_drop = 0.6,
                 attn_drop = 0.6,
                 negative_slope = 0.2,
                 residual = False,
                 lr = 0.005,
                 weight_decay = 5e-4,
                 n_epochs = 140,
                 early_stop = False,
                 activation = F.relu,
                 seed = None):
        super(GATNodeClassifier, self).__init__()
        
        # set parameters
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.num_heads = num_heads
        self.num_out_heads = num_out_heads
        self.num_hidden = num_hidden
        self.in_drop = in_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.activation = activation
        self.heads = ([num_heads] * num_layers) + [num_out_heads]
        self.seed = seed
        
    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits
    
    def fit(self, g, features, labels, train_mask, test_mask):
        # get the seed if set
        if self.seed is not None:
            torch.set_rng_state(self.seed)
        else: self.seed = torch.get_rng_state()
        
        # Calculate graph parameters
        in_dim = features.shape[1]
        num_classes = torch.max(labels).item() + 1
        n_edges = g.number_of_edges()
        
        # input projection (no residual)
        self.gat_layers.append(dgl.nn.GATConv(
            in_dim, self.num_hidden, self.heads[0],
            self.in_drop, self.attn_drop, self.negative_slope, False, self.activation))
        
        # hidden layers
        for l in range(1, self.num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(dgl.nn.GATConv(
                self.num_hidden * self.heads[l-1], self.num_hidden, self.heads[l],
                self.in_drop, self.attn_drop, self.negative_slope, self.residual, self.activation))
            
        # output projection
        self.gat_layers.append(dgl.nn.GATConv(
            self.num_hidden * self.heads[-2], num_classes, self.heads[-1],
            self.in_drop, self.attn_drop, self.negative_slope, self.residual, None))
        
        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        best_test_acc = 0
        for epoch in range(self.n_epochs):
            self.train()
            # forward
            logits = self.forward(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            test_acc = self.score(g, features, labels, test_mask)
            if(test_acc > best_test_acc): best_test_acc = test_acc
            #print('In epoch {}, loss: {:.3f}, test acc: {:.3f} (best {:3f})'.format(epoch, loss, test_acc, best_test_acc))

    def predict(self, g, features):
        # for each class, predict the node most likely to belong to it
        logits = self.forward(g, features)
        pred = logits.argmax(1)
            
        return pred
            
    def score(self, g, features, labels, mask):
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)


class GCNNodeClassifier(nn.Module):
    """
    A class that implements the GraphConv from dgl into a neural network.
    
    Attributes
    ----------
    num_layers : int
        The number of layers in the neural network; each layer includes a convolutional layer, hidden layers, followed by another convolutional layer.
    num_hidden : int
        Number of hidden layers in each layer of the network
    dropout : float
        Dropout rate
    lr : float
        the learning rate of the network
    weight_decay : float
        weight_decay
    n_epochs : int
        the number of epochs during training
    activation : function
        the activation function in the network
        
    Methods
    ----------
    forward()
        feeds the input through the network
    fit()
        trains the model on a graph
    predict()
        predicts the nodes on an input graph
    score()
        computes the accuracy on the input mask
    """
    def __init__(self,
                 n_hidden=16,
                 n_layers=1,
                 activation=F.relu,
                 dropout=0.5,
                 lr=1e-2,
                 weight_decay=5e-4,
                 n_epochs=200,
                 seed=None):
        super(GCNNodeClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
    
    def fit(self, g, features, labels, train_mask, test_mask):
        # get the seed if set
        if self.seed is not None:
            torch.set_rng_state(self.seed)
        else: self.seed = torch.get_rng_state()
            
        # Calculate graph parameters
        in_feats = features.shape[1]
        n_classes = torch.max(labels).item() + 1
        n_edges = g.number_of_edges()
        
        # input layer
        self.layers.append(dgl.nn.GraphConv(in_feats, self.n_hidden, activation=self.activation))
        
        # hidden layers
        for i in range(self.n_layers - 1):
            self.layers.append(dgl.nn.GraphConv(self.n_hidden, self.n_hidden, activation=self.activation))
            
        # output layer
        self.layers.append(dgl.nn.GraphConv(self.n_hidden, n_classes))

        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        g.ndata['norm'] = norm.unsqueeze(1)
        
        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        
        best_test_acc = 0
        for epoch in range(self.n_epochs):
            self.train()
            # forward
            logits = self.forward(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            test_acc = self.score(g, features, labels, test_mask)
            if(test_acc > best_test_acc): best_test_acc = test_acc
            #print('In epoch {}, loss: {:.3f}, test acc: {:.3f} (best {:3f})'.format(epoch, loss, test_acc, best_test_acc))
        
    def predict(self, g, features):
        # for each class, predict the node most likely to belong to it
        logits = self.forward(g, features)
        pred = logits.argmax(1)
            
        return pred
    
    def score(self, g, features, labels, mask):
        # add self loop
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)
        
class ChebNetNodeClassifier(nn.Module):
    """
    A class that implements the ChebConv (aka the spectral convolution) from dgl into a neural network.
    
    Attributes
    ----------
    n_layers : int
        The number of layers in the neural network; each layer includes a convolutional layer, hidden layers, followed by another convolutional layer.
    n_hidden : int
        Number of hidden layers in each layer of the network
    dropout : float
        Dropout rate
    lr : float
        the learning rate of the network
    weight_decay : float
        weight_decay
    n_epochs : int
        the number of epochs during training
    k : int
        something idk
        
    Methods
    ----------
    forward()
        feeds the input through the network
    fit()
        trains the model on a graph
    predict()
        predicts the nodes on an input graph
    score()
        computes the accuracy on the input mask
    """
    def __init__(self,
                 n_hidden=16,
                 n_layers=1,
                 k=3,
                 lr=1e-1,
                 weight_decay=5e-4,
                 n_epochs=200,
                 seed=None):
        super(ChebNetNodeClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.k = k
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.seed = seed
        
    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h, [2])
        return h
    
    def fit(self, g, features, labels, train_mask, test_mask):
        # get the seed if set
        if self.seed is not None:
            torch.set_rng_state(self.seed)
        else: self.seed = torch.get_rng_state()
            
        # Calculate graph parameters
        in_feats = features.shape[1]
        n_classes = torch.max(labels).item() + 1
        n_edges = g.number_of_edges()
        
        self.layers.append(
            dgl.nn.ChebConv(in_feats, self.n_hidden, self.k))
        
        for _ in range(self.n_layers - 1):
            self.layers.append(
                dgl.nn.ChebConv(self.n_hidden, self.n_hidden, self.k))

        self.layers.append(
            dgl.nn.ChebConv(self.n_hidden, n_classes, self.k))
        
        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        
        for epoch in range(self.n_epochs):
            self.train()
            
            # forward
            logits = self.forward(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('In epoch {}, loss: {:.3f})'.format(epoch, loss))
            
    def predict(self, g, features):
        # for each class, predict the node most likely to belong to it
        logits = self.forward(g, features)
        pred = logits.argmax(1)
            
        return pred

    def score(self, g, features, labels, mask):
        # add self loop
        self.eval()
        with torch.no_grad():
            logits = model(g, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)
        
class TAGNodeClassifier(nn.Module):
    """
    A class that implements the GraphConv from dgl into a neural network.
    
    Attributes
    ----------
    num_layers : int
        The number of layers in the neural network; each layer includes a convolutional layer, hidden layers, followed by another convolutional layer.
    num_hidden : int
        Number of hidden layers in each layer of the network
    dropout : float
        Dropout rate
    lr : float
        the learning rate of the network
    weight_decay : float
        weight_decay
    n_epochs : int
        the number of epochs during training
    activation : function
        the activation function in the network
    k : int
        something
    bias : bool
        something
        
    Methods
    ----------
    forward()
        feeds the input through the network
    fit()
        trains the model on a graph
    predict()
        predicts the nodes on an input graph
    score()
        computes the accuracy on the input mask
    """
    def __init__(self,
                 n_hidden=16,
                 n_layers=1,
                 activation=F.relu,
                 dropout=0.5,
                 lr=1e-2,
                 weight_decay=5e-4,
                 n_epochs=200,
                 seed=None,
                 k=2,
                 bias=True):
        super(TAGNodeClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.k = k
        self.bias = bias

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
    
    def fit(self, g, features, labels, train_mask, test_mask):
        # get the seed if set
        if self.seed is not None:
            torch.set_rng_state(self.seed)
        else: self.seed = torch.get_rng_state()
            
        # Calculate graph parameters
        in_feats = features.shape[1]
        n_classes = torch.max(labels).item() + 1
        n_edges = g.number_of_edges()
        
        # input layer
        self.layers.append(dgl.nn.TAGConv(in_feats, self.n_hidden, activation=self.activation, bias=self.bias, k=self.k))
        
        # hidden layers
        for i in range(self.n_layers - 1):
            self.layers.append(dgl.nn.TagConv(self.n_hidden, self.n_hidden, activation=self.activation, bias=self.bias, k=self.k))
            
        # output layer
        self.layers.append(dgl.nn.TAGConv(self.n_hidden, n_classes))

        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        g.ndata['norm'] = norm.unsqueeze(1)
        
        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        
        best_test_acc = 0
        for epoch in range(self.n_epochs):
            self.train()
            # forward
            logits = self.forward(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            test_acc = self.score(g, features, labels, test_mask)
            if(test_acc > best_test_acc): best_test_acc = test_acc
            #print('In epoch {}, loss: {:.3f}, test acc: {:.3f} (best {:3f})'.format(epoch, loss, test_acc, best_test_acc))
        
    def predict(self, g, features):
        # for each class, predict the node most likely to belong to it
        logits = self.forward(g, features)
            
        return torch.Tensor(predictions).to(torch.int)
    
    def score(self, g, features, labels, mask):
        # add self loop
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)
        
        
class SAGENodeClassifier(nn.Module):
    """
    A class that implements the SAGEConv from dgl into a neural network.
    
    Attributes
    ----------
    num_layers : int
        The number of layers in the neural network; each layer includes a convolutional layer, hidden layers, followed by another convolutional layer.
    num_hidden : int
        Number of hidden layers in each layer of the network
    dropout : float
        Dropout rate
    lr : float
        the learning rate of the network
    weight_decay : float
        weight_decay
    n_epochs : int
        the number of epochs during training
    activation : function
        the activation function in the network
    k : int
        something
    bias : bool
        something
        
    Methods
    ----------
    forward()
        feeds the input through the network
    fit()
        trains the model on a graph
    predict()
        predicts the nodes on an input graph
    score()
        computes the accuracy on the input mask
    """
    def __init__(self,
                 n_hidden=16,
                 n_layers=1,
                 lr=1e-2,
                 dropout=0.5,
                 weight_decay=5e-4,
                 n_epochs=200,
                 seed=None,
                 feat_drop=0,
                 activation=F.relu,
                 aggregator_type='mean',
                 norm=None,
                 bias=True):
        super(SAGENodeClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.feat_drop=feat_drop
        self.bias = bias
        self.aggregator_type=aggregator_type
        self.norm = norm

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
    
    def fit(self, g, features, labels, train_mask, test_mask):
        # get the seed if set
        if self.seed is not None:
            torch.set_rng_state(self.seed)
        else: self.seed = torch.get_rng_state()
            
        # Calculate graph parameters
        in_feats = features.shape[1]
        n_classes = torch.max(labels).item() + 1
        n_edges = g.number_of_edges()
        
        # input layer
        self.layers.append(dgl.nn.SAGEConv(in_feats, self.n_hidden, aggregator_type=self.aggregator_type, feat_drop=self.feat_drop, activation=self.activation, bias=self.bias, norm=self.norm))
        
        # hidden layers
        for i in range(self.n_layers - 1):
            self.layers.append(dgl.nn.SAGEConv(self.n_hidden, self.n_hidden, aggregator_type=self.aggregator_type, feat_drop=self.feat_drop, activation=self.activation, bias=self.bias, norm=self.norm))
            
        # output layer
        self.layers.append(dgl.nn.SAGEConv(self.n_hidden, n_classes, self.aggregator_type))

        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        g.ndata['norm'] = norm.unsqueeze(1)
        
        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        
        best_test_acc = 0
        for epoch in range(self.n_epochs):
            self.train()
            # forward
            logits = self.forward(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            test_acc = self.score(g, features, labels, test_mask)
            if(test_acc > best_test_acc): best_test_acc = test_acc
            #print('In epoch {}, loss: {:.3f}, test acc: {:.3f} (best {:3f})'.format(epoch, loss, test_acc, best_test_acc))
        
    def predict(self, g, features):
        # for each class, predict the node most likely to belong to it
        logits = self.forward(g, features)
        pred = logits.argmax(1)
            
        return pred
    
    def score(self, g, features, labels, mask):
        # add self loop
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)