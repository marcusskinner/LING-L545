# Relevant Imports
import pathlib
import numpy as np
import sys
import json
import random
import torch as th
import numpy as np
import dgl
from sklearn.preprocessing import normalize
import re


def load_normalized_example(receipt_data):
    """
    Extracts the information needed to create a dgl graph from a json file in the cord-v2 format. If a different
    format is used, this function will need to be remade.
    
    Parameters
    ----------
    receipt_dict : a dictinary containing an example from the cord-v2 dataset.
    
    Returns
    ----------
    word_areas : dict
        a dictionary of all the features in the json file
    labels : list
        a list of each text box's label in the json file
    width : int
        the width (in pixels) of the receipt image
    height : int
        the height (in pixels) of the receipt image
    ocr_text : list [str]
        a list of the text in each textbox of the receipt, in order of the node index
    """
    word_areas = []
    labels = []
    ocr_text = []
    features = []
    
    width = receipt_data['image'].width
    height = receipt_data['image'].height
    
    receipt_dict = json.loads(receipt_data['ground_truth'])
    
    for line in receipt_dict['valid_line']:
        
        for word in line['words']:
            
            word_areas.append([word['quad']['x1'], word['quad']['x2'], word['quad']['y1'], word['quad']['y3']])
            ocr_text.append(word['text'])
            
            is_price, is_date, is_numeric = get_string_features(word['text'])
            features.append([is_price, is_date, is_numeric])
            
            # A little inefficient but the dataset uses multiple labels for different price totals. This is the best I got
            # to figure out the price total on a receipt.
            if 'total' in receipt_dict['gt_parse'] and 'total_price' in receipt_dict['gt_parse']['total']:
                
                price = receipt_dict['gt_parse']['total']['total_price']
                
            elif 'sub_total' in receipt_dict['gt_parse'] and 'sub_total_price' in receipt_dict['gt_parse']['sub_total']:    
                
                price = receipt_dict['gt_parse']['sub_total']['sub_total_price']
                
            else:
                
                price = None
                
            if word['text'] == price:
                labels.append(1)
            else:
                labels.append(0)
        
    return np.array(word_areas), labels, width, height, ocr_text, features

def compute_parameters(word_areas):
    """
    Computes the parameters to model the graph
    
    Parameters
    ----------
    word_areas : dict
        the information extracted from the json file using the load_normalized_example method
    
    Returns
    ----------
    midpoints : list [int]
        a list of integers representing the midpoint of each textbox in the receipt
    textbox_dims : list [[int, int]]
        a list of the width and height of each textbox
    corners : list [[int, int]]
        a list of the minimum x value and the maximum y value of each textbox
    features : list [[float, boolean, boolean, float]]
        a list of the features tf-idf, is-price, is-date, and area of each textbox
    """
    midpoints = []
    textbox_dims = []
    corners = []
    
    for word in word_areas:
        x_min = float(word[0])
        x_max = float(word[1])
        y_min = float(word[2])
        y_max = float(word[3])
        
        x = (x_min + x_max)//2
        y = (y_min + y_max)//2
        
        height = y_max - y_min
        width = x_max - x_min
        
        midpoints += [[x,y]]
        corners += [[[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]]]
        textbox_dims += [[width, height]]
    
    return np.array(midpoints), np.array(textbox_dims), corners


# takes in a line and determines some features and characteristics about the line
# such as if the number contains a price if so whats the price, is date, and is numerical
def get_string_features(line):
    # check line for containing price pattern
    search_obj = re.search(r'\d\.\d\d', line, re.I | re.M)
    if search_obj:
        is_price = 1
    else:
        is_price = 0

    # check line for containing date pattern
    search_obj = re.search(r'\d([\/.\-]\d\d){2}|\D{3}(\, {0,1}| |\.)\d\d(\, {0,1}| |\.)\d\d|\D{3}.*\d{2}\'\d{2}|\d\d[\/\- ]\D{3}[\/\- ]\d\d|\d\d\D{3}\'\d\d', line, re.I | re.M)
    if search_obj:
        is_date = 1
    else:
        is_date = 0

    # check line for containing numeric values
    search_obj = re.search(r'\d', line, re.I | re.M)
    if search_obj:
        is_numeric = 1
    else:
        is_numeric = 0
    
    return is_price, is_date, is_numeric


def group_neighbors(point, midpoints, width, height):
    """
    Finds all the points in a given direction to the point
    
    Parameters
    ----------
    point : int
        the index to the target point in midpoints
    midpoints : list [int, int]
        the list of midpoints of each textbox in the receipt image
    width : int
        the width of the receipt image
    height : int
        the height of the receipt image
        
    Returns
    ----------
    top_neighbors : list int
        the indices of all the neighbors in the top direction of point
    bottom_neighbors : list int
        the indices of all hte bottom neighbors in the bottom direction of point
    right_neighbors : list int
        the indices of all the right neighbors in the right direction of point
    left_neighbors : list int
        the indices of all the left neibhros in the left direction of point
    """
    # the coordinates to the point we're trying to find the neghbors for
    x = point[0]
    y = point[1]
    
    # filter array with constraints
    top_neighbors = []
    bottom_neighbors = []
    left_neighbors = []
    right_neighbors = []
    
    for i in range(0, midpoints.size//2):
        # neighbor point in the form (w,z)
        w = midpoints[i][0]
        z = midpoints[i][1]
        
        # inefficient, will find a way to optimize
        # checks which quadrant a neighbor belongs to (top, bottom, left, right)
        if z < y and w > z*(x/y) and w > z*(-y/width-x):
            top_neighbors += [i]
        if z > y and w > ((z-height)*x)/(y-height) and w < (z - height + ((y-height)/(x-width))*width)/((y-height)/(x-width)):
            bottom_neighbors += [i]    
        if w < x and z > (y/x)*w and z < (((y-height)*w)/x) + height:
            left_neighbors += [i]
        if w > x and z > ((y*w)/(x-width)) - ((y*width)/(x-width)) and z < (((y-height)/(x-width))*(w-width)) + height:
            right_neighbors += [i]
            
    # Return the filtered array of top neighbors
    return top_neighbors, bottom_neighbors, left_neighbors, right_neighbors

def nearest_neighbor(point, midpoints, neighbors):
    """
    A nearest_neighbor algorithm that finds the closest point from a list of points
    
    Parameters
    ----------
    point : int
        the index of a point in the midpoints list to find the nearest neighbor of
    midpoints : list [int, int]
        the midpoints of each textbox in the receipt image
    neighbors : list [int]
        a list of indices that represent all midpoints in a given direction
        
    Returns
    ----------
    nearest_neighbor : int
        the index in midpoints to the nearest neighbor in the neighbors list
    """
    nearest_neighbor = None
    distance = sys.maxsize
    
    for i in range(0, len(neighbors)):
        dist = np.linalg.norm(point - midpoints[neighbors[i]])
        if dist < distance:
            distance = dist
            nearest_neighbor = neighbors[i]
    
    return nearest_neighbor

def connect_graph(midpoints, textbox_dims, width, height):
    """
    Connects each node in a graph to its nearest neighbor in each direction (top, bottom, left, and right)
    
    Parameters
    ----------
    midpoints : list int
        the midpoints of each textbox
    textbox_dims : list [int, int]
        a list of the dimensions of each textbox
    width : int
        the width of the receipt image (in pixels)
    height : int
        the height of the receipt image (in pixels)
    """
    # lists for edges
    edges_u = []
    edges_v = []
    
    # lists for edge features
    left_distance = []
    right_distance = []
    top_distance = []
    bottom_distance = []
    point = []
    
    for u in range(midpoints.size//2):
        point = midpoints[u]
        
        top_neighbors, bottom_neighbors, left_neighbors, right_neighbors = group_neighbors(midpoints[u], midpoints, width, height)
        
        top = nearest_neighbor(point, midpoints, top_neighbors)
        bottom = nearest_neighbor(point, midpoints, bottom_neighbors)
        left = nearest_neighbor(point, midpoints, left_neighbors)
        right = nearest_neighbor(point, midpoints, right_neighbors)
        
        for v in [top, bottom, left, right]:
            if v is not None:
                edges_u += [u]
                edges_v += [v]
        
        top_distance += [0 if top is None else np.linalg.norm(midpoints[top] - point)]
        bottom_distance += [0 if bottom is None else np.linalg.norm(midpoints[bottom] - point)]
        right_distance += [0 if right is None else np.linalg.norm(midpoints[right] - point)]
        left_distance += [0 if left is None else np.linalg.norm(midpoints[left] - point)]

    return edges_u, edges_v, top_distance, bottom_distance, right_distance, left_distance

def graph_modeling(word_areas, labels, width, height, features):
    """
    Creates a graph from the inormation extracted from a json using the load_normalized_example method.
    
    Parameters
    ----------
    word_areas : dict
        the information extracted from the json file using the load_normalized_example method
        
    Returns
    ----------
    graph : obj
        a dgl graph representing a receipt image
    """
    # compute the parameters from word_areas
    midpoints, textbox_dims, corners = compute_parameters(word_areas)
    
    # connect the graph a return the edges from u to v. Also computed the weight of each edge.
    u, v, dt, db, rd, ld = connect_graph(midpoints, textbox_dims, width, height)
    # get the label list
    y = labels
    
    # Convert lists to torch tensors
    u = th.Tensor(u).to(th.int32)
    v = th.Tensor(v).to(th.int32)
    dt = th.Tensor(dt).to(th.float32)
    bd = th.Tensor(db).to(th.float32)
    rd = th.Tensor(rd).to(th.float32)
    ld = th.Tensor(ld).to(th.float32)
    features = th.Tensor(features).to(th.float32)
    
    midpoints_feature = th.Tensor(midpoints).to(th.int32)
    corners = th.Tensor(corners).to(th.int32)
    
    y = th.Tensor(y).to(th.int32)
    
    # create a graph from the edges computed using connect_graph
    graph = dgl.graph((u,v), num_nodes=midpoints.size//2)
    
    # add node features
    graph.ndata['label'] = y
    graph.ndata['corners'] = corners
    graph.ndata['top_dist'] = dt
    graph.ndata['bottom_dist'] = bd
    graph.ndata['left_dist'] = ld
    graph.ndata['right_dist'] = rd
    graph.ndata['midpoints'] = midpoints_feature
    graph.ndata['other_features'] = features
     
    return graph
    
def normalize_graph(graph):
    """
    Normalizes all the feature in a graph and put the features into a node features called 'feat'
    
    Parameters
    ----------
    graph : object
        a dgl graph object for a receipt graph
    """
    num_nodes = graph.num_nodes()
    
    # Normalize data
    # normalize left neighbor distance
    left_dist = graph.ndata['left_dist']
    left_dist = th.Tensor(left_dist/np.linalg.norm(left_dist))
    
    # normalize right neighbor distance
    right_dist = graph.ndata['right_dist']
    right_dist = th.Tensor(right_dist/np.linalg.norm(right_dist))
        
    # normalize top neighbor distance
    top_dist = graph.ndata['top_dist']
    top_dist = th.Tensor(top_dist/np.linalg.norm(top_dist))
    
    # normalize bottom neighbor distance
    bottom_dist = graph.ndata['bottom_dist']
    bottom_dist = th.Tensor(bottom_dist/np.linalg.norm(bottom_dist))
    
    # get all the other features
    other_features = graph.ndata['other_features'].tolist()
     
    # put all features into one big list
    node_features = [[top_dist[i], bottom_dist[i], left_dist[i], right_dist[i]] + other_features[i] for i in range(num_nodes)]
    
    # put all features into a node feature
    node_features = th.Tensor(node_features).to(th.float32)
    graph.ndata['feat'] = node_features
    
    return graph

def create_graph(json_file):
    """
    Creates a single graph from a single json file
    
    Parameters
    ----------
    json_file : str
        the path to the json file to create a graph from
        
    Returns
    ----------
    graph : obj
        The dgl graph created from the json_file
    ocr_text : list str
        a list of strings containing the ocr_text in order of node index
    """
    word_areas, labels, width, height, ocr_text, features = load_normalized_example(json_file)
    graph = graph_modeling(word_areas, labels, width, height, features)
    graph = normalize_graph(graph)
    graph = dgl.add_self_loop(graph)
    
    return graph, ocr_text
    
class ReceiptDataset(dgl.data.DGLDataset):
    """ 
    Creates a dataset of graphs for receipt images. The raw, input data is the json files created using the ReceiptInvoiceOCR class and 
    json_labeler.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(ReceiptDataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
    
    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        ##############################################
        # process each json in raw dir to a graph
        ##############################################
        word_files = get_normalized_filepaths(self.raw_dir)
        graph_data = []
        
        for word_file in word_files:
            # reads normalized data for one image
            word_areas, labels, width, height, ocr_text = load_normalized_example(word_file)

            # Creates the graph from the data
            graph = graph_modeling(word_areas, labels, width, height, features)
            
            # adds the graph to the graph_data list
            graph_data += [graph]
        
        ##############################################
        # set aside 10% of the graphs for validation 
        ##############################################
        # split the data into training and validation data
        val_start = int(len(graph_data) * 0.92) # the start index of the validation dataset
        random.shuffle(graph_data)
        validation = graph_data[val_start:]
        
        # normalize each graph in the validation set
        for i in range (len(validation)):
            graph = normalize_graph(validation[i])
            graph = dgl.add_self_loop(graph)
            validation[i] = graph
            
        # save the validation set to the dataset
        self.validation = validation
        
        # batch graphs into a big graph
        big_graph = dgl.batch(graph_data[:val_start])
        big_graph = dgl.add_self_loop(big_graph)
        
        ############################################# 
        # split data into testing and training data and save graph
        #############################################
        num_nodes = big_graph.number_of_nodes()
        graph_ids = np.random.permutation(num_nodes)

        # 20% testing data / 80% training data
        test_size = int(num_nodes * 0.2)
        train_size = num_nodes - test_size

        test_mask = th.zeros(num_nodes)
        train_mask = th.zeros(num_nodes)

        for i in range(test_size):
            test_mask[graph_ids[i]] = 1

        for i in range(test_size, num_nodes):
            train_mask[graph_ids[i]] = 1

        test_mask = test_mask.to(th.bool)
        train_mask = train_mask.to(th.bool)

        big_graph.ndata['test_mask'] = test_mask
        big_graph.ndata['train_mask'] = train_mask
        big_graph.ndata['label'] = big_graph.ndata['label'].to(th.long)
        
        # Normalize data
        big_graph = normalize_graph(big_graph)
     
        # save graph to dataset object
        self.graphs = [big_graph]
    
    def __getitem__(self, idx):
        # get one example by index
        pass

    def __len__(self):
        # number of data examples
        pass

    def save(self):
        # save_graphs(self.save_dir + '/graph_data.bin', self.graphs[0])
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass