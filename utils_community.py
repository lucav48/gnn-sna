import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
from networkx.readwrite import json_graph
import pdb
sys.setrecursionlimit(99999)
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
import networkx.algorithms.community as nx_comm

import pandas as pd

########################################################

#Funzioni aggiunte
#TO DO:
#Sistemare jaccard
#inserire nella loss tutte le metriche
#applicare anche a validation e test
def precision_recall(pred, true, num_classes):
    """
    Calculates precision and recall for multiclass classification.
    
    Arguments:
    pred -- PyTorch tensor of shape (n_samples,) containing the predicted labels.
    true -- PyTorch tensor of shape (n_samples,) containing the true labels.
    num_classes -- integer, the number of classes in the classification problem.
    
    Returns:
    precision -- PyTorch tensor of shape (num_classes,), the precision for each class.
    recall -- PyTorch tensor of shape (num_classes,), the recall for each class.
    """
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    for c in range(num_classes):
        tp[c] = ((pred == c) & (true == c)).sum().item()
        fp[c] = ((pred == c) & (true != c)).sum().item()
        fn[c] = ((pred != c) & (true == c)).sum().item()

    # Calculate precision and recall for each class
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    #print('precision',precision, 'recall',recall)
    return precision, recall


def average_precision_recall(output, labels, num_classes):
    """
    Calculates the average precision and recall for multiclass classification.
    
    Arguments:
    pred -- PyTorch tensor of shape (n_samples,) containing the predicted labels.
    true -- PyTorch tensor of shape (n_samples,) containing the true labels.
    num_classes -- integer, the number of classes in the classification problem.
    
    Returns:
    avg_precision -- float, the average precision over all classes.
    avg_recall -- float, the average recall over all classes.
    """
    pred = output.max(1)[1].type_as(labels)
   
    

    precision_sum = 0
    recall_sum = 0

    for c in range(num_classes):
        precision, recall = precision_recall(pred, labels, num_classes)
        precision_sum += precision[c]
        recall_sum += recall[c]
        #print('label:',c, 'precision',precision,'recall',recall)
    avg_precision = precision_sum / num_classes
    avg_recall = recall_sum / num_classes

    return avg_precision, avg_recall


def create_graphs(g, idx_train, idx_val, idx_test):
    G = nx.Graph(g)
    adj_mat=pd.DataFrame(columns=['target','source'])
    adj_mat = nx.to_pandas_edgelist(G)
    adj_mat = adj_mat[adj_mat['source']!=adj_mat['target']]
    G0= nx.from_pandas_edgelist(adj_mat, 'source', 'target', edge_attr='weight', create_using=nx.DiGraph())
    nodes = list(G0.nodes)
    nodes.sort()
    nodes= torch.tensor(nodes)
    nodes_train =nodes[idx_train].tolist()
    nodes_val = nodes[idx_val].tolist()
    nodes_test = nodes[idx_test].tolist()
    
    G_train = G0.subgraph(nodes = nodes_train)
    G_val = G0.subgraph(nodes = nodes_val)
    G_test = G0.subgraph(nodes = nodes_test)

    return G0, G_train, G_val, G_test



def calculate_metrics(G,G_sub, embedding, idx):
    # trasformo la matrice di diacenza in grafo
    df_emb = pd.DataFrame(embedding.detach().numpy(), columns=['feature_' + str(x) for x in range(embedding.shape[1])])

    sub_nodes = list(G_sub.nodes())
   
    adj_mat = nx.to_pandas_edgelist(G)
    adj_mat['weight'] = adj_mat.apply(lambda x:
                                    cosine_similarity(np.array([df_emb.iloc[x['target']]]),
                                    np.array([df_emb.iloc[x['source']]]))[0][0],
                                    axis=1)

    GNN_sim_all= nx.from_pandas_edgelist(adj_mat, 'source', 'target', edge_attr='weight', 
                             create_using=nx.DiGraph())
    
    GNN_sim = GNN_sim_all.subgraph(nodes = sub_nodes)

    G0 = G.subgraph(nodes = sub_nodes)

    # calcolo le metriche sul grafo
#     partition_G0 = nx_comm.louvain_communities(G0, weight='weight', seed=42)
    partition_GNN = nx_comm.louvain_communities(GNN_sim, weight='weight', seed=42)

    #partition_GNN_all = nx_comm.greedy_modularity_communities(GNN_sim_all, weight='weight') #partizione su tutto il grafp
    
    #modularity_G0 = round(nx.algorithms.community.quality.modularity(G0, partition_G0),3)
    modularity_GNN = round(nx.algorithms.community.quality.modularity(GNN_sim, partition_GNN, weight='weight'),3) # modularità su tutto il grafo

    #per ogni community della prima rete faccio jaccard con ogni community della seconda (da sistemare)
#     overlap = 0
#     n_overlap = 0
#     for c1 in partition_G0:
#         max_12 = 0
#         if len(c1) > 2:
#             for c2 in partition_GNN:
#                 j_12 = jaccard(c1, c2)
#                 if j_12 > max_12:
#                     max_12 = j_12
#             overlap += max_12
#             n_overlap += 1
    max_jaccard = 0#overlap / n_overlap
    
    
    #calcolo assortatività su entrambi i subgraph
    #assortativity_G0 = round(nx.degree_pearson_correlation_coefficient(G0),3)
    assortativity_GNN = 0#round(nx.degree_pearson_correlation_coefficient(GNN_sim, weight='weight'),3)

    #calcolo Kenall's Tau su entrambi i sottografi
    out_degree_k_tau, in_degree_k_tau = 0, 0#all_K_tau(G0, GNN_sim, GNN_dist)
    #average clustering
    
    avg_cluster = 0#nx.average_clustering(GNN_sim, weight = 'weight')
    
    return modularity_GNN, out_degree_k_tau, in_degree_k_tau, assortativity_GNN, max_jaccard, avg_cluster


def weighted_out_degree(G):
  # Calcolo della weighted out-degree centrality
  out_degrees = dict(G.out_degree(weight='weight'))
  max_weight = max(out_degrees.values())
  weighted_out_degrees = {node: (degree / max_weight) for node, degree in out_degrees.items()}

  # Calcolo della somma delle weighted out-degree centralities
  total_weighted_out_degree = sum(weighted_out_degrees.values())

  # Normalizzazione della weighted out-degree centrality
  weighted_out_degree_centrality_normalized = {node: (degree / total_weighted_out_degree) for node, degree in weighted_out_degrees.items()}
  return dict(sorted(weighted_out_degree_centrality_normalized.items(), key=lambda x: x[1], reverse=True))
# Print the result

def weighted_in_degree(G):
  in_degrees = dict(G.in_degree(weight='weight'))
  max_weight = max(in_degrees.values())
  weighted_in_degrees = {node: (degree / max_weight) for node, degree in in_degrees.items()}

  # Calcolo della somma delle weighted in-degree centralities
  total_weighted_in_degree = sum(weighted_in_degrees.values())

  # Normalizzazione della weighted in-degree centrality
  weighted_in_degree_centrality_normalized = {node: (degree / total_weighted_in_degree) for node, degree in weighted_in_degrees.items()}

  return dict(sorted(weighted_in_degree_centrality_normalized.items(), key=lambda x: x[1], reverse=True))


#define Jaccard Similarity function
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return round((float(intersection) / union),3)

def K_tau(dict1,dict2):
    sorted_dict1 = dict(sorted(dict1.items(), key=lambda item: item[1],reverse=True))
    sorted_dict2 = dict(sorted(dict2.items(), key=lambda item: item[1],reverse=True))
    x1 = list(sorted_dict1.keys())[:50]
    x2 = list(sorted_dict2.keys())[:50]
    tau, p_value = stats.kendalltau(x1, x2)
    #j = jaccard(x1,x2)
    return round(tau,3) #,round(p_value,2),round(j,2),



def all_K_tau(G0, G_sim, G_dist):
    out_degree_GNN = weighted_out_degree(G_dist)
    in_degree_GNN = weighted_in_degree(G_dist)
    out_degree = weighted_out_degree(G0)
    in_degree = weighted_in_degree(G0)
    #betweeness_GNN = nx.betweenness_centrality(G_dist, weight = 'weight')
    #betweeness = nx.betweenness_centrality(G0)
    #closeness_GNN = nx.closeness_centrality(G_dist, distance = 'weight')
    #closeness = nx.closeness_centrality(G0)
    #eigenvector_GNN = nx.eigenvector_centrality(G_sim, weight = 'weight', max_iter = 2000)
    #eigenvector = nx.eigenvector_centrality(G0, max_iter = 2000)

    out_degree_k_tau = K_tau(out_degree, out_degree_GNN)
    in_degree_k_tau = K_tau(in_degree, in_degree_GNN)
    #betweeness_k_tau = K_tau(betweeness, betweeness_GNN)
    #closeness_k_tau = K_tau(closeness, closeness_GNN)
    #eigenvector_k_tau = K_tau(eigenvector, eigenvector_GNN)

    return out_degree_k_tau, in_degree_k_tau



#################################################################################################################################################
def accuracy(output, labels, batch=False):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    
    if batch == True:
        return correct
    return correct / len(labels)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   #adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sys_normalized_adjacency_i(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# adapted from tkipf/gcn
def load_citation(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    ##adj = torch.from_numpy(adj.todense()).float()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    n_adj = sys_normalized_adjacency(adj)
    n_adj_i = sys_normalized_adjacency_i(adj)

    n_adj = sparse_mx_to_torch_sparse_tensor(n_adj)
    n_adj_i = sparse_mx_to_torch_sparse_tensor(n_adj_i)
    return n_adj, n_adj_i, features, labels, idx_train, idx_val, idx_test


# adapted from PetarV/GAT
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        #for v in range(nb_nodes):
        for v in adj[u,:].nonzero()[1]:
            #if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)

def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret

def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
              #  if adj[i,j] == 1:
                 return False
    return True

def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits={}
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i]==0 or mapping[j]==0:
                dict_splits[0]=None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'

                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]]='test'

                        else:
                            dict_splits[mapping[i]] = 'train'

                    else:
                        if ds_label[i]['test']:
                            ind_label='test'
                        elif ds_label[i]['val']:
                            ind_label='val'
                        else:
                            ind_label='train'
                        if dict_splits[mapping[i]]!= ind_label:
                            print ('inconsistent labels within a graph exiting!!!')
                            return None
                else:
                    print ('label of both nodes different, exiting!!')
                    return None
    return dict_splits

