from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils_community import *
from model import *
import uuid
import pickle
import pandas as pd

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--mod_w', type=float, default=0)
parser.add_argument('--seed', type=int, default=42, help='Random seed.') #Default seed same as GCNII
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--w_att',type=float, default=0.0005, help='Weight decay scalar')
parser.add_argument('--w_fc2',type=float, default=0.0005, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.0005, help='Weight decay layer-1')
parser.add_argument('--lr_fc',type=float, default=0.02, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_att',type=float, default=0.02, help='Learning rate Scalar')
parser.add_argument('--feat_type',type=str, default='all', help='Type of features to be used')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
num_layer = args.layer
feat_type = args.feat_type
layer_norm = bool(int(args.layer_norm))

print("==========================")
print(f"Dataset: {args.data}")
print(f"Dropout:{args.dropout}, layer_norm: {layer_norm}")
print(f"w_att:{args.w_att}, w_fc2:{args.w_fc2}, w_fc1:{args.w_fc1}, lr_fc:{args.lr_fc}, lr_att:{args.lr_att}")

cudaid = "cuda:"+str(args.dev)
#device = torch.device(cudaid)
device = torch.device('cpu')
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

mod_w=args.mod_w
print("MOD_W: ", mod_w)


def train_step(G0, G_train,model,optimizer,labels,list_mat,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat, layer_norm)
    ######################################
    #Aggiunta per calcolare la loss
    
    embedding = model.emb(list_mat, layer_norm)#
    modularity_GNN, out_degree_k_tau, in_degree_k_tau, assortativity_GNN,jacc,avg_cluster = calculate_metrics(G0, G_train, embedding,idx_train)  #modularita, kendall tau, assortativity, jacc
    
    structure_penalty =   mod_w*(modularity_GNN) #+ out_degree_k_tau + in_degree_k_tau + assortativity_GNN + jacc
    ###########################
    
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = ((1-mod_w)*(F.nll_loss(output[idx_train], labels[idx_train].to(device)))) - structure_penalty
    # print('\tTraining', modularity_GNN, loss_train)
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(G0, G_val, model,labels,list_mat,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm)

        ######################################
        #Aggiunta per calcolare la loss
        
        
        embedding = model.emb(list_mat, layer_norm)#[idx_train]
        modularity_GNN, out_degree_k_tau, in_degree_k_tau, assortativity_GNN,jacc, avg_cluster = calculate_metrics(G0, G_val, embedding,idx_val)  #modularita, kendall tau, assortativity, jacc
        
        structure_penalty =   mod_w*( modularity_GNN)# + out_degree_k_tau + in_degree_k_tau + assortativity_GNN + jacc
        
        ###########################

        loss_val =  ((1-mod_w)*(F.nll_loss(output[idx_val], labels[idx_val].to(device)))) - structure_penalty
        # print('\tValidation', modularity_GNN, loss_val)
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(G0, G_test, model,labels,list_mat,idx_test, num_labels):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():

        ######################################
        #Aggiunta per calcolare la loss
        
        embedding = model.emb(list_mat, layer_norm)#[idx_train]
        modularity_GNN, out_degree_k_tau, in_degree_k_tau, assortativity_GNN,jacc, avg_cluster = calculate_metrics(G0, G_test, embedding, idx_test)  #modularita, kendall tau, assortativity, jacc
        
        structure_penalty =   mod_w*( modularity_GNN) #+ out_degree_k_tau + in_degree_k_tau + assortativity_GNN + jacc
        
        ###########################

        
        output = model(list_mat, layer_norm)
        loss_test = ((1-mod_w) * (F.nll_loss(output[idx_test], labels[idx_test].to(device)))) - structure_penalty
        
        # print('\tTest', modularity_GNN, loss_test)
        
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        #print('labels:',labels[idx_test].to(device))
        #print(mask_val)
        avg_prec, avg_rec = average_precision_recall(output[idx_test], labels[idx_test].to(device), num_labels)
        #print(avg_prec, avg_rec)
        return loss_test.item(),acc_test.item(), avg_prec, avg_rec


def train(datastr,splitstr):
    adj, adj_i, features, labels, idx_train, idx_val, idx_test, num_features, num_labels, g = full_load_data(datastr,splitstr)
    features = features.to(device)

    adj = adj.to(device)
    adj_i = adj_i.to(device)
    list_mat = []
    list_mat.append(features)
    no_loop_mat = features
    loop_mat = features

    for ii in range(args.layer):
        no_loop_mat = torch.spmm(adj, no_loop_mat)
        loop_mat = torch.spmm(adj_i, loop_mat)
        list_mat.append(no_loop_mat)
        list_mat.append(loop_mat)

    # Select X and self-looped features 
    if feat_type == "homophily":
        select_idx = [0] + [2*ll for ll in range(1,num_layer+1)]
        list_mat = [list_mat[ll] for ll in select_idx]

    #Select X and no-loop features
    elif feat_type == "heterophily":
        select_idx = [0] + [2*ll-1 for ll in range(1,num_layer+1)]
        list_mat = [list_mat[ll] for ll in select_idx]
        
    #Otherwise all hop features are selected

    model = FSGNN(nfeat=num_features,
                nlayers=len(list_mat),
                nhidden=args.hidden,
                nclass=num_labels,
                dropout=args.dropout).to(device)


    optimizer_sett = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
        {'params': model.att, 'weight_decay': args.w_att, 'lr': args.lr_att},
    ]

    optimizer = optim.Adam(optimizer_sett)

    bad_counter = 0
    best = 999999999

    G0, G_train, G_val, G_test = create_graphs(g, idx_train, idx_val, idx_test)

    for epoch in range(args.epochs):
#         print("EPOCH:", epoch)
        loss_tra,acc_tra = train_step(G0,G_train, model,optimizer,labels,list_mat,idx_train)
        loss_val,acc_val = validate_step(G0, G_val, model,labels,list_mat,idx_val)
        #Uncomment following lines to see loss and accuracy values
        
        '''if(epoch+1)%1 == 0:

            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))'''
                

        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    test_out = test_step(G0, G_test, model,labels,list_mat,idx_test, num_labels)
    acc = test_out[1]
    loss = test_out[0]
    avg_prec = test_out[2]
    avg_rec = test_out[3]
    return acc*100, loss, avg_prec, avg_rec






t_total = time.time()
acc_list = []
loss_list= []
avg_prec_list = []
avg_rec_list = []

#####################

for i in range(5):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    train_out = train(datastr,splitstr)
    accuracy_data = train_out[0]
    acc_list.append(accuracy_data)
    loss_data = train_out[1]
    loss_list.append(loss_data)
    prec_data = train_out[2]
    avg_prec_list.append(prec_data)
    rec_data = train_out[3]
    avg_rec_list.append(rec_data)

    ##print(i,": {:.2f}".format(acc_list[-1]))

print("Train cost: {:.4f}s".format(time.time() - t_total))
#print("Test acc.:{:.2f}".format(np.mean(acc_list)))
print(f"Test accuracy: {round(np.mean(acc_list),3)}, {np.round(np.std(acc_list),2)}")
print(f"Test loss: {round(np.mean(loss_list),3)}, {np.round(np.std(loss_list),2)}")
print(f"Test precision: {np.mean(avg_prec_list)}, {np.round(np.std(avg_prec_list),2)}")
print(f"Test recall: {np.mean(avg_rec_list)}, {np.round(np.std(avg_rec_list),2)}")


