import numpy as np
import pickle as pkl
import csv, sys
from sklearn.model_selection import train_test_split
from MPNN import Model

argv1 = sys.argv[1]#'1H' '13C'#

data1_path = './graph_'+argv1+'_train.pickle'
data2_path = './graph_'+argv1+'_test.pickle'
save_dict = './'
save_path=save_dict+'model_'+argv1+'.ckpt'

# import data
with open(data1_path,'rb') as f: [DV_trn, DE_trn, DY_trn, DM_trn, Dsmi_trn] = pkl.load(f)
with open(data2_path,'rb') as f: [DV_tst, DE_tst, DY_tst, DM_tst, Dsmi_tst] = pkl.load(f)

# basic hyperparam
n_max=DV_trn.shape[1]
dim_node=DV_trn.shape[2]
dim_edge=DE_trn.shape[3]

# trn data processing
DV_trn = DV_trn.todense()
DE_trn = DE_trn.todense()

if argv1 == '13C': DY_trn = DY_trn.todense()
elif argv1 == '1H':
    def list_to_mean(y):
        vec = np.zeros((n_max, 1))
        for i in range(len(y)):       
            if len(y[i])>0: vec[i] = np.mean(y[i])
        
        return vec
        
    DY_trn = np.array([list_to_mean(y) for y in DY_trn])

DM_trn = DM_trn.todense()

# tst data processing
DV_tst = DV_tst.todense()
DE_tst = DE_tst.todense()
if argv1 == '13C': DY_tst = DY_tst.todense()
DM_tst = DM_tst.todense()

DV_tst = np.pad(DV_tst, ((0, 0), (0, n_max - DV_tst.shape[1]), (0, 0)))
DE_tst = np.pad(DE_tst, ((0, 0), (0, n_max - DE_tst.shape[1]), (0, n_max - DE_tst.shape[2]), (0, 0))) 
if argv1 == '13C': DY_tst = np.pad(DY_tst, ((0, 0), (0, n_max - DY_tst.shape[1]), (0, 0)))   
DM_tst = np.pad(DM_tst, ((0, 0), (0, n_max - DM_tst.shape[1]), (0, 0)))  

#trn/val split
DV_trn, DV_val, DE_trn, DE_val, DY_trn, DY_val, DM_trn, DM_val = train_test_split(DV_trn, DE_trn, DY_trn, DM_trn, test_size=0.05)

#summary stat
print(DV_trn.shape, DE_trn.shape, DY_trn.shape, DM_trn.shape)
print(DV_val.shape, DE_val.shape, DY_val.shape, DM_val.shape)
print(DV_tst.shape, DE_tst.shape, DY_tst.shape, DM_tst.shape)

# model
model = Model(n_max, dim_node, dim_edge)

# trainining
with model.sess:
    model.train(DV_trn, DE_trn, DY_trn, DM_trn, DV_val, DE_val, DY_val, DM_val, save_path)
    
    print(':: MAE on test set', model.test_mae(DV_tst, DE_tst, DY_tst, DM_tst, 30))