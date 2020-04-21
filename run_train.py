import numpy as np
import pickle as pkl
import csv, sys
from MPNN import Model

argv1 = sys.argv[1]#'1H' '13C'#

data1_path = './graph_'+argv1+'_train.pickle'
data2_path = './graph_'+argv1+'_test.pickle'
save_dict = './'
save_path=save_dict+'model_'+argv1+'.ckpt'

# import data
with open(data1_path,'rb') as f: [DV_trn, DE_trn, DY_trn, DM_trn, Dsmi_trn] = pkl.load(f)
with open(data2_path,'rb') as f: [DV_tst, DE_tst, DY_tst, _, Dsmi_tst] = pkl.load(f)

# basic hyperparam
n_max=DV_trn.shape[1]
dim_node=DV_trn.shape[2]
dim_edge=DE_trn.shape[3]

# trn data processing
DV_trn = DV_trn.todense()
DE_trn = DE_trn.todense()
DM_trn = DM_trn.todense()

# tst data processing
DV_tst = DV_tst.todense()
DE_tst = DE_tst.todense()

DV_tst = np.pad(DV_tst, ((0, 0), (0, n_max - DV_tst.shape[1]), (0, 0)))
DE_tst = np.pad(DE_tst, ((0, 0), (0, n_max - DE_tst.shape[1]), (0, n_max - DE_tst.shape[2]), (0, 0))) 

#summary stat
print(DV_trn.shape, DE_trn.shape, DY_trn.shape, DM_trn.shape)
print(DV_tst.shape, DE_tst.shape, DY_tst.shape)

# model
model = Model(n_max, dim_node, dim_edge)

# trainining
with model.sess:
    model.train(DV_trn, DE_trn, DY_trn, DM_trn, save_path)
    
    print(':: MAE on test set', model.test_mae(DV_tst, DE_tst, DY_tst, 30))