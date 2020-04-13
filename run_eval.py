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

# tst data processing
DV_tst = DV_tst.todense()
DE_tst = DE_tst.todense()

DV_tst = np.pad(DV_tst, ((0, 0), (0, n_max - DV_tst.shape[1]), (0, 0)))
DE_tst = np.pad(DE_tst, ((0, 0), (0, n_max - DE_tst.shape[1]), (0, n_max - DE_tst.shape[2]), (0, 0)))

#summary stat
print(DV_tst.shape, DE_tst.shape, DY_tst.shape)

# model
model = Model(n_max, dim_node, dim_edge)

# evaluation
with model.sess:
    model.saver.restore(model.sess, save_path)  
    
    print(':: MAE on test set', model.test_mae(DV_tst, DE_tst, DY_tst, 30))
    
    
    DY_tst_hat_list = [model.test(DV_tst, DE_tst) for _ in range(30)]

    DY_tst_hat_mean = np.mean(DY_tst_hat_list, 0)
    DY_tst_hat_std = np.std(DY_tst_hat_list, 0)

    # prediction
    abs_err = []
    std_err = []
    for i, dy in enumerate(DY_tst):
        for j in range(len(dy)):
            if len(dy[j]) > 0:
                abs_err = abs_err + np.abs(dy[j] - DY_tst_hat_mean[i,j]).tolist()
                std_err = std_err + np.repeat(DY_tst_hat_std[i,j], len(dy[j])).tolist()

    assert len(abs_err) == len(std_err)
    abs_err = np.array(abs_err)
    std_err = np.array(std_err)    
    

    # evaluation of prediction performance
    print('-- prediction performance evaluation')
    mae_res = []
    for frac in [1, 0.95, 0.80, 0.50]:
        
        cutoff = np.sort(std_err)[int(np.ceil(len(std_err)*frac) - 1)]
        mae = np.mean(abs_err[std_err <= cutoff])
        mae_res.append(mae)
        
        
    print(':: MAE frac at rate=1, 0.95, 0.80, 0.50 ', mae_res)
    
    
    # evaluation of molecule search
    print('-- molecule search performance evaluation') 
    
    DY_tst_sort = []
    DY_tst_hat_sort = []
    for i, dy in enumerate(DY_tst):
        dy_aggr = []
        dy_hat_aggr = []
        for j in range(len(dy)):
            if len(dy[j]) > 0:
                dy_aggr = dy_aggr + dy[j]
                dy_hat_aggr = dy_hat_aggr + np.repeat(DY_tst_hat_mean[i,j], len(dy[j])).tolist()
        
        assert len(dy_aggr) == len(dy_hat_aggr)
        DY_tst_sort.append(np.sort(dy_aggr))
        DY_tst_hat_sort.append(np.sort(dy_hat_aggr))

    DY_tst_sort = np.array(DY_tst_sort)
    DY_tst_hat_sort = np.array(DY_tst_hat_sort)
    DM_tst_cnt = np.array([len(dy) for dy in DY_tst_sort])

    
    search_res = []
    for i in range(len(DV_tst)):
    
        DY_query = DY_tst_sort[i]
    
        candidate_idx = np.where(DM_tst_cnt == DM_tst_cnt[i])[0]
        Dsmi_candidate = Dsmi_tst[candidate_idx]
        DY_candidate = np.array(DY_tst_hat_sort[candidate_idx])

        diff = [np.mean(np.abs(dy - DY_query)) for dy in DY_candidate]
        arg_diff = np.argsort(diff)
        smi_diff = Dsmi_candidate[arg_diff]
        
        search_res.append([int(Dsmi_tst[i] in smi_diff[:k]) for k in [1, 2, 5, 10]])


    print(':: top-K accuacy at K=1, 2, 5, 10', np.mean(search_res,0))