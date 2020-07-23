import numpy as np
import pickle as pkl
import os, sys, sparse, warnings
from util import atomFeatures, bondFeatures
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
import pandas as pds

warnings.filterwarnings('ignore')

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    
data_path = './reaction_data.pkl'
with open(data_path, 'rb') as f:
    [X_list, Y_list] = pkl.load(f)

n_max = 60
atom_list = ['C','O','N','Cl','F','S','Br','Na','P','K','B','I','Pd','Si','H','Li','Cs','Al','Fe','Cu','Mg','Sn','Zn','others']
dim_node = len(atom_list) + 22
dim_edge = 10

DV1 = sparse.COO.from_numpy(np.empty(shape=(0, n_max, dim_node), dtype=np.int8))
DE1 = sparse.COO.from_numpy(np.empty(shape=(0, n_max, n_max, dim_edge), dtype=np.int8)) 
DV2 = sparse.COO.from_numpy(np.empty(shape=(0, n_max, dim_node), dtype=np.int8))
DE2 = sparse.COO.from_numpy(np.empty(shape=(0, n_max, n_max, dim_edge), dtype=np.int8)) 
DV3 = sparse.COO.from_numpy(np.empty(shape=(0, n_max, dim_node), dtype=np.int8))
DE3 = sparse.COO.from_numpy(np.empty(shape=(0, n_max, n_max, dim_edge), dtype=np.int8)) 
DY = []

print(len(X_list))

for xid, X in enumerate(X_list):

    Y = Y_list[xid]
      
    mol1 = Chem.MolFromSmiles(X.split(' ')[0])
    mol2 = Chem.MolFromSmiles(X.split(' ')[1])
    mol3 = Chem.MolFromSmiles(X.split(' ')[2])

    if mol1.GetNumAtoms() > n_max or mol2.GetNumAtoms() > n_max or mol3.GetNumAtoms() > n_max: continue

    rings1 = mol1.GetRingInfo().AtomRings() 
    rings2 = mol2.GetRingInfo().AtomRings() 
    rings3 = mol3.GetRingInfo().AtomRings()      
    
    feats1 = chem_feature_factory.GetFeaturesForMol(mol1)
    donor_list1 = []
    acceptor_list1 = []
    for j in range(len(feats1)):
        if feats1[j].GetFamily() == 'Donor':
            donor_list1.append(feats1[j].GetAtomIds()[0])
        elif feats1[j].GetFamily() == 'Acceptor':
            acceptor_list1.append(feats1[j].GetAtomIds()[0])

    feats2 = chem_feature_factory.GetFeaturesForMol(mol2)
    donor_list2 = []
    acceptor_list2 = []
    for j in range(len(feats2)):
        if feats2[j].GetFamily() == 'Donor':
            donor_list2.append(feats2[j].GetAtomIds()[0])
        elif feats2[j].GetFamily() == 'Acceptor':
            acceptor_list2.append(feats2[j].GetAtomIds()[0])

    feats3 = chem_feature_factory.GetFeaturesForMol(mol3)
    donor_list3 = []
    acceptor_list3 = []
    for j in range(len(feats3)):
        if feats3[j].GetFamily() == 'Donor':
            donor_list3.append(feats3[j].GetAtomIds()[0])
        elif feats3[j].GetFamily() == 'Acceptor':
            acceptor_list3.append(feats3[j].GetAtomIds()[0])

    # node DV
    node1 = np.zeros((n_max, dim_node), dtype=np.int8)
    for j in range(mol1.GetNumAtoms()):
        node1[j, :] = atomFeatures(j, mol1, rings1, atom_list, donor_list1, acceptor_list1)

    node2 = np.zeros((n_max, dim_node), dtype=np.int8)
    for j in range(mol2.GetNumAtoms()):
        node2[j, :] = atomFeatures(j, mol2, rings2, atom_list, donor_list2, acceptor_list3)

    node3 = np.zeros((n_max, dim_node), dtype=np.int8)
    for j in range(mol3.GetNumAtoms()):
        node3[j, :] = atomFeatures(j, mol3, rings3, atom_list, donor_list3, acceptor_list3)

    # edge DE
    edge1 = np.zeros((n_max, n_max, dim_edge), dtype=np.int8)
    for j in range(mol1.GetNumAtoms() - 1):
        for k in range(j + 1, mol1.GetNumAtoms()):
            edge1[j, k, :] = bondFeatures(j, k, mol1, rings1)
            edge1[k, j, :] = edge1[j, k, :]

    edge2 = np.zeros((n_max, n_max, dim_edge), dtype=np.int8)
    for j in range(mol2.GetNumAtoms() - 1):
        for k in range(j + 1, mol2.GetNumAtoms()):
            edge2[j, k, :] = bondFeatures(j, k, mol2, rings2)
            edge2[k, j, :] = edge2[j, k, :]

    edge3 = np.zeros((n_max, n_max, dim_edge), dtype=np.int8)
    for j in range(mol3.GetNumAtoms() - 1):
        for k in range(j + 1, mol3.GetNumAtoms()):
            edge3[j, k, :] = bondFeatures(j, k, mol3, rings3)
            edge3[k, j, :] = edge3[j, k, :]

    # append
    DV1 = np.concatenate([DV1, sparse.COO.from_numpy([node1])], 0)
    DE1 = np.concatenate([DE1, sparse.COO.from_numpy([edge1])], 0)
    DV2 = np.concatenate([DV2, sparse.COO.from_numpy([node2])], 0)
    DE2 = np.concatenate([DE2, sparse.COO.from_numpy([edge2])], 0)
    DV3 = np.concatenate([DV3, sparse.COO.from_numpy([node3])], 0)
    DE3 = np.concatenate([DE3, sparse.COO.from_numpy([edge3])], 0)
    DY.append(Y)

    if xid % 10000 == 0:
        print(xid, len(DV1), X, Y, flush=True)
        #print(xid, Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2), Y, flush=True)

# np array    
DY = np.asarray(DY)

print(DV1.shape, DE1.shape, DV2.shape, DE2.shape, DV3.shape, DE3.shape, DY.shape)

# save
with open('reaction_graph_da.pkl','wb') as fw:
    pkl.dump([DV1, DE1, DV2, DE2, DV3, DE3, DY], fw)
