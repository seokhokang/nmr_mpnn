import numpy as np
import pickle as pkl
import os, sys, sparse
from util import atomFeatures, bondFeatures
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures

argv1 = sys.argv[1]#'1H' '13C'#
argv2 = sys.argv[2]#'train' 'test'

suppl = pkl.load(open('data_'+argv1+'.pickle','rb'))
mol_trn = suppl[argv2+'_df']

molsuppl = mol_trn['rdmol'].to_list()
molprops = mol_trn['value'].to_list()

n_max=64
dim_node=30
dim_edge=10
atom_list=['H','C','N','O','F','P','S','Cl']

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')
        
fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

current_max = 0

DV = []
DE = []
DY = []
DM = []
Dsmi = []
for i, mol in enumerate(molsuppl):

    try:
        Chem.SanitizeMol(mol)
    except:
        continue
    
    if '.' in Chem.MolToSmiles(mol):
        continue
        
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistry(mol)   
    
    n_atom = mol.GetNumAtoms()
    
    rings = mol.GetRingInfo().AtomRings() 
    
    feats = chem_feature_factory.GetFeaturesForMol(mol)
    donor_list = []
    acceptor_list = []
    for j in range(len(feats)):
        if feats[j].GetFamily() == 'Donor':
            assert len(feats[j].GetAtomIds())==1
            donor_list.append (feats[j].GetAtomIds()[0])
        elif feats[j].GetFamily() == 'Acceptor':
            assert len(feats[j].GetAtomIds())==1
            acceptor_list.append (feats[j].GetAtomIds()[0])
    
    # node DV
    node = np.zeros((n_max, dim_node), dtype=np.int8)
    for j in range(n_atom):
        node[j, :] = atomFeatures(j, mol, rings, atom_list, donor_list, acceptor_list)
    
    # edge DE
    edge = np.zeros((n_max, n_max, dim_edge), dtype=np.int8)
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            edge[j, k, :] = bondFeatures(j, k, mol, rings)
            edge[k, j, :] = edge[j, k, :]

    # property DY and mask DM
    props = molprops[i][0]
    mask = np.zeros((n_max, 1), dtype=np.int8)
    
    property = []
    
    if argv1 == '13C':
        for j in range(n_atom):
            atom_property = []
            if j in props:
                atom_property.append(props[j])
                mask[j] = 1
                assert mol.GetAtomWithIdx(j).GetAtomicNum()==6
            
            property.append(atom_property)
                
    elif argv1 == '1H':
        for j in range(n_atom):
            neighbors_property = []
            if mol.GetAtomWithIdx(j).GetAtomicNum() != 1:
                neighbors_id = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors()]
                for k in neighbors_id:
                    if k in props:
                        neighbors_property.append(props[k])
                        mask[j] = 1
                        assert mol.GetAtomWithIdx(k).GetAtomicNum()==1
            
            property.append(neighbors_property)   

    property = np.array(property)

    # compression
    del_ids = np.where(node[:,0]==1)[0]

    node = np.delete(node, del_ids, 0)
    node = np.delete(node, [0], 1)
    edge = np.delete(edge, del_ids, 0)
    edge = np.delete(edge, del_ids, 1)
    property = np.delete(property, del_ids, 0)
    mask = np.delete(mask, del_ids, 0)

    if current_max < mol.GetNumHeavyAtoms():
        current_max = mol.GetNumHeavyAtoms()
        
    node = np.pad(node, ((0, n_max - node.shape[0]), (0, 0)))
    edge = np.pad(edge, ((0, n_max - edge.shape[0]), (0, n_max - edge.shape[1]), (0, 0))) 
    mask = np.pad(mask, ((0, n_max - mask.shape[0]), (0, 0)))

    # append
    DV.append(np.array(node))
    DE.append(np.array(edge))
    DY.append(np.array(property))
    DM.append(np.array(mask))
    Dsmi.append(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol))))
    
    if i % 1000 == 0:
        print(i, current_max, flush=True)

# np array    
DV = np.asarray(DV, dtype=np.int8)
DE = np.asarray(DE, dtype=np.int8)
DY = np.asarray(DY)
DM = np.asarray(DM, dtype=np.int8)
Dsmi = np.asarray(Dsmi)

DV = DV[:,:current_max,:]
DE = DE[:,:current_max,:current_max,:]
DM = DM[:,:current_max,:]

print(DV.shape, DE.shape, DY.shape, DM.shape)

# compression
DV = sparse.COO.from_numpy(DV)
DE = sparse.COO.from_numpy(DE)
DM = sparse.COO.from_numpy(DM)

# save
with open('graph_'+argv1+'_'+argv2+'.pickle','wb') as fw:
    pkl.dump([DV, DE, DY, DM, Dsmi], fw)