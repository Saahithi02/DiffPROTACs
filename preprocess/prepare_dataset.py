import multiprocessing
import os
import torch
import pandas as pd
from tqdm import tqdm 
import random
from os.path import exists

from const import NUMBER_OF_ATOM_TYPES, ATOM2IDX, IDX2ATOM, CHARGES
from data_utils import smi2sdffile, sdf2nx, get_map_ids_from_nx

# Helper to catch all types of invalid SMILES
def is_invalid_smi(smi):
    return (
        pd.isna(smi) or
        not isinstance(smi, str) or
        smi.strip().lower() in ["", "none", "nan"]
    )

# This version accepts a row dict (not index)
def smi2sdf(row):
    smi_linker = row['linker_canonical']
    smi_protac = row['smiles_canonical']
    protac_id = row['id_protac']

    if is_invalid_smi(smi_linker):
        print(f"Skipping {protac_id} due to invalid linker SMILES")
        return

    if is_invalid_smi(smi_protac):
        print(f"Skipping {protac_id} due to invalid PROTAC SMILES")
        return

    try:
        smi2sdffile(smi_linker, f'data/{protac_id}_linker.sdf')
        smi2sdffile(smi_protac, f'data/{protac_id}_protac.sdf')
    except Exception as e:
        print(f"Error processing {protac_id}: {e}")

if __name__ == "__main__":
    datas = pd.read_csv('protac_linker_smiles.csv')
    datas.columns = datas.columns.str.strip().str.lower()
    datas = datas.drop_duplicates(subset='id_protac', keep='first')


    # generate sdf files using multiprocessing (pass rows not index!)
    rows = datas.to_dict(orient='records')
    pool = multiprocessing.Pool(30)
    pool.map(smi2sdf, rows)
    pool.close()
    pool.join()

    # generate files
    ids = list(set([_.split('_')[0] for _ in os.listdir('data')]))
    train_sets = []
    for id_i in tqdm(ids):
        protac_path = f'data/{id_i}_protac.sdf'
        linker_path = f'data/{id_i}_linker.sdf'

        if not exists(protac_path) or not exists(linker_path):
            print(f"Skipping {id_i} — missing sdf file(s)")
            continue

        try:
            G = sdf2nx(protac_path)
            G_linker = sdf2nx(linker_path)
        except Exception as e:
            print(f"Skipping {id_i} — sdf2nx failed: {e}")
            continue

        maps, anchors = get_map_ids_from_nx(G, G_linker)
        if len(maps) == 1:
            n = len(G.nodes)
            n0 = G.nodes
            n1 = maps[0]  # linker
            n2 = list(set(n0) - set(n1))  # ligand
            positions = []
            one_hot = [] 
            charges = []
            in_anchors = []
            fragment_mask = []
            linker_mask = []

            for ligand_atom in n2:
                positions.append(G.nodes[ligand_atom]['positions'])
                fragment_mask.append(1.)
                linker_mask.append(0.)

                tmp = [0.] * NUMBER_OF_ATOM_TYPES
                tmp[ATOM2IDX[G.nodes[ligand_atom]['element']]] = 1.
                one_hot.append(tmp)
                charges.append(CHARGES[G.nodes[ligand_atom]['element']])
                in_anchors.append(1. if ligand_atom in anchors[0] else 0.)

            for linker_atom in n1:
                positions.append(G.nodes[linker_atom]['positions'])
                fragment_mask.append(0.)
                linker_mask.append(1.)

                tmp = [0.] * NUMBER_OF_ATOM_TYPES
                tmp[ATOM2IDX[G.nodes[linker_atom]['element']]] = 1.
                one_hot.append(tmp)
                charges.append(CHARGES[G.nodes[linker_atom]['element']])

            train_sets.append({
                'uuid': id_i,
                'name': datas['smiles_canonical'][int(id_i) - 1],
                'positions': torch.tensor(positions),
                'one_hot': torch.tensor(one_hot),
                'charges': torch.tensor(charges),
                'anchors': torch.tensor(in_anchors),
                'fragment_mask': torch.tensor(fragment_mask),
                'linker_mask': torch.tensor(linker_mask),
                'num_atoms': n,
            })

    random.shuffle(train_sets)
    train_data = train_sets[-800:]
    val_data = train_sets[-800: -400]
    test_data = train_sets[-400:]
    torch.save(train_data, 'protac_train.pt')
    torch.save(val_data, 'protac_val.pt')
    torch.save(test_data, 'protac_test.pt')
