import pandas as pd
import numpy as np
import torch
import os
import pickle
import networkx as nx
from functools import reduce
from dgl.nn import GraphConv
import torch.nn as nn

# Define the merge operation setup
def merge_dfs(left_df, right_df):
    # Merging on 'key' and expanding with 'how=outer' to include all records
    return pd.merge(left_df, right_df, left_index=True, right_index=True, how='outer')









def data_parsing(DATA_PATH, MODALITIES, TARGET_COLS, INDEX_COL, LABEL_FILE):
    datModalities = {}
    
    # 读取标签数据
    label_path = os.path.join(DATA_PATH, LABEL_FILE)
    if os.path.exists(label_path):
        labels = pd.read_csv(label_path)
        labels.set_index(INDEX_COL, inplace=True)
    else:
        raise FileNotFoundError(f"Label file {label_path} not found.")

    # 初始化 meta 数据框，假设 TARGET_COLS 是 ['ostime', 'os']
    datMeta = labels[TARGET_COLS].copy()  # 包含 ostime 和 os 列
    datMeta['diagnosis_label'] = labels[TARGET_COLS[1]].astype(str)  # 这里我们可以使用 os 作为描述标签

    for mod in MODALITIES:
        # 构建文件路径
        file_path = os.path.join(DATA_PATH, f"{mod}.csv")

        # 读取模态数据
        if os.path.exists(file_path):
            datExpr = pd.read_csv(file_path)
            datExpr.set_index(INDEX_COL, inplace=True)

            # 检查行索引是否匹配
            if len(set(datExpr.index.astype(str)) & set(datMeta.index)) == 0:
                datExpr = datExpr.T  # 转置

            # 填充缺失值
            if datExpr.isna().sum().sum() > 0:
                datExpr = datExpr.fillna(datExpr.mean())

            # 存储模态数据
            datModalities[mod] = datExpr.loc[sorted(datExpr.index)]
        else:
            raise FileNotFoundError(f"File {file_path} not found.")

    return datModalities, datMeta.reset_index()




def get_gpu_memory():
    t = torch.cuda.get_device_properties(0).total_memory*(1*10**-9)             
    r = torch.cuda.memory_reserved(0)*(1*10**-9)
    a = torch.cuda.memory_allocated(0)*(1*10**-9)
    
    return print("Total = %1.1fGb \t Reserved = %1.1fGb \t Allocated = %1.1fGb" % (t,r,a))

def indices_removal_adjust(idx_to_swap , all_idx,  new_idx) : 
    update_idx = all_idx[all_idx.isin(new_idx)].reset_index()['index']
    
    update_idx_swap = pd.Series(update_idx.index , index=update_idx.values)
    
    return update_idx_swap[list(set(update_idx) & set(idx_to_swap))].values
    

def network_from_csv(NETWORK_PATH , no_psn , weighted=False ) :
    '''
    Generate a networkx network from a as_long_data_frame() object from igraph in R
    '''
    # Open csv as pandas dataframe
    network = pd.read_csv(NETWORK_PATH , index_col=0)

    # Obtain node names (ids) and numbers
    node_from = network[['from' , 'from_name']]
    node_from.columns = ['node' , 'id']

    node_to = network[['to' , 'to_name']]
    node_to.columns = ['node' , 'id']

    # Create networkx Graph object and add nodes to network 
    G = nx.Graph()
    
    # Add nodes to Graph object, resetting index to begin from 0
    nodes = pd.concat([node_from , node_to]).drop_duplicates().reset_index(drop=True)
    nodes['id'] = [str(i) for i in nodes['id']] # Convert node names to strings
    
    G.add_nodes_from(nodes['id'])
    
    nx.set_node_attributes(G , nodes.reset_index().set_index('id')['index'] , 'idx')

    # Add edges and weights (if applicable to network)
    edges = []
    if weighted == True :
        for edge1 , edge2 , weight  in zip(network['from'] , network['to'] , network['weight'] ) : 
                edges.append((nodes[nodes['node'] == edge1].index[0] ,nodes[nodes['node'] == edge2].index[0] , weight ))

        G.add_weighted_edges_from(edges)
    elif no_psn == True :
         pass
    else :
        for edge1 , edge2  in zip(network['from_name'] , network['to_name'] ) :
                edges.append((nodes[nodes['id'] == edge1]['id'].iloc[0] ,nodes[nodes['id'] == edge2]['id'].iloc[0] ))

        G.add_edges_from(edges)
        
    return G

def init_weights(m):
    if isinstance(m, GraphConv):
        m.reset_parameters() ## or simply use your layer.reset_parameters()
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(1 / m.in_features))
        if m.bias is not None: 
            nn.init.zeros_(m.bias)
