from binn import BINN, Network, SuperLogger,BINNExplainer,ImportanceNetwork;
import pandas as pd;
from sklearn import preprocessing;
import numpy as np;
from typing import Union;
import torch;
from pytorch_lightning import Trainer
import torch.nn.functional as F
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import joblib

def fit_data_matrix_to_network_input(data_matrix: pd.DataFrame, features, feature_column="Protein") -> pd.DataFrame:
    nr_features_in_matrix = len(data_matrix.index)
    if len(features) > nr_features_in_matrix:
        features_df = pd.DataFrame(features, columns=[feature_column])
        data_matrix = data_matrix.merge(
            features_df, how='right', on=feature_column)
    if len(features) > 0:
        data_matrix.set_index(feature_column, inplace=True)
        data_matrix = data_matrix.loc[features]
    return data_matrix


def generate_data(data_matrix: pd.DataFrame, design_matrix: pd.DataFrame):
    GroupOneCols = design_matrix[design_matrix['group']
                                 == 0]['sample'].values
    GroupTwoCols = design_matrix[design_matrix['group']
                                 == 1]['sample'].values

    df1 = data_matrix[GroupOneCols].T
    df2 = data_matrix[GroupTwoCols].T
    y = np.array([0 for _ in GroupOneCols] + [1 for _ in GroupTwoCols])
    X = pd.concat([df1, df2]).fillna(0).to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)
    return X, y

input_data = pd.read_csv("./data/SKCM110/pre-processing/brca_maf12.txt",sep="\t");
translation = pd.read_csv("./data/SKCM110/pre-processing/reactome_data12.txt",sep="\t");
pathways = pd.read_csv("./data/pathways.tsv", sep="\t");

network = Network(input_data=input_data,pathways=pathways,mapping=translation,input_data_column = "Protein",source_column = "child",target_column = "parent")

torch.manual_seed(666)
binn = BINN(network=network,n_layers=4,dropout=0.5,validate=True,residual=False,learning_rate=0.001)
design_matrix = pd.read_csv('./data/SKCM110/pre-processing/sample_data12.txt',sep="\t")
protein_matrix = fit_data_matrix_to_network_input(input_data, features=binn.features)
X, y = generate_data(protein_matrix, design_matrix=design_matrix)
dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float64),torch.tensor(y, dtype=torch.float64))
torch.manual_seed(666)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=27, shuffle=True,drop_last=False,num_workers=0)
#validation1
test_da1=pd.read_csv('./data/SKCM144/pre-processing/test_data_12.txt',sep="\t")
test_sam1=pd.read_csv('./data/SKCM144/pre-processing/sample_test_12.txt',sep="\t")
protein_matrix1 = fit_data_matrix_to_network_input(test_da1, features=binn.features)
test_input1, test_group1 = generate_data(protein_matrix1, design_matrix=test_sam1)
a1=tuple([tuple(e) for e in test_input1])
#validation2
test_da2=pd.read_csv('./data/SKCM60/pre-processing/test_data_12.txt',sep="\t")
test_sam2=pd.read_csv('./data/SKCM60/pre-processing/sample_test_12.txt',sep="\t")
protein_matrix2 = fit_data_matrix_to_network_input(test_da2, features=binn.features)
test_input2, test_group2 = generate_data(protein_matrix2, design_matrix=test_sam2)
a2=tuple([tuple(e) for e in test_input2])
#validation3
test_da3=pd.read_csv('./data/SKCM30/pre-processing/test_data_12.txt',sep="\t")
test_sam3=pd.read_csv('./data/SKCM30/pre-processing/sample_test_12.txt',sep="\t")
protein_matrix3 = fit_data_matrix_to_network_input(test_da3, features=binn.features)
test_input3, test_group3 = generate_data(protein_matrix3, design_matrix=test_sam3)
a3=tuple([tuple(e) for e in test_input3])

GroupOneCols = test_sam1[test_sam1['group']== 0]['sample'].values
GroupTwoCols = test_sam1[test_sam1['group']== 1]['sample'].values
a_144=GroupOneCols.tolist()+GroupTwoCols.tolist()

GroupOneCols = test_sam2[test_sam2['group']== 0]['sample'].values
GroupTwoCols = test_sam2[test_sam2['group']== 1]['sample'].values
a_60=GroupOneCols.tolist()+GroupTwoCols.tolist()

GroupOneCols = test_sam3[test_sam3['group']== 0]['sample'].values
GroupTwoCols = test_sam3[test_sam3['group']== 1]['sample'].values
a_30=GroupOneCols.tolist()+GroupTwoCols.tolist()

GroupOneCols = design_matrix[design_matrix['group']== 0]['sample'].values
GroupTwoCols = design_matrix[design_matrix['group']== 1]['sample'].values
a_110=GroupOneCols.tolist()+GroupTwoCols.tolist()

a=tuple([tuple(e) for e in X])
optimizer = binn.configure_optimizers()[0][0]
num_epochs =10000
avg_loss_all = [];avg_auc_all = [];avg_auc_all1 = [];avg_auc_all2 = [];avg_auc_all3 = []
i=0
torch.manual_seed(666)
for epoch in range(num_epochs):
    i=i+1
    if (i%1000) ==0:
        print(i)
    binn.train() 
    total_loss = 0
    total_auc = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(binn.device).type(torch.float32)
        targets = targets.to(binn.device).type(torch.LongTensor)
        optimizer.zero_grad()
        output = binn(inputs).to(binn.device)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        outputs=output[:,1].detach().numpy()
        fpr,tpr,thersholds=roc_curve(targets, outputs)
        roc_auc= auc(fpr,tpr)
        total_auc = total_auc + roc_auc
        
    avg_loss=total_loss/(len(dataloader))
    avg_auc=total_auc/(len(dataloader))

    if i==5973:
    #    joblib.dump(binn,"./data/110_train_model.joblib")
       b=binn.predict_step(torch.tensor(a,dtype=torch.float32),1)
       fpr,tpr,thersholds=roc_curve(y,b[:,1].detach().numpy())
       roc_auc=auc(fpr,tpr)
       avg_auc_all.append(roc_auc)
       b=pd.DataFrame(b.detach().numpy())
       b.index=a_110
    #    b.to_csv('./data/output_110.csv')
       b1=binn.predict_step(torch.tensor(a1,dtype=torch.float32),1)
       fpr1,tpr1,thersholds1=roc_curve(test_group1,b1[:,1].detach().numpy())
       roc_auc1=auc(fpr1,tpr1)
       avg_auc_all1.append(roc_auc1)
       b1=pd.DataFrame(b1.detach().numpy())
       b1.index=a_144
    #    b1.to_csv('./data/output_144.csv')
       b2=binn.predict_step(torch.tensor(a2,dtype=torch.float32),1)
       fpr2,tpr2,thersholds2=roc_curve(test_group2,b2[:,1].detach().numpy())
       roc_auc2=auc(fpr2,tpr2)
       avg_auc_all2.append(roc_auc2)
       b2=pd.DataFrame(b2.detach().numpy())
       b2.index=a_60
    #    b2.to_csv('./data/output_60.csv')
       b3=binn.predict_step(torch.tensor(a3,dtype=torch.float32),1)
       fpr3,tpr3,thersholds3=roc_curve(test_group3,b3[:,1].detach().numpy())
       roc_auc3=auc(fpr3,tpr3)
       avg_auc_all3.append(roc_auc3)
       b3=pd.DataFrame(b3.detach().numpy())
       b3.index=a_30
    #    b3.to_csv('./data/output_30.csv')
       plt.figure(figsize=(5, 5))
       plt.plot(fpr, tpr, lw=2, label='110_train (area = %0.3f)' % roc_auc, linestyle='-') 
       plt.plot(fpr1, tpr1,lw=2, label='144_test (area = %0.3f)' % roc_auc1, linestyle='--')
       plt.plot(fpr2, tpr2,lw=2, label='60_test (area = %0.3f)' % roc_auc2, linestyle='-') 
       plt.plot(fpr3, tpr3,lw=2, label='30_test (area = %0.3f)' % roc_auc3, linestyle='--')

       plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
       plt.xlim([-0.02, 1.05])
       plt.ylim([-0.02, 1.05])
       plt.xlabel('1 - Specificity')
       plt.ylabel('Sensitivity')
       plt.title('ROC')
       plt.legend(loc="lower right")
       plt.savefig("./data/110_train.pdf")
    else:
        b=binn.predict_step(torch.tensor(a,dtype=torch.float32),1)
        fpr,tpr,thersholds=roc_curve(y,b[:,1].detach().numpy())
        roc_auc=auc(fpr,tpr)
        avg_auc_all.append(roc_auc)
        b1=binn.predict_step(torch.tensor(a1,dtype=torch.float32),1)
        fpr1,tpr1,thersholds1=roc_curve(test_group1,b1[:,1].detach().numpy())
        roc_auc1=auc(fpr1,tpr1)
        avg_auc_all1.append(roc_auc1)
        b2=binn.predict_step(torch.tensor(a2,dtype=torch.float32),1)
        fpr2,tpr2,thersholds2=roc_curve(test_group2,b2[:,1].detach().numpy())
        roc_auc2=auc(fpr2,tpr2)
        avg_auc_all2.append(roc_auc2)
        b3=binn.predict_step(torch.tensor(a3,dtype=torch.float32),1)
        fpr3,tpr3,thersholds3=roc_curve(test_group3,b3[:,1].detach().numpy())
        roc_auc3=auc(fpr3,tpr3)
        avg_auc_all3.append(roc_auc3)

avg_auc_all<-np.array(avg_auc_all)
avg_auc_all1<-np.array(avg_auc_all1)
avg_auc_all2<-np.array(avg_auc_all2)
avg_auc_all3<-np.array(avg_auc_all3)

new_list = [x for x in [avg_auc_all, avg_auc_all1,avg_auc_all2,avg_auc_all3]]
np.savetxt("./data/performance_110.txt",new_list,delimiter='\t')
