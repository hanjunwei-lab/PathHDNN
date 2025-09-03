import torch
from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
from dataset import MyDataSet
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
# TCGA
TCGA_labelDict ={0: '0', 1: '1', 2: '2', 3: '3'}

# ICGC
ICGC_labelDict = {0: '0', 1: '1', 2: '2', 3: '3'}

# GEO_metastatic & GEO_metastatic_primary
GEO_metastatic_labelDict = {0: '0', 1: '1', 2: '2', 3: '3'}
GEO_metastatic_primary_labelDict = GEO_metastatic_labelDict

# GEO_primary
GEO_primary_labelDict = {0: '0', 1: '1', 2: '2', 3: '3'}

TolabelDict = {0: '0', 1: '1', 2: '2', 3: '3'}

def train_one_epoch(now_epoch, all_epoch, model, optimizer, data_loader, num, criterion, device):
    model.train()
    optimizer.zero_grad()
    print("train: {}".format(num))
    accurate = 0
    loss_sum = 0
    running_loss = 0.0
    with tqdm(data_loader, desc=f"Epoch {now_epoch+1}/{all_epoch}", ncols=80) as pbar:
        for x_mRNA, y_label in pbar:
            x_mRNA = x_mRNA.to(device)
            y_label = y_label.to(device)
            pred = model(x_mRNA)
            loss = criterion(pred, y_label)
            _, pred = torch.max(pred, 1)
            accurate += (pred == y_label).sum()
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})
            pbar.update()
    
    return loss_sum/num, accurate/num
 
# @torch.no_grad()
def train_evaluate(model, data_loader, num, criterion, device):
    accurate = 0
    loss_sum = 0
    true_label = []
    pred_label = []
    print("test: {}".format(num))
    model.eval() 
    with torch.no_grad():
        for x_mRNA, y_label in data_loader:
            x_mRNA = x_mRNA.to(device)
            y_label = y_label.to(device)
            pred = model(x_mRNA)
            loss = criterion(pred, y_label)
            _, pred = torch.max(pred, 1)
            true_label.extend(y_label.tolist())
            pred_label.extend(pred.tolist())
            accurate += (pred == y_label).sum()
            loss_sum += loss.item()

    return loss_sum/num, accurate/num, true_label, pred_label

# @torch.no_grad()
def evaluate(model, data_loader, num, criterion, device):
    accurate = 0
    loss_sum = 0
    print("test: {}".format(num))
    model.eval()
    pred_label = []
    with torch.no_grad():
        for x_mRNA, y_label in data_loader:
            print("x_mRNA shape:", x_mRNA.shape)
            print("y_label:", y_label) 
            x_mRNA = x_mRNA.to(device)
            y_label = torch.tensor([int(label) for label in y_label]).to(device)
            pred = model(x_mRNA)
            loss = criterion(pred, y_label)
            _, pred = torch.max(pred, 1)
            pred_label.extend(pred.tolist())
            accurate += sum(pred == y_label)
            loss_sum += loss.item()

    return loss_sum/num, accurate/num, pred_label


def test_TCGA(path_test_data, path_weight, prediction_path, model, batch_size, device):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0)

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [TCGA_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)

    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)

    Y_true = y_test 
    Y_true_label = test_data['label']
    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]

    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)
    #mean_acc = cal_TCGA_acc(prediction_path)
    
    #print('average precision_score', precision_score(Y_pre_label, Y_true_label, average='weighted', zero_division=0))
   #print('average recall_score', recall_score(Y_pre_label, Y_true_label, average='weighted', zero_division=0))
    #print('average f1_score', f1_score(Y_pre_label, Y_true_label, average='weighted'))
    #print('average accuracy', mean_acc)
    #return mean_acc
    
def cal_TCGA_acc(prediction_path):
    predict_data = pd.read_csv(prediction_path)
    predict_num = predict_data.shape[0]
    correct_num = 0
    cancer_num = {'0':0, '1':1, '2':2, '3':3}
    cancer_num_correct ={'0':0, '1':1, '2':2, '3':3}
    multi_label = {'0': ['0'], '1': ['1'], '2': ['2'], '3': ['3'], }
    
    for i in range(predict_num):
        for cancer_kind in cancer_num.keys():
            if predict_data.at[i, 'Y_true_label']==cancer_kind:
                T_label = multi_label[cancer_kind]
                cancer_num[predict_data.at[i, 'Y_true_label']] = cancer_num[predict_data.at[i, 'Y_true_label']] + 1
                if predict_data.at[i, 'Y_pre_label'] in T_label:
                    correct_num = correct_num+1
                    cancer_num_correct[predict_data.at[i, 'Y_true_label']] = cancer_num_correct[predict_data.at[i, 'Y_true_label']]+1

    print("The number of samples")
    print(Counter(predict_data['Y_true_label']))
    cancer_list = list(cancer_num_correct.keys())
    per_cancer_acc = {}
    for c in cancer_list:
        per_cancer_acc[c]=round(cancer_num_correct[c]/cancer_num[c], 4)
    print("The accuracy of each cancer")
    print(per_cancer_acc)

    return round(correct_num/predict_num, 4)

def test_ICGC(path_test_data, path_weight, prediction_path, model, batch_size, device):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0)

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [ICGC_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)
 
    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)

    Y_true = y_test 
    Y_true_label = test_data['label']

    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]
    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)

    print("The number of samples")
    print(Counter(test_data['label']))
    print('precision_score', precision_score(y_test, pre_label, average='weighted', zero_division=0))
    print('recall_score', recall_score(y_test, pre_label, average='weighted', zero_division=0))
    print('f1_score', f1_score(y_test, pre_label, average='weighted'))
    print('test_acc', test_acc.item())
    return round(test_acc.item(), 4)

def test_GEO_m(path_test_data, path_weight, prediction_path, model, batch_size, device):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0)

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [GEO_metastatic_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)

    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)

    Y_true = y_test
    Y_true_label = test_data['label']

    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]

    Y_pre_label_change = []
    find_label_list = ['COADREAD', 'BRCA', 'LIHC', 'PAAD', 'STAD', 'ESCA']
    find_label_dict = {'COADREAD':'CRC', 'BRCA':'Breast', 'LIHC':'Liver', 'PAAD': 'pancrease', 'STAD': 'Gastric', 'ESCA': 'Gastric'}
    for i in Y_pre_label:
        if i in find_label_list:
            i = find_label_dict[i]
        Y_pre_label_change.append(i)
    
    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)
    mean_acc = cal_GEO_m_acc(prediction_path)

    print('precision_score', precision_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('recall_score', recall_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('f1_score', f1_score(Y_pre_label_change, Y_true_label,average='weighted'))
    print('average accuracy', mean_acc)
    return round(mean_acc, 4)

def cal_GEO_m_acc(prediction_path):
    predict_data = pd.read_csv(prediction_path)
    predict_num = predict_data.shape[0]
    correct_num = 0
    cancer_num = {'CRC': 0, 'Breast': 0, 'Liver': 0, 'pancrease': 0, 'Gastric': 0}
    cancer_num_correct = {'CRC': 0, 'Breast': 0, 'Liver': 0, 'pancrease': 0, 'Gastric': 0}

    multi_label = {'CRC': ['COADREAD'],'Breast': ['BRCA'],'Liver': ['LIHC'],'pancrease': ['PAAD'], 'Gastric': ['STAD', 'ESCA']}
    
    for i in range(predict_num):
        for cancer_kind in cancer_num.keys():
            if predict_data.at[i, 'Y_true_label']==cancer_kind:
                T_label = multi_label[cancer_kind]
                cancer_num[predict_data.at[i, 'Y_true_label']] = cancer_num[predict_data.at[i, 'Y_true_label']] + 1
                if predict_data.at[i, 'Y_pre_label'] in T_label:
                    correct_num = correct_num+1
                    cancer_num_correct[predict_data.at[i, 'Y_true_label']] = cancer_num_correct[predict_data.at[i, 'Y_true_label']]+1

    print("The number of samples")
    print(Counter(predict_data['Y_true_label']))
    cancer_list = list(cancer_num_correct.keys())
    per_cancer_acc = {}
    for c in cancer_list:
        per_cancer_acc[c]=round(cancer_num_correct[c]/cancer_num[c], 4)
    print("The accuracy of each cancer")
    print(per_cancer_acc)
    return round(correct_num/predict_num, 4)

def test_GEO_p_m(path_test_data, path_weight, prediction_path, model, batch_size, device):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0)

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [GEO_metastatic_primary_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)
  
    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)

    Y_true = y_test
    Y_true_label = test_data['label']

    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]

    Y_pre_label_change = []
    find_label_list = ['COADREAD', 'SKCM', 'UVM', 'KIRC', 'KIRP', 'KICH', 'CESC', 'OV']
    find_label_dict = {'COADREAD':'CRC', 'SKCM':'Melanoma', 'UVM':'Melanoma', 'KIRC':'kidney', 'KIRP':'kidney', 'KICH':'kidney', 'CESC':'cervical', 'OV':'ovarian'}

    for i in Y_pre_label:
        if i in find_label_list:
            i = find_label_dict[i]
        Y_pre_label_change.append(i)

    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)
    mean_acc = cal_GEO_p_m_acc(prediction_path)

    print('precision_score', precision_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('recall_score', recall_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('f1_score', f1_score(Y_pre_label_change, Y_true_label, average='weighted'))
    print('average accuracy', mean_acc)
    return mean_acc

def cal_GEO_p_m_acc(predicition):
    predict_data = pd.read_csv(predicition)
    predict_num = predict_data.shape[0]
    correct_num = 0
    cancer_num = {'CRC': 0, 'Melanoma': 0, 'kidney': 0, 'cervical': 0, 'ovarian': 0}
    cancer_num_correct = {'CRC': 0, 'Melanoma': 0, 'kidney': 0, 'cervical': 0, 'ovarian': 0}

    multi_label = {'CRC': ['COADREAD'],'Melanoma': ['SKCM', 'UVM'],'kidney': ['KIRC', 'KIRP', 'KICH'],'cervical': ['CESC'], 'ovarian': ['OV']}
    
    for i in range(predict_num):
        for cancer_kind in cancer_num.keys():
            if predict_data.at[i, 'Y_true_label']==cancer_kind:
                T_label = multi_label[cancer_kind]
                cancer_num[predict_data.at[i, 'Y_true_label']] = cancer_num[predict_data.at[i, 'Y_true_label']] + 1
                if predict_data.at[i, 'Y_pre_label'] in T_label:
                    correct_num = correct_num+1
                    cancer_num_correct[predict_data.at[i, 'Y_true_label']] = cancer_num_correct[predict_data.at[i, 'Y_true_label']]+1

    print("The number of samples")
    print(Counter(predict_data['Y_true_label']))
    cancer_list = list(cancer_num_correct.keys())
    per_cancer_acc = {}
    for c in cancer_list:
        per_cancer_acc[c]=round(cancer_num_correct[c]/cancer_num[c],4)
    print("The accuracy of each cancer")
    print(per_cancer_acc)
    return round(correct_num/predict_num, 4)

def test_GEO_p(path_test_data, path_weight, prediction_path, model, batch_size, device):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0)

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [GEO_primary_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)
    
    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)

    Y_true = y_test
    Y_true_label = test_data['label']

    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]

    Y_pre_label_change = []
    find_label_list = ['BRCA', 'CESC', 'PRAD', 'OV', 'THCA', 'KIRC', 'KIRP', 'KICH', 'LUAD', 'LUSC', 'COADREAD']
    find_label_dict = {'BRCA':'breast', 'CESC':'Cervix', 'PRAD':'Prostate', 'OV':'ovary', 'THCA': 'thyroid', 'KIRC':'kidney', 
                       'KIRP':'kidney', 'KICH':'kidney', 'LUAD':'lung', 'LUSC':'lung', 'COADREAD':'CRC'}

    for i in Y_pre_label:
        if i in find_label_list:
            i = find_label_dict[i]
        Y_pre_label_change.append(i)
    
    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)
    mean_acc = cal_GEO_p_acc(prediction_path)

    print('precision_score', precision_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('recall_score', recall_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('f1_score', f1_score(Y_pre_label_change, Y_true_label,  average='weighted'))
    print('average accuracy', mean_acc)
    return mean_acc

def cal_GEO_p_acc(prediction_path):
    predict_data = pd.read_csv(prediction_path)
    predict_num = predict_data.shape[0]
    correct_num = 0
    cancer_num = {'breast': 0, 'Cervix': 0, 'Prostate': 0, 'ovary': 0, 'thyroid': 0, 
    'kidney': 0,  'lung': 0, 'CRC':0}
    cancer_num_correct = {'breast': 0, 'Cervix': 0, 'Prostate': 0, 'ovary': 0, 'thyroid': 0, 
    'kidney': 0, 'lung': 0, 'CRC':0}

    multi_label = {'breast': ['BRCA'], 'Cervix': ['CESC'], 'Prostate': ['PRAD'], 'ovary': ['OV'], 'thyroid': ['THCA'], 
    'kidney': ['KIRC', 'KIRP', 'KICH'], 'lung': ['LUAD', 'LUSC'], 'CRC':['COADREAD']}
    
    for i in range(predict_num):
        for cancer_kind in cancer_num.keys():
            if predict_data.at[i, 'Y_true_label']==cancer_kind:
                T_label = multi_label[cancer_kind]
                cancer_num[predict_data.at[i, 'Y_true_label']] = cancer_num[predict_data.at[i, 'Y_true_label']] + 1
                if predict_data.at[i, 'Y_pre_label'] in T_label:
                    correct_num = correct_num+1
                    cancer_num_correct[predict_data.at[i, 'Y_true_label']] = cancer_num_correct[predict_data.at[i, 'Y_true_label']]+1

    print("The number of samples")
    print(Counter(predict_data['Y_true_label']))
    cancer_list = list(cancer_num_correct.keys())
    per_cancer_acc = {}
    for c in cancer_list:
        per_cancer_acc[c]=round(cancer_num_correct[c]/cancer_num[c], 4)
    print("The accuracy of each cancer")
    print(per_cancer_acc)
    return round(correct_num/predict_num, 4)
