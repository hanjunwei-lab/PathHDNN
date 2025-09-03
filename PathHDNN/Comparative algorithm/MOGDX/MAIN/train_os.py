import torch
import torch.optim as optim
import torch.nn as nn
import sklearn as sk
from sklearn.metrics import precision_recall_curve , average_precision_score , recall_score ,  PrecisionRecallDisplay
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from dgl.dataloading import DataLoader, NeighborSampler
import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os
from lifelines.utils import concordance_index 
orig_sys_path = sys.path[:]
sys.path.insert(0 , os.path.dirname(os.path.abspath(__file__)))
from preprocess_functions import gen_new_graph
sys.path = orig_sys_path
import gc


def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return(indicator_matrix)

class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, pred, ytime, yevent):
        n_observed = yevent.sum(0)
        ytime_indicator = R_set(ytime)  # Assuming R_set is a predefined function
        if torch.cuda.is_available():
            ytime_indicator = ytime_indicator.cuda()
        
        risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
        diff = pred - torch.log(risk_set_sum)

        # 确保 diff 是二维张量
        if diff.dim() == 1:
            diff = diff.unsqueeze(1)  # 如果 diff 是一维的，添加一个维度

        # 确保 yevent 是二维张量，并转换为 float 类型
        if yevent.dim() == 1:
            yevent = yevent.unsqueeze(1)  # 如果 yevent 是一维的，添加一个维度
        yevent = yevent.float()  # 将 yevent 转换为 float 类型，确保与 diff 相同的类型
        
        sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
        cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
        
        return cost

def c_index(true_T, true_E, pred_risk, include_ties=True):
    """
    Calculate c-index for survival prediction downstream task
    """
    #true_T = true_T.unsqueeze(1)  # 在维度 1 位置添加一个新的维度
    # Ordering true_T, true_E and pred_score in descending order according to true_T
    order = np.argsort(-true_T.detach().cpu().numpy())
    true_T = true_T.detach().cpu().numpy()[order]
    true_E = true_E.detach().cpu().numpy()[order]
    pred_risk = pred_risk.detach().cpu().numpy()[order]

    # Calculating the c-index
    result = concordance_index(true_T, -pred_risk, true_E)

    return result

def train(g, train_index, device ,  model  , epochs , lr , patience, pretrain = False , pnet=False):
    # loss function, optimizer and scheduler
    #loss_fcn = nn.BCEWithLogitsLoss()
    loss_fcn = NegativeLogLikelihood() 
    optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    sampler = NeighborSampler(
        [15 for i in range(len(model.gnnlayers))],  # fanout for each layer
        prefetch_node_feats=['feat'],
        #prefetch_labels=['label'],
    )
    train_dataloader = DataLoader(
        g,
        torch.Tensor(train_index).to(torch.int64).to(device),
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=False,
    )

    best_loss = float('inf')
    consecutive_epochs_without_improvement = 0
    
    train_loss = []

    # training loop
    c_indices = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_acc  = 0
        
        # 在训练循环中
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]  # 输入特征
            #y = blocks[-1].dstdata["label"]  # 目标标签
            y_time = blocks[-1].dstdata["OS.time"]  # 获取时间信息
            y_event = blocks[-1].dstdata["OS"]  # 获取事件信息
            # 将 y 转换为 Long 类型
            #y = y.long()  # 确保 y 的类型为 Long

            logits = model(blocks, x)  # 获取模型预测

            # 计算损失，不需要将 y 转换为 float
            #loss = loss_fcn(logits, y)  # 计算损失
            loss = loss_fcn(logits, y_time, y_event)
            # 计算预测和真实标签
            _, predicted = torch.max(logits, 1)
            #true = y  # 确保 y 是 Long 类型的标签

            # 计算准确率
            #train_acc += (predicted == true).float().mean().item()
            total_loss += loss.item()
                    # 计算 c-index
        # 这里选取 logits 作为风险预测值
            c_index_score = c_index(y_time, y_event, logits)
            c_indices.append(c_index_score)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


        train_loss.append(total_loss/(it+1))
        #train_acc = train_acc/(it+1)
        if (epoch % 5) == 0:
            print(
                "Epoch {:05d} | Loss {:.4f} | C-Index {:.4f} | ".format(
                    epoch, train_loss[-1], c_index_score
                )
            )
        # Check for early stopping
        if train_loss[-1] < best_loss:
            best_loss = train_loss[-1]
            consecutive_epochs_without_improvement = 0
        else:
            consecutive_epochs_without_improvement += 1

        if consecutive_epochs_without_improvement >= patience:
            print(f"Early stopping! No improvement for {patience} consecutive epochs.")
            break

    fig , ax = plt.subplots(figsize=(6,4))
    ax.plot(train_loss  , label = 'Train Loss')
    ax.legend()
    del train_dataloader

 
    return fig 

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, recall_score



def evaluate(model, graph, dataloader):
    model.eval()
    y_time = []
    y_event = []
    logits = []

    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            y_time.append(blocks[-1].dstdata["OS.time"])  # 获取时间信息
            y_event.append(blocks[-1].dstdata["OS"])  # 获取事件信息
            logits.append(model(blocks, x))

    # 合并 logits、y_time 和 y_event
    logits = torch.cat(logits, dim=0)
    y_time = torch.cat(y_time, dim=0)
    y_event = torch.cat(y_event, dim=0)

    # 计算 c-index
    c_index_score = c_index(y_time, y_event, logits)

    # 计算损失，假设 logits 和 y_time, y_event 需要用于计算损失
    loss_fcn = NegativeLogLikelihood()  # 确保使用生存分析的损失函数
    loss = loss_fcn(logits, y_time, y_event)  # 计算损失

    return loss.item(), c_index_score, logits, y_time, y_event


            
def confusion_matrix(logits , targets , display_labels ) : 

    _, predicted = torch.max(logits, 1)

    _, true = torch.max(targets , 1)

    cm = sk.metrics.confusion_matrix(true.cpu().detach().numpy(), predicted.cpu().detach().numpy())

    cmat = sns.heatmap(cm , annot = True , fmt = 'd' , cmap = 'Blues' , xticklabels=display_labels , yticklabels=display_labels , cbar = False)
    
    return cmat

def AUROC(logits, targets , meta) : 

    n_classes = targets.shape[1]
    y_score = targets.cpu().detach().numpy()

    Y_test = nn.functional.softmax(logits , dim = 1).cpu().detach().numpy()
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    average_recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_score[:, i], Y_test[:, i])
        average_precision[i] = average_precision_score(y_score[:, i], Y_test[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_score.ravel(), Y_test.ravel()
    )
    average_precision["micro"] = average_precision_score(y_score, Y_test, average="micro")
    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    fig, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {meta.astype('category').cat.categories[i]}", color=color)

    # add the legend for the iso-f1 curves
    handles, plt_labels = display.ax_.get_legend_handles_labels()
    #handles.extend([l])
    #labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=plt_labels, loc="lower left")
    ax.set_title("Multi-class Precision-Recall curve")
    
    return fig , Y_test

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, graph.ndata['feat'] ,device, batch_size
        )  # pred in buffer_device
        pred = pred[nid].argmax(1)
        label = graph.ndata["label"][nid].to(pred.device).argmax(1)
        
        return sum(pred==label)/len(pred)
    
def tsne_embedding_plot(emb , meta) : 
    tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='random')
    embeddings_2d = tsne.fit_transform(emb)
    
    # Unique labels and colors
    unique_labels = meta.unique()
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    # Map each label to a color
    label_color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Color each point based on its label
    point_colors = [label_color_map[label] for label in meta]

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=point_colors, alpha=0.6, edgecolors='w', s=50)
    plt.title('t-SNE of Model Embeddings Colored by Labels')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_color_map[label], markersize=10) for label in unique_labels]
    plt.legend(handles, unique_labels, title="Labels", 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.clim(-0.5, len(unique_labels) - 0.5)
    