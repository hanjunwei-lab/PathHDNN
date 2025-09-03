library(igraph)
make.knn.graph<-function(D,k){
  # calculate euclidean distances between cells
  dist<-as.matrix(dist(D))
  # make a list of edges to k nearest neighbors for each cell
  edges <- mat.or.vec(0,2)
  for (i in 1:nrow(dist)){
    # find closes neighbours
    matches <- setdiff(order(dist[i,],decreasing = F)[1:(k+1)],i)
    if (length(matches) > k) {
      edges <- rbind(edges,cbind(rep(i,length(matches)),matches))
      #edges <- rbind(edges,cbind(matches,rep(i,length(matches))))
    } else {
      edges <- rbind(edges,cbind(rep(i,k),matches))
      #edges <- rbind(edges,cbind(matches,rep(i,k)))
    }
    # add edges in both directions
    
    #edges <- rbind(edges,cbind(matches,rep(i,k)))  
  }
  # create a graph from the edgelist
  graph <- graph_from_edgelist(edges,directed=F)
  V(graph)$frame.color <- NA
  # make a layout for visualizing in 2D
  set.seed(1)
  g.layout<-layout_with_fr(graph)
  return(list(graph=graph,layout=g.layout))        
}
path<-read.csv("D:\\验证（乳腺癌亚型）\\数据（更新）\\训\\path.csv", header = TRUE, stringsAsFactors = FALSE)
rownames(path)<-path[,1]
path<-path[,-1]
path<-as.data.frame(t(path))
path<-path[order(rownames(path)),]
cell<-read.csv("D:\\验证（乳腺癌亚型）\\数据（更新）\\训\\cell.csv", header = TRUE, stringsAsFactors = FALSE)
rownames(cell)<-cell[,1]
cell<-cell[,-1]
cell<-as.data.frame(t(cell))
cell<-cell[order(rownames(cell)),]
write.csv(path, file = "D:\\path.csv", row.names = TRUE, quote = FALSE)
write.csv(cell, file = "D:\\cell.csv", row.names = TRUE, quote = FALSE)
datMeta<-read.csv("D:\\验证（乳腺癌亚型）\\数据（更新）\\训\\metabric_clinical_4subtype_1551.csv", header = TRUE, stringsAsFactors = FALSE)
colnames(datMeta)<-c("sample_id","diagnosis")
datMeta<-datMeta[order(datMeta$sample_id),]
write.csv(datMeta, file = "D:\\labels.csv", row.names = FALSE, quote = FALSE)
corr_path <- cor(t(path), method = "pearson")
# 调用函数
k <- 15
knn_path <- make.knn.graph(corr_path, k)
g_path <- knn_path$graph
g_path <- simplify(g_path, remove.multiple=TRUE, remove.loops=TRUE)
g_path <- delete_vertices(g_path, degree(g_path)==0)
V(g_path)$name <- datMeta$sample_id#注意名称修改
V(g_path)$class <- as.character(datMeta$diagnosis)
V(g_path)$color <- as.numeric(as.factor(V(g_path)$class))
V(g_path)$vertex.frame.color <- "black"
g_path<-as_long_data_frame(g_path)
write.csv(g_path, file = "D:\\path_graph.csv", row.names = FALSE, quote = FALSE)

# 调用函数
k <- 15
corr_cell <- cor(t(cell), method = "pearson")
knn_cell <- make.knn.graph(corr_cell, k)
g_cell <- knn_cell$graph
g_cell <- simplify(g_cell, remove.multiple=TRUE, remove.loops=TRUE)
g_cell <- delete_vertices(g_cell, degree(g_cell)==0)
V(g_cell)$name <- datMeta$sample_id
V(g_cell)$class <- as.character(datMeta$diagnosis)
V(g_cell)$color <- as.numeric(as.factor(V(g_cell)$class))
V(g_cell)$vertex.frame.color <- "black"
g_cell<-as_long_data_frame(g_cell)
write.csv(g_cell, file = "D:\\cell_graph.csv", row.names = FALSE, quote = FALSE)
library(SNFtool)
library(igraph)
library(data.table)

# 1. 读取网络数据
network1 <- g_path
network2 <- g_cell

# 2. 构建图
create_graph <- function(network_data) {
  relation <- data.frame(from = network_data$from_name, to = network_data$to_name)
  patients <- unique(data.frame(id = c(network_data$from_name, network_data$to_name),
                                class = c(network_data$from_class, network_data$to_class)))
  g_net <- graph_from_data_frame(relation, directed = FALSE, vertices = patients)
  g_net <- simplify(g_net, remove.multiple = TRUE, remove.loops = TRUE)
  return(g_net)
}

g1 <- create_graph(network1)
g2 <- create_graph(network2)

# 3. 提取邻接矩阵
adjacency_matrix1 <- as.matrix(as_adjacency_matrix(g1))
adjacency_matrix2 <- as.matrix(as_adjacency_matrix(g2))

# 4. 计算相似性矩阵
adjacency_graphs <- list(adj1 = adjacency_matrix1, adj2 = adjacency_matrix2)
K = 15  # number of neighbors
T = 10  # Number of Iterations
W = SNF(adjacency_graphs, K, T)
rownames(W)<-rownames(path)
colnames(W)<-rownames(path)
# 6. 生成最终图
# 指定患者 ID 列和特征列
idx <- "sample_id"
trait <- "diagnosis"
sub_mod_list <- c("path", "cell")  # 根据你的模态命名
snf.to.graph <- function(W , datMeta , trait , idx , sub_mod_list) {
  
  g <- make.knn.graph(W , 15)
  
  plot.igraph(g$graph,layout=g$layout, vertex.frame.color='black', vertex.color=as.numeric(as.factor(datMeta[idx,][[trait]])),
              vertex.size=5,vertex.label=NA,main=paste0(sub_mod_list , collapse = '_'))
  
  g <- g$graph
  g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)
  
  # Remove any vertices remaining that have no edges
  g <- delete_vertices(g, degree(g)==0)
  
  # Assign names to the graph vertices 
  V(g)$name <- rownames(datMeta[idx,])
  V(g)$class <- as.character(datMeta[idx,][[trait]])
  V(g)$color <- as.numeric(as.factor(V(g)$class))
  V(g)$vertex.frame.color <- "black"
  
  return(g)
}
final_graph <- snf.to.graph(W, datMeta, trait, idx, sub_mod_list)
g <- make.knn.graph(W , 15)
g <- g$graph
g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)

# Remove any vertices remaining that have no edges
g <- delete.vertices(g, degree(g)==0)

# Assign names to the graph vertices 
V(g)$name <- datMeta$sample_id
V(g)$class <- datMeta$diagnosis
V(g)$color <- as.numeric(as.factor(V(g)$class))
V(g)$vertex.frame.color <- "black"
print(length(V(g)))
g<-as_long_data_frame(g)
write.csv(g, file = "D:\\snf_graph.csv", row.names = FALSE, quote = FALSE)

expr.to.graph<-function(datExpr , datMeta , trait , top_genes , modality){
  
  if (modality %in% c('mRNA' , 'miRNA')) {
    mat <- datExpr[top_genes, ]
  } else {
    mat <- t(datExpr[ , top_genes[[trait]]])
  }
  
  if (modality %in% c('mRNA' , 'miRNA' , 'DNAm' , 'RPPA' , 'CSF')) {
    mat <- mat - rowMeans(mat)
    corr_mat <- cor(mat, method="pearson")
  } else {
    corr_mat <- t(mat)
  }
  
  print(dim(mat))
  g <- make.knn.graph(corr_mat , 15)
  
  plot.igraph(g$graph,layout=g$layout, vertex.frame.color='black', vertex.color=as.factor(datMeta[[trait]]),
              vertex.size=5,vertex.label=NA, vertex.label.cex = 0.3 , main=modality )
  
  g <- g$graph
  g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)
  
  # Remove any vertices remaining that have no edges
  g <- delete.vertices(g, degree(g)==0)
  
  # Assign names to the graph vertices 
  V(g)$name <- rownames(datMeta)
  V(g)$class <- as.character(datMeta[[trait]])
  V(g)$color <- as.numeric(as.factor(V(g)$class))
  V(g)$vertex.frame.color <- "black"
  
  return(as_long_data_frame(g))
}

##结果保存
pre<-read.csv("D:\\b.csv", header =FALSE, stringsAsFactors = FALSE)
te<-read.csv("D:\\a.csv", header = FALSE, stringsAsFactors = FALSE)
te=te$V2+1
pre<-pre[te,]
true<-datMeta[te,]
pre<-cbind(true,pre)
pre<-pre[,-3]
colnames(pre)[3]<-"pred"
write.csv(pre, file = "D:\\predict.csv", row.names = FALSE, quote = FALSE)





###测试
data<-read.csv("D:\\验证（乳腺癌亚型）\\数据（更新）\\验\\BRCA_term_1551.csv", header = TRUE, stringsAsFactors = FALSE)
path_xun<-as.data.frame(t(data[1:343,]))
cell_xun<-as.data.frame(t(data[344:429,]))
path_xun <- path_xun[order(rownames(path_xun)), ]
colnames(path_xun)<-colnames(path)
colnames(cell_xun)<-colnames(cell)
datMeta_xun<-read.csv("D:\\验证（乳腺癌亚型）\\数据（更新）\\验\\BRCA_clinical_4subtype_1551.csv", header = TRUE, stringsAsFactors = FALSE)
corr_path_xun <- cor(t(path_xun), method = "pearson")
# 调用函数
k <- 15
knn_path_xun <- make.knn.graph(corr_path_xun, k)
g_path_xun <- knn_path_xun$graph
g_path_xun<- simplify(g_path_xun, remove.multiple=TRUE, remove.loops=TRUE)
g_path_xun <- delete_vertices(g_path_xun, degree(g_path_xun)==0)
V(g_path_xun)$name <- rownames(path_xun)
V(g_path_xun)$class <- as.character(datMeta_xun$PAM50Call_RNAseq)
V(g_path_xun)$color <- as.numeric(as.factor(V(g_path_xun)$class))
V(g_path_xun)$vertex.frame.color <- "black"
g_path_xun<-as_long_data_frame(g_path_xun)
#write.csv(g_path, file = "D:\\path_graph.csv", row.names = FALSE, quote = FALSE)
corr_cell_xun <- cor(t(cell_xun), method = "pearson")
# 调用函数
k <- 15
knn_cell_xun <- make.knn.graph(corr_cell_xun, k)
g_cell_xun <- knn_cell_xun$graph
g_cell_xun<- simplify(g_cell_xun, remove.multiple=TRUE, remove.loops=TRUE)
g_cell_xun <- delete_vertices(g_cell_xun, degree(g_cell_xun)==0)
V(g_cell_xun)$name <- rownames(cell_xun)
V(g_cell_xun)$class <- as.character(datMeta_xun$PAM50Call_RNAseq)
V(g_cell_xun)$color <- as.numeric(as.factor(V(g_cell_xun)$class))
V(g_cell_xun)$vertex.frame.color <- "black"
g_cell_xun<-as_long_data_frame(g_cell_xun)
# 1. 读取网络数据
network1_xun <- g_path_xun
network2_xun <- g_cell_xun

# 2. 构建图
g1_xun <- create_graph(network1_xun)
g2_xun <- create_graph(network2_xun)

# 3. 提取邻接矩阵
adjacency_matrix1_xun <- as.matrix(as_adjacency_matrix(g1_xun))
adjacency_matrix2_xun <- as.matrix(as_adjacency_matrix(g2_xun))

# 4. 计算相似性矩阵
adjacency_graphs_xun <- list(adj1 = adjacency_matrix1_xun, adj2 = adjacency_matrix2_xun)
K = 15  # number of neighbors
T = 10  # Number of Iterations
W_xun = SNF(adjacency_graphs_xun, K, T)
rownames(W_xun)<-rownames(path_xun)
colnames(W_xun)<-rownames(path_xun)
# 6. 生成最终图
# 指定患者 ID 列和特征列
idx <- 'sampleID'
trait <- 'PAM50Call_RNAseq'
sub_mod_list <- c("path", "cell")  # 根据你的模态命名

final_graph_xun <- snf.to.graph(W_xun, datMeta_xun, trait, idx, sub_mod_list)
g_xun <- make.knn.graph(W_xun , 15)
g_xun <- g_xun$graph
g_xun <- simplify(g_xun, remove.multiple=TRUE, remove.loops=TRUE)

# Remove any vertices remaining that have no edges
g_xun <- delete.vertices(g_xun, degree(g_xun)==0)

# Assign names to the graph vertices 
V(g_xun)$name <- datMeta_xun$sampleID
V(g_xun)$class <- datMeta_xun$PAM50Call_RNAseq
V(g_xun)$color <- as.numeric(as.factor(V(g_xun)$class))
V(g_xun)$vertex.frame.color <- "black"
print(length(V(g_xun)))
g_xun<-as_long_data_frame(g_xun)
colnames(datMeta_xun)<-c("sample_id","diagnosis")
write.csv(datMeta_xun, file = "D:\\labels.csv", row.names = FALSE, quote = FALSE)
write.csv(g_xun, file = "D:\\snf_graph.csv", row.names = FALSE, quote = FALSE)
write.csv(g_path_xun, file = "D:\\path_graph.csv", row.names = FALSE, quote = FALSE)
write.csv(g_cell_xun, file = "D:\\cell_graph.csv", row.names = FALSE, quote = FALSE)
write.csv(path_xun, file = "D:\\path.csv", row.names = TRUE, quote = FALSE)
write.csv(cell_xun, file = "D:\\cell.csv", row.names = TRUE, quote = FALSE)
