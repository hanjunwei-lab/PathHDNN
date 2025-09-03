term<-read.csv("D:\\验证（乳腺癌亚型）\\数据（更新）\\验\\BRCA_term_1551.csv", header = TRUE, stringsAsFactors = FALSE)
path<-term[1:343,]
cell<-term[344:429,]
rownames(path)<-pathway_genes$pathname
rownames(cell)<-TMEcell$TMEcells
path<-path[,order(colnames(path))]
cell<-cell[,order(colnames(cell))]
path<-as.data.frame(t(path)) 
cell<-as.data.frame(t(cell)) 
write.csv(path, file = "D:\\path.csv", row.names = FALSE, quote = FALSE)
write.csv(cell, file = "D:\\cell.csv", row.names = FALSE, quote = FALSE)



path<-as.data.frame(t(path))  
#path<-as.data.frame(t(cell)) 
# 加载必要的库
library(igraph)
# 假设您有一个基因表达谱数据框，行是基因，列是样本
# 计算 Pearson 相关性矩阵
similarity_matrix <- cor(path, method = "pearson")
# 创建一个空的边列表
edges <- data.frame(from=character(), to=character(), stringsAsFactors=FALSE)
# 为每个样本选择前 3 个相似样本
for (i in 1:ncol(similarity_matrix)) {
  # 获取当前样本的相似性
  current_sample <- similarity_matrix[, i]
  # 排序并选择前 3 个相似样本（排除自己）
  similar_samples <- sort(current_sample, decreasing = TRUE)[-1][1:3]
  # 创建边
  for (sample in names(similar_samples)) {
    edges <- rbind(edges, data.frame(from = colnames(path)[i], to = sample))
  }
}

# 创建一个新的列，将每一行的 from 和 to 组合成一个有序的字符串
edges$pair <- apply(edges, 1, function(x) {
  paste(sort(x), collapse = ",")
})

# 检查是否有重复的组合
duplicates <- edges[duplicated(edges$pair) | duplicated(edges$pair, fromLast = TRUE), ]

# 输出结果
if (nrow(duplicates) > 0) {
  print("存在相同的行（包括顺序不同的情况）：")
  print(duplicates)
} else {
  print("没有相同的行。")
}
edges_unique <- edges[!duplicated(edges$pair), c("from", "to")]
# 创建样本数字映射
# 假设从 0 开始的连续数字映射
# 创建样本数字映射
sample_names <- colnames(path)
sample_mapping <- setNames(0:(length(sample_names) - 1), sample_names)
# 替换样本
edges_unique$from <- sample_mapping[edges_unique$from]
edges_unique$to <- sample_mapping[edges_unique$to]
# 创建一个与原始矩阵行数相同的列，值都为 1
edge_column <- rep(1, nrow(edges_unique))
# 使用 cbind 添加新列
edges_unique <- cbind(edges_unique, edge = edge_column)
write.csv(edges_unique, file = "D:\\edges_path.csv", row.names = FALSE, quote = FALSE)
#write.csv(edges_unique, file = "D:\\edges_cell.csv", row.names = FALSE, quote = FALSE)
colnames(label)<-c("sample_id","diagnosis")
write.csv(label, file = "D:\\labels.csv", row.names = FALSE, quote = FALSE)




pred<-read.csv("D:\\test_results.csv", header = TRUE, stringsAsFactors = FALSE)
label<-read.csv("D:\\比较方法\\MOGAT-main（测试部分没弄完）\\训练\\labels.csv", header = TRUE, stringsAsFactors = FALSE)
library(dplyr)
label <- label %>%
  mutate(diagnosis = row_number() - 1)
pred<-merge(pred,label,by.x="Test.Index",by.y="diagnosis")
pred<-pred[,-1]
write.csv(pred, file = "D:\\test_results.csv", row.names = FALSE, quote = FALSE)
