term<-read.csv("D:\\验证（乳腺癌亚型）\\数据（更新）\\训\\metabric_term_1551.csv", header = TRUE, stringsAsFactors = FALSE)
load("D:\\验证（乳腺癌亚型）\\数据（更新）\\训\\label.RData")
term<-term[,order(colnames(term))]
# 加载必要的库
set.seed(202)  # 更改
# 1. 按 8:2 的比例随机分割样本
n_samples <- ncol(term)
indices <- sample(1:n_samples, n_samples)  # 随机打乱样本顺序

train_indices <- indices[1:round(0.8 * n_samples)]  # 训练集索引
test_indices <- indices[(round(0.8 * n_samples) + 1):n_samples]  # 测试集索引

train_data <- term[, train_indices]  # 训练集
test_data <- term[, test_indices]  # 测试集

train_labels <- label[train_indices,]
test_labels <- label[test_indices,]

# 2. 将特征分为前 343 行和后面的行
# 确保特征行数大于 343
if (nrow(term) >= 343) {
  train_data_first_half <- train_data[1:343, ]
  train_data_second_half <- train_data[-(1:343), ]
  
  test_data_first_half <- test_data[1:343, ]
  test_data_second_half <- test_data[-(1:343), ]
} else {
  stop("特征行数少于343行")
}
train_data_first_half<-as.data.frame(t(train_data_first_half))
write.csv(train_data_first_half, "D:\\MOGONET-main\\5\\1_tr.csv", row.names = FALSE, col.names = FALSE)  # 替换为您想要保存的路径
train_data_second_half<-as.data.frame(t(train_data_second_half))
write.csv(train_data_second_half, "D:\\MOGONET-main\\5\\2_tr.csv", row.names = FALSE, col.names = FALSE)  # 替换为您想要保存的路径
test_data_first_half<-as.data.frame(t(test_data_first_half))
write.csv(test_data_first_half, "D:\\MOGONET-main\\5\\1_te.csv", row.names = FALSE, col.names = FALSE)  # 替换为您想要保存的路径
test_data_second_half<-as.data.frame(t(test_data_second_half))
write.csv(train_data_second_half, "D:\\MOGONET-main\\5\\2_te.csv", row.names = FALSE, col.names = FALSE)  # 替换为您想要保存的路径
write.csv(train_labels, "D:\\MOGONET-main\\5\\label_train.csv", row.names = FALSE)  # 替换为您想要保存的路径
write.csv(test_labels, "D:\\MOGONET-main\\5\\label_test.csv", row.names = FALSE)  # 替换为您想要保存的路径
write.csv(train_labels$CLAUDIN_SUBTYPE, "D:\\MOGONET-main\\5\\labels_tr.csv", row.names = FALSE)  # 替换为您想要保存的路径
write.csv(test_labels$CLAUDIN_SUBTYPE, "D:\\MOGONET-main\\5\\labels_te.csv", row.names = FALSE)  # 替换为您想要保存的路径


###外部测试集
term_e<-read.csv("D:\\验证（乳腺癌亚型）\\数据（更新）\\验\\BRCA_term_1551.csv", header = TRUE, stringsAsFactors = FALSE)
load("D:\\验证（乳腺癌亚型）\\数据（更新）\\验\\label_wai.RData")
first_half <- term_e[1:343, ]
second_half <- term_e[-(1:343), ]
first_half<-as.data.frame(t(first_half))
write.csv(first_half, "D:\\MOGONET-main\\外部验证\\1_ext.csv", row.names = FALSE, col.names = FALSE)  # 替换为您想要保存的路径
second_half<-as.data.frame(t(second_half))
write.csv(second_half, "D:\\MOGONET-main\\外部验证\\2_ext.csv", row.names = FALSE, col.names = FALSE)  # 替换为您想要保存的路径
write.csv(label, "D:\\MOGONET-main\\外部验证\\labels.csv", row.names = FALSE)  # 替换为您想要保存的路径
write.csv(label$PAM50Call_RNAseq, "D:\\MOGONET-main\\外部验证\\labels_ext.csv", row.names = FALSE)  # 替换为您想要保存的路径











