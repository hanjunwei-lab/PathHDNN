library(reticulate)
# 指定您的 Conda 环境
use_condaenv("myenv", required = TRUE)
# 检查当前使用的 Python 环境
py_config()
library(DeepCC)
library(keras)
library(org.Hs.eg.db)
library(clusterProfiler)
exp<-read.csv("D:\\比较方法\\比较方法\\DeepCC-master（5）（亚型）\\data(li)delete10\\训练\\exp.csv", header = TRUE, stringsAsFactors = FALSE)
rownames(exp)<-exp[,1]
exp<-exp[,-1]
# 设置随机种子
set.seed(333)

# 获取总行数
total_rows <- nrow(exp)

# 计算要删除的行数（10%）
delete_count <- floor(total_rows * 0.1)

# 随机选择要删除的行索引
rows_to_delete <- sample(1:total_rows, delete_count)

# 删除选中的行
exp <- exp[-rows_to_delete, ]
colnames(exp) <- gsub("^X", "", colnames(exp))

fs <- getFunctionalSpectra(exp)
# 确定训练集的大小
train_size <- 0.9 * nrow(fs)
# 随机抽取训练集的索引
train_indices <- sample(seq_len(nrow(fs)), size = train_size)
# 创建训练集和测试集
train_data <- fs[train_indices, ]
test_data <- fs[-train_indices, ]
label<-read.csv("D:\\比较方法\\比较方法\\DeepCC-master（5）（亚型）\\data(li)delete10\\训练\\labels.csv", header = TRUE, stringsAsFactors = FALSE)
label<-label[-rows_to_delete, ]
train_label<-label[train_indices, ]
test_label<-label[-train_indices,]
label<-as.character(train_label$diagnosis)
# train DeepCC model
deepcc_model <- train_DeepCC_model(train_data,label)
# obtain deep features 
df <- get_DeepCC_features(deepcc_model, train_data)
# classify new data set use trained DeepCC model
# for a batch of samples
get_DeepCC_label <- function(DeepCCModel, newData, cutoff = 0, prob_mode = F, prob_raw = F)
{
  res <- predict(DeepCCModel$classifier, newData)
  predicted <- apply(res, 1, function(z){
    if (max(z) >= cutoff){
      which.max(z)
    }
    else {
      NA
    }
  })
  pred <- factor(predicted, levels = seq(length(DeepCCModel$levels)),
                 labels = DeepCCModel$levels)
  if (prob_mode) {
    pred <- data.frame(DeepCC = as.character(pred),
                       Probability = round(apply(res, 1, max), digits =3))
  }
  
  if (prob_mode & prob_raw) {
    pred <- res
  }
  
  pred
}
pred_labels <- get_DeepCC_label(deepcc_model, test_data)
save_DeepCC_model(deepcc_model,"D:\\比较方法\\比较方法\\DeepCC-master（5）（亚型）\\data(li)delete10\\model")
###外部测试
exp<-read.csv("D:\\比较方法\\比较方法\\DeepCC-master（5）（亚型）\\data(li2)\\外部测试\\exp.csv", header = TRUE, stringsAsFactors = FALSE)
rownames(exp)<-exp[,1]
exp<-exp[,-1]
label<-read.csv("D:\\比较方法\\比较方法\\DeepCC-master（5）（亚型）\\data(li2)\\外部测试\\labels.csv", header = TRUE, stringsAsFactors = FALSE)
colnames(exp) <- gsub("^X", "", colnames(exp))

fs <- getFunctionalSpectra(exp)
pred_labels_test <- get_DeepCC_label(deepcc_model, fs)
###结果导出
nei<-cbind(test_label,pred_labels)
wai<-cbind(label,pred_labels_test)
write.csv(nei, file = "D:\\比较方法\\比较方法\\DeepCC-master（5）（亚型）\\data(li)delete10\\nei_pred.csv", row.names = FALSE)
write.csv(wai, file = "D:\\比较方法\\比较方法\\DeepCC-master（5）（亚型）\\data(li)delete10\\wai_pred.csv", row.names = FALSE)

