#数据预处理
library(clusterProfiler)
library(survival)
library(survminer)
setwd("")
source("get_matrix.R")
#突变数据预处理，筛选至少在三个样本中发生非沉默突变的候选基因
maf<-read.csv("data_mutations_extended.csv")
brca_maf<-maf[,c(1,2,10,11,17)]
brca_maf<-get_freq_matrix(mut = brca_maf,is.TCGA = T)
brca_maf[brca_maf>1]<-1
sample<-read.table("data_clinical_patient.txt",sep = "\t",header = T,row.names = 1)
sample_data<-as.data.frame(cbind(sample=rownames(sample),sample[,1:2]))
sample_data$OS_STATUS<-ifelse(sample_data$OS_STATUS=="0:LIVING",0,1)
rownames(sample_data)<-sample_data$sample
inter<-intersect(colnames(brca_maf),rownames(sample_data))
brca_maf<-brca_maf[,inter]
sample_data<-sample_data[inter,]
brca_maf<-brca_maf[which(apply(brca_maf,1,function(x){length(which(x!=0))})>3),]
cox_data<-as.data.frame(cbind(t(brca_maf),sample_data[colnames(brca_maf),-1]))
colnames(cox_data)<-gsub(pattern = "-",replacement = "\\.",x = colnames(cox_data))
cox_data<-cox_data[,-2093]
colnames(cox_data)[c(4343,4344)]<-c("event","survival")
res<-get_univarCox_result(cox_data)
rownames(res)<-gsub(pattern = "\\.",replacement = "-",x = rownames(res))
brca_maf<-brca_maf[rownames(res)[which(res$p.value<0.05)],]
reactome<-read.gmt("ReactomePathways.gmt")
reactome<-reactome[-which(reactome$gene==""),]
inter_gene<-intersect(unique(reactome$gene),rownames(brca_maf))
b<-c()
for(i in 1:length(inter_gene)){
  a<-which(reactome$gene==inter_gene[i])
  b<-c(b,a)
}
reactome<-reactome[b,]
reactome_data<-as.data.frame(cbind(input=reactome$gene,translation=as.character(reactome$term)))
brca_maf<-brca_maf[inter_gene,]
brca_maf<-as.data.frame(cbind(Protein=rownames(brca_maf),as.data.frame(brca_maf)))
#write.csv(brca_maf,file = "brca_maf.csv")
#write.csv(reactome_data,file = "reactome_data.csv")
#拷贝数（CNA）数据预处理，筛选至少在三个样本中发生CNA的候选基因
cna<-read.table("data_CNA.txt",header = T,stringsAsFactors = F,sep = "\t")
cna<-cna[-which(duplicated(cna$Hugo_Symbol)),]
rownames(cna)<-cna$Hugo_Symbol
cna<-cna[,-c(1,2)]
genes<-rownames(cna)
eg<-bitr(geneID = genes,fromType = "SYMBOL",toType = "ENSEMBL",OrgDb = "org.Hs.eg.db")
eg<-eg[-which(duplicated(eg$SYMBOL)),]
eg<-eg[-which(duplicated(eg$ENSEMBL)),]
rownames(eg)<-eg$SYMBOL
cna<-cna[eg$SYMBOL,]
rownames(cna)<-eg$ENSEMBL
cna<-cna[,intersect(colnames(cna),colnames(brca_maf))]
library(tinyarray)
da<-trans_exp(cna,mrna_only = T)
cna_del<-da
cna_del[cna_del==1|cna_del==2]<-0
cna_del<-abs(cna_del)
cna_amp<-da
cna_amp[cna_amp== -1|cna_amp== -2]<-0
cna_amp<-cna_amp[which(rowSums(cna_amp)>3),]
cna_del<-cna_del[which(rowSums(cna_del)>3),]
cna_del[cna_del>1]<-1
cna_amp[cna_amp>1]<-1
#保留可以映射到Reactome通路中的基因组特征，并生成相对应的映射关系矩阵
#拷贝数扩增
reactome1<-read.gmt("ReactomePathways.gmt")
reactome1<-reactome1[-which(reactome1$gene==""),]
inter_gene<-intersect(unique(reactome1$gene),rownames(cna_amp))
b<-c()
for(i in 1:length(inter_gene)){
  a<-which(reactome1$gene==inter_gene[i])
  b<-c(b,a)
}
reactome1<-reactome1[b,]
reactome_data1<-as.data.frame(cbind(input=reactome1$gene,translation=as.character(reactome1$term)))
cna_amp1<-cna_amp[inter_gene,]
cna1<-as.data.frame(cbind(Protein=rownames(cna_amp1),as.data.frame(cna_amp1)))
#拷贝数删失
reactome2<-read.gmt("ReactomePathways.gmt")
reactome2<-reactome2[-which(reactome2$gene==""),]
inter_gene<-intersect(unique(reactome2$gene),rownames(cna_del))
b<-c()
for(i in 1:length(inter_gene)){
  a<-which(reactome2$gene==inter_gene[i])
  b<-c(b,a)
}
reactome2<-reactome2[b,]
reactome_data2<-as.data.frame(cbind(input=reactome2$gene,translation=as.character(reactome2$term)))
cna_del1<-cna_del[inter_gene,]
cna2<-as.data.frame(cbind(Protein=rownames(cna_del1),as.data.frame(cna_del1)))
#基因突变
brca_maf$Protein<-paste0(brca_maf$Protein,"_mut")
cna1$Protein<-paste0(cna1$Protein,"_amp")
cna2$Protein<-paste0(cna2$Protein,"_del")
reactome_data$input<-paste0(reactome_data$input,"_mut")
reactome_data1$input<-paste0(reactome_data1$input,"_amp")
reactome_data2$input<-paste0(reactome_data2$input,"_del")
input_data<-as.data.frame(rbind(brca_maf,cna1,cna2))
reactome_com<-as.data.frame(rbind(reactome_data,reactome_data1,reactome_data2))

write.csv(input_data,file = "input_data.csv")
write.csv(reactome_com,file = "reactome_mut_del_amp.csv")
write.csv(sample_data,file = "sample_information.csv")
