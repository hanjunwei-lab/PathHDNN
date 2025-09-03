library(pROC)
library(survival)
library(survminer)
library(rms)
library(Hmisc)
library(mltools)
library(MLmetrics)
library(pROC)
library(survival)
library(survminer)
setwd("")
#144
output_144<-read.csv("./data/pre-trained model/output_144.csv",header = T,row.names = 1)
sample<-read.table("../data/SKCM144/data_clinical_patient.txt",sep = "\t",header = T,row.names = 1)
sample_data<-as.data.frame(cbind(response=sample$BR,sample[,c(6,14,1,7,10,12,15,28)]))
sample_data$response<-ifelse(sample_data$response=="Progressive Disease"|sample_data$response=="Stable Disease",0,1)
sample_data$OS_STATUS<-ifelse(sample_data$OS_STATUS=="0:LIVING",0,1)
inter<-intersect(rownames(output_144),rownames(sample_data))
sample_data<-sample_data[inter,]
output_144<-output_144[inter,]
sur_data<-as.data.frame(cbind(sample_data,pre=output_144$X1))
as.numeric(coords(roc(sur_data$OS_STATUS,sur_data$pre), "best")[1])
sur_data$group<-ifelse(sur_data$pre>median(sur_data$pre),1,0)
auc_144<-roc(sur_data$response,sur_data$pre)$auc
cIndex_144<-rcorr.cens(sur_data$pre,sur_data$response)[1]
mccz_144<-mcc(sur_data$group,sur_data$response)
F1_144<-F1_Score(sur_data$group,sur_data$response)
cut_off_144<-median(sur_data$pre)
time_points <- c(12,24,36,48,60) 
library(timeROC)
roc_results <- timeROC(
  T = sur_data$OS_MONTHS,
  delta = sur_data$OS_STATUS,
  marker = -sur_data$pre,
  cause = 1,  # 事件状态
  times = time_points,
  iid = TRUE
)
roc_results$AUC
sur_roc_144<-roc_results$AUC
win.graph(width = 60,height = 60)
ROC<-roc_results
plot(ROC,time=12,col="#F2E9EB")        
plot(ROC,time=24,add=TRUE,col="#DD8EA4") 
plot(ROC,time=36,add=TRUE,col="#FEDD89") 
plot(ROC,time=48,add=TRUE,col="#31383F") 
legend("bottomright",c(paste0("Y-1: ",round(ROC$AUC[1],3)),paste0("Y-2: ",round(ROC$AUC[2],3)),paste0("Y-3: ",round(ROC$AUC[3],3)),paste0("Y-4: ",round(ROC$AUC[4],3))),col=c("#F2E9EB","#DD8EA4","#FEDD89","#31383F"),lty=1,lwd=2)

sur_data$group<-ifelse(sur_data$pre>median(sur_data$pre),"pre_response","pre_nonresponse")
fit <- survfit(Surv(OS_MONTHS, OS_STATUS) ~ group, data = sur_data)
ggsurvplot(fit,
           pval = TRUE,
           risk.table = TRUE, 
           surv.median.line = "hv",
           ggtheme = theme_test(),
           palette = c("#E7B800", "#2E9FDF"))

a1<-length(which(sur_data$group=="pre_nonresponse" & sur_data$response==1))
a2<-length(which(sur_data$group=="pre_nonresponse" & sur_data$response==0))
a3<-length(which(sur_data$group=="pre_response" & sur_data$response==1))
a4<-length(which(sur_data$group=="pre_response" & sur_data$response==0))
da<-matrix(c(a1,a3,a2,a4),2,2)
rownames(da)<-c("pre_nonresponse","pre_response")
colnames(da)<-c("True","False")
fisher.test(da)
data<-as.data.frame(matrix(0,4,2))
data[1:2,1]<-"pre_nonresponse"
data[3:4,1]<-"pre_response"
data[,2]<-c("CR/PR","PD/SD","CR/PR","PD/SD")
colnames(data)<-c("level","response")
data<-as.data.frame(cbind(data,percent=c(a1/(a1+a2),a2/(a1+a2),a3/(a3+a4),a4/(a3+a4))))
data1<-data[c(1,3),]
ggplot(data1,aes(x=level,y=percent))+geom_bar(stat="identity",width=0.4,fill="red")+geom_text(label=paste(round(data1$percent,3)*100,'%',sep = ''),colour = "black")+theme_gray()+theme_test()+ylab("Objective response rate")
#############
############
#30
output_30<-read.csv("./data/pre-trained model/output_30.csv",header = T,row.names = 1)
sample<-read.table("./data/SKCM30/data_clinical_patient.txt",sep = "\t",header = T,row.names = 1)
sample_data<-as.data.frame(cbind(response=sample$DURABLE_CLINICAL_BENEFIT,sample[,c(1,2)]))
sample_data$response<-ifelse(sample_data$response=="PD",0,1)
sample_data$OS_STATUS<-ifelse(sample_data$OS_STATUS=="0:LIVING",0,1)
inter<-intersect(rownames(output_30),rownames(sample_data))
sample_data<-sample_data[inter,]
output_30<-output_30[inter,]
sur_data<-as.data.frame(cbind(sample_data,pre=output_30$X1))
as.numeric(coords(roc(sur_data$OS_STATUS,sur_data$pre), "best")[1])
sur_data$group<-ifelse(sur_data$pre>median(sur_data$pre),1,0)
auc_30<-roc(sur_data$response,sur_data$pre)$auc
cIndex_30<-rcorr.cens(sur_data$pre,sur_data$response)[1]
mccz_30<-mcc(sur_data$group,sur_data$response)
F1_30<-F1_Score(sur_data$group,sur_data$response)
time_points <- c(12, 24,36,48,60) 
roc_results <- timeROC(
  T = sur_data$OS_MONTHS,
  delta = sur_data$OS_STATUS,
  marker = -sur_data$pre,
  cause = 1,  # 事件状态
  times = time_points,weighting = "cox"
)
roc_results$AUC
sur_roc_30<-roc_results$AUC
sur_data$group<-ifelse(sur_data$pre>median(sur_data$pre),"pre_response","pre_nonresponse")
fit <- survfit(Surv(OS_MONTHS, OS_STATUS) ~ group, data = sur_data)
ggsurvplot(fit,
           pval = TRUE,
           risk.table = TRUE, 
           surv.median.line = "hv",
           ggtheme = theme_test(),
           palette = c("#E7B800", "#2E9FDF"))

a1<-length(which(sur_data$group=="pre_nonresponse" & sur_data$response==1))
a2<-length(which(sur_data$group=="pre_nonresponse" & sur_data$response==0))
a3<-length(which(sur_data$group=="pre_response" & sur_data$response==1))
a4<-length(which(sur_data$group=="pre_response" & sur_data$response==0))
da<-matrix(c(a1,a3,a2,a4),2,2)
rownames(da)<-c("pre_nonresponse","pre_response")
colnames(da)<-c("True","False")
fisher.test(da)
data<-as.data.frame(matrix(0,4,2))
data[1:2,1]<-"pre_nonresponse"
data[3:4,1]<-"pre_response"
data[,2]<-c("CR/PR","PD/SD","CR/PR","PD/SD")
colnames(data)<-c("level","response")
data<-as.data.frame(cbind(data,percent=c(a1/(a1+a2),a2/(a1+a2),a3/(a3+a4),a4/(a3+a4))))
data1<-data[c(1,3),]
ggplot(data1,aes(x=level,y=percent))+geom_bar(stat="identity",width=0.4,fill="red")+geom_text(label=paste(round(data1$percent,3)*100,'%',sep = ''),colour = "black")+theme_gray()+theme_test()+ylab("Objective response rate")
mydata<-data.frame(AUC=c(auc_144,auc_110,auc_60,auc_30),Cindex=c(cIndex_144,cIndex_110,cIndex_60,cIndex_30),dataset=c("144","110","60",'30'))
library(reshape2)
mydata<-melt(mydata)
ggplot(data = mydata,aes(x=variable,y=value,fill=as.character(dataset)))+geom_bar(stat = "identity")
############
#110
output_110<-read.csv("./data/pre-trained model/output_110.csv",header = T,row.names = 1)
sample<-read.table("../data/SKCM110/data_clinical_patient.txt",sep = "\t",header = T,row.names = 1)
sample_data<-as.data.frame(cbind(response=sample$DURABLE_CLINICAL_BENEFIT,sample[,c(1,2,3,4,10,11)]))
sample_data<-sample_data[-which(sample_data$response=="X"),]
sample_data$response<-ifelse(sample_data$response=="PD"|sample_data$response=="SD",0,1)
sample_data$OS_STATUS<-ifelse(sample_data$OS_STATUS=="0:LIVING",0,1)
inter<-intersect(rownames(output_110),rownames(sample_data))
sample_data<-sample_data[inter,]
output_110<-output_110[inter,]
sur_data<-as.data.frame(cbind(sample_data,pre=output_110$X1))
as.numeric(coords(roc(sur_data$OS_STATUS,sur_data$pre), "best")[1])
sur_data$group<-ifelse(sur_data$pre>median(sur_data$pre),1,0)
auc_110<-roc(sur_data$response,sur_data$pre)$auc
cIndex_110<-rcorr.cens(sur_data$pre,sur_data$response)[1]
mccz_110<-mcc(sur_data$group,sur_data$response)
F1_110<-F1_Score(sur_data$group,sur_data$response)
roc(sur_data$response,sur_data$pre)
rcorr.cens(sur_data$pre,sur_data$response)
time_points <- c(12, 24,36,48,60) 
roc_results <- timeROC(
  T = sur_data$OS_MONTHS,
  delta = sur_data$OS_STATUS,
  marker = -sur_data$pre,
  cause = 1,  # 事件状态
  times = time_points,weighting = "cox"
)
roc_results$AUC
sur_roc_110<-roc_results$AUC
sur_data$group<-ifelse(sur_data$pre>median(sur_data$pre),"pre_response","pre_nonresponse")
fit <- survfit(Surv(OS_MONTHS, OS_STATUS) ~ group, data = sur_data)
ggsurvplot(fit,
           pval = TRUE,
           risk.table = TRUE, 
           surv.median.line = "hv",
           ggtheme = theme_test(),
           palette = c("#E7B800", "#2E9FDF"))

a1<-length(which(sur_data$group=="pre_nonresponse" & sur_data$response==1))
a2<-length(which(sur_data$group=="pre_nonresponse" & sur_data$response==0))
a3<-length(which(sur_data$group=="pre_response" & sur_data$response==1))
a4<-length(which(sur_data$group=="pre_response" & sur_data$response==0))
da<-matrix(c(a1,a3,a2,a4),2,2)
rownames(da)<-c("pre_nonresponse","pre_response")
colnames(da)<-c("True","False")
fisher.test(da)
data<-as.data.frame(matrix(0,4,2))
data[1:2,1]<-"pre_nonresponse"
data[3:4,1]<-"pre_response"
data[,2]<-c("CR/PR","PD/SD","CR/PR","PD/SD")
colnames(data)<-c("level","response")
data<-as.data.frame(cbind(data,percent=c(a1/(a1+a2),a2/(a1+a2),a3/(a3+a4),a4/(a3+a4))))
data1<-data[c(1,3),]
ggplot(data1,aes(x=level,y=percent))+geom_bar(stat="identity",width=0.4,fill="red")+geom_text(label=paste(round(data1$percent,3)*100,'%',sep = ''),colour = "black")+theme_gray()+theme_test()+ylab("Objective response rate")
############
#60
output_60<-read.csv("./data/pre-trained model/output_60.csv",header = T,row.names = 1)
sample<-read.table("./data/SKCM60/data_clinical_patient.txt",sep = "\t",header = T,row.names = 1)
sample_data<-as.data.frame(cbind(response=sample$TREATMENT_RESPONSE,sample[,c(1,2)]))
sample_data<-sample_data[-which(sample_data$response==""),]
sample_data$response<-ifelse(sample_data$response=="nonresponse",0,1)
sample_data$OS_STATUS<-ifelse(sample_data$OS_STATUS=="0:LIVING",0,1)
inter<-intersect(rownames(output_60),rownames(sample_data))
sample_data<-sample_data[inter,]
output_60<-output_60[inter,]
sur_data<-as.data.frame(cbind(sample_data,pre=output_60$X1))
as.numeric(coords(roc(sur_data$response,sur_data$pre), "best")[1])
sur_data$group<-ifelse(sur_data$pre>median(sur_data$pre),1,0)
auc_60<-roc(sur_data$response,sur_data$pre)$auc
cIndex_60<-rcorr.cens(sur_data$pre,sur_data$response)[1]
mccz_60<-mcc(sur_data$group,sur_data$response)
F1_60<-F1_Score(sur_data$group,sur_data$response)
roc(sur_data$response,sur_data$pre)
rcorr.cens(sur_data$pre,sur_data$response)
roc(sur_data$OS_STATUS,sur_data$pre)
rcorr.cens(sur_data$pre,sur_data$OS_STATUS)
time_points <- c(12, 24,36,48,60) 
roc_results <- timeROC(
  T = sur_data$OS_MONTHS,
  delta = sur_data$OS_STATUS,
  marker = -sur_data$pre,
  cause = 1,  # 事件状态
  times = time_points,weighting = "cox"
)
roc_results$AUC
sur_roc_60<-roc_results$AUC
sur_data$group<-ifelse(sur_data$pre>median(sur_data$pre),"pre_response","pre_nonresponse")
fit <- survfit(Surv(OS_MONTHS, OS_STATUS) ~ group, data = sur_data)
ggsurvplot(fit,
           pval = TRUE,
           risk.table = TRUE, 
           surv.median.line = "hv",
           ggtheme = theme_test(),
           palette = c("#E7B800", "#2E9FDF"))

a1<-length(which(sur_data$group=="pre_nonresponse" & sur_data$response==1))
a2<-length(which(sur_data$group=="pre_nonresponse" & sur_data$response==0))
a3<-length(which(sur_data$group=="pre_response" & sur_data$response==1))
a4<-length(which(sur_data$group=="pre_response" & sur_data$response==0))
da<-matrix(c(a1,a3,a2,a4),2,2)
rownames(da)<-c("pre_nonresponse","pre_response")
colnames(da)<-c("True","False")
fisher.test(da)
data<-as.data.frame(matrix(0,4,2))
data[1:2,1]<-"pre_nonresponse"
data[3:4,1]<-"pre_response"
data[,2]<-c("CR/PR","PD/SD","CR/PR","PD/SD")
colnames(data)<-c("level","response")
data<-as.data.frame(cbind(data,percent=c(a1/(a1+a2),a2/(a1+a2),a3/(a3+a4),a4/(a3+a4))))
data1<-data[c(1,3),]
ggplot(data1,aes(x=level,y=percent))+geom_bar(stat="identity",width=0.4,fill="red")+geom_text(label=paste(round(data1$percent,3)*100,'%',sep = ''),colour = "black")+theme_gray()+theme_test()+ylab("Objective response rate")

mydata<-data.frame(AUC=c(auc_144,auc_30,auc_110,auc_60),
                   C_index=c(cIndex_144,cIndex_30,cIndex_110,cIndex_60),
                   MCC=c(mccz_144,mccz_30,mccz_110,mccz_60),
                   F1_score=c(F1_144,F1_30,F1_110,F1_60))
mydata<-data.frame(auc_144=sur_roc_144,
                   auc_30=sur_roc_30,
                   auc_110=sur_roc_110,
                   auc_60=sur_roc_60)
mydata[which(is.na(mydata),arr.ind = T)]<-0
pheatmap::pheatmap(mydata,color = colorRampPalette(colors = c("white","red"))(50))
mydata<-melt(mydata,variable.name = "datasets")
mydata<-as.data.frame(cbind(mydata,years=rep(1:5,times=4)))
win.graph(width = 60,height = 60)
ggplot(data = mydata,aes(x = datasets,fill = as.character(years),group=as.character(years),color=as.character(years)))+geom_bar(aes(y = value), stat = "identity", position = 'dodge2')+theme_test()
