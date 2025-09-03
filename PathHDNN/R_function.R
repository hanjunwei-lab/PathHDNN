get_freq_matrix<-function(mut,is.TCGA=TRUE){
  data<-as.data.frame(mut)
  if(is.TCGA){
    data[,5]<-substr(gsub(pattern = "\\-",replacement = "\\.",data[,5]),1,16)}
  data_9<-data[which(data[,3]=="Missense_Mutation"|data[,3]=="Frame_Shift_Del"|data[,3]=="Frame_Shift_Ins"|data[,3]=="In_Frame_Del"|data[,3]=="Nonsense_Mutation"|data[,3]=="In_Frame_Ins"|data[,3]=="Splice_Site"|data[,3]=="Nonstop_Mutation"|data[,3]=="Translation_Start_Site"),]
  samples<-unique(data_9[,5])
  genes<-unique(data_9[,1])
  freq_matrix<-matrix(0,length(genes),length(samples))
  rownames(freq_matrix)<-genes
  colnames(freq_matrix)<-samples
  for(i in 1:length(samples)){
    s1<-table(data_9[which(data_9[,5]==samples[i]),1])
    freq_matrix[names(s1),i]<-s1
    }
  if(length(which(rownames(freq_matrix)=="."))>0){
    freq_matrix<-freq_matrix[-which(rownames(freq_matrix)=="."),]}
  return(freq_matrix)
}

get_univarCox_result<-function(DE_path_sur){
  covariates<-colnames(DE_path_sur)[1:(length(DE_path_sur[1,])-2)]
  univ_formulas <- sapply(covariates,function(x) as.formula(paste('Surv(survival, event) ~', x)))
  #univ_formulas <- sapply(covariates,function(x) as.formula(paste('Surv(survival, event) ~',"`",x,"`",sep = "")))
  univ_models <- lapply( univ_formulas, function(x){coxph(x, data =DE_path_sur)})
  # Extract data
  univ_results <- lapply(univ_models,
                         function(x){
                           x <- summary(x)
                           p.value<-signif(x$wald["pvalue"], digits=2)
                           wald.test<-signif(x$wald["test"], digits=2)
                           beta<-signif(x$coef[1], digits=2);#coeficient beta
                           HR <-signif(x$coef[2], digits=2);#exp(beta)
                           HR.confint.lower <- signif(x$conf.int[,"lower .95"], 2)
                           HR.confint.upper <- signif(x$conf.int[,"upper .95"],2)
                           
                           res<-c(beta,HR,HR.confint.lower,HR.confint.upper, p.value)
                           names(res)<-c("beta","HR", "HR.95L", "HR.95H","p.value")
                           return(res)
                           #return(exp(cbind(coef(x),confint(x))))
                         })
  res <- t(as.data.frame(univ_results, check.names = FALSE))
  result<-as.data.frame(res)
  return(result)
}
