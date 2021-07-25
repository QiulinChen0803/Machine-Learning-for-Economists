# 0.1  Information #############################################################
#                                              					
#    ECON 725 Group 11
#             
#    Yarui Zhang, Qiulin Chen, Han Liu, Xin Yang, Yuan Yuan
#              
#    Updated: 2020-12-11    
#                                              					
################################################################################

# 0.2 Load Required Packages and Files
library(formatR)
library(data.table)
library(knitr)
library(glmnet)
library(dplyr)
library(readr)
library(tidyverse)
library(stringr)

no_prom <- read.csv("D:/2020Fall/ECON 725/PROJECT/1/shampoo/finalfinaldataset/no_prom.csv")
b <- read.csv("D:/2020Fall/ECON 725/PROJECT/1/shampoo/finalfinaldataset/b.csv")
c <- read.csv("D:/2020Fall/ECON 725/PROJECT/1/shampoo/finalfinaldataset/c.csv")
s <- read.csv("D:/2020Fall/ECON 725/PROJECT/1/shampoo/finalfinaldataset/s.csv")
g <- read.csv("D:/2020Fall/ECON 725/PROJECT/1/shampoo/finalfinaldataset/g.csv")

# 0.3 Train Model on the group off promotion
no_prom$special_events1[no_prom$special_events ==''] <-0
no_prom$special_events1[no_prom$special_events !=''] <-1

h2o.init(max_mem_size = '16g')


y_n <- "profit"
x_n <- c('zone','quarter','size_1','binded','discounted',
         'combo_shop','share','special_events1')

set.seed(0)
test <- sample.int(nrow(no_prom),0.2*nrow(no_prom),replace = F)
no_prom_train <- no_prom[-test,]
no_prom_test <- no_prom[test,]

## 0.3.1 simple linear

f1 <- lm(profit~ zone+quarter+
           special_events1+size_1+binded+travel_size+discounted+
           combo_shop+share,
         data = no_prom_train)
f1.mse = mean((predict(f1,no_prom_test)-no_prom_test$profit)^2)


## Lasso

cvg_lasso_lambda <- cv.glmnet(as.matrix(no_prom_train[,x_n]),no_prom_train$profit, 
                              type.measure = "mse",nfolds = 8, alpha = 1)$lambda.min
f2 <- glmnet(as.matrix(no_prom_train[,x_n]),no_prom_train$profit, 
             type.measure = "mse",nfolds = 8, alpha = 1,lambda=cvg_lasso_lambda)
f2.mse <- mean((predict(f2,as.matrix(no_prom_test[,x_n]),s=cvg_lasso_lambda) - no_prom_test$profit)^2)

## RandomForest
no_prom_train.h2o <- as.h2o(no_prom_train)
no_prom_test.h2o <- as.h2o(no_prom_test)

f3<-h2o.randomForest(x_n,y_n,no_prom_train.h2o,mtries = 4,ntrees=500, seed=1122)

f3.perf<- h2o.performance(f3,no_prom_test.h2o)
f3.mse <- f3.perf@metrics[["MSE"]]
f3.importance <- h2o.varimp(f3)


## neural network
f4 <- h2o.deeplearning(x_n, y_n, no_prom_train.h2o,epochs = 50,hidden = c(64,32,16))
f4.perf <- h2o.performance(f4,no_prom_test.h2o)
f4.mse<-f6.perf@metrics[["MSE"]]
f4.importacne <- h2o.varimp(f4)

# 0.4 get the predicted profit in group on promotion

#### data cleaning
b$special_events1[b$special_events ==''] <-0
b$special_events1[b$special_events !=''] <-1
c$special_events1[c$special_events ==''] <-0
c$special_events1[c$special_events !=''] <-1
s$special_events1[s$special_events ==''] <-0
s$special_events1[s$special_events !=''] <-1
g$special_events1[g$special_events ==''] <-0
g$special_events1[g$special_events !=''] <-1

b.h2o <- as.h2o(b)
c.h2o <- as.h2o(c)
s.h2o <- as.h2o(s)
g.h2o <- as.h2o(g)

### get the predict value of b group
predict.h2o <- h2o.predict(f3,b.h2o)
predict <- as.data.frame(predict.h2o)
b$predict <- predict$predict
b$resi <- b$profit - b$predict  ### the treatment effect of promotion

### get the predict value of c group
predict.h2o <- h2o.predict(f3,c.h2o)
predict <- as.data.frame(predict.h2o)
c$predict <- predict$predict
c$resi <- c$profit - c$predict  ### the treatment effect of promotion

### get the predict value of s group
predict.h2o <- h2o.predict(f3,s.h2o)
predict <- as.data.frame(predict.h2o)
s$predict <- predict$predict
s$resi <- s$profit - s$predict  ### the treatment effect of promotion

### get the predict value of g group
predict.h2o <- h2o.predict(f3,g.h2o)
predict <- as.data.frame(predict.h2o)
g$predict <- predict$predict
g$resi <- g$profit - g$predict  ### the treatment effect of promotion


# 0.5 train the model of resi against control varibles on b,c,s,g

x <- c('PRICE','quarter','zone','size_1','binded','discounted','combo_shop','special_events','share')
y <- 'resi'

## bonus
b <- filter(b,resi>0)
set.seed(0)
test <- sample.int(nrow(b),0.2*nrow(b),replace = F)
b_train <- b[-test,]
b_test <- b[test,]
b_train.h2o <- as.h2o(b_train)
b_test.h2o <- as.h2o(b_test)

### linear b
#ln_b <- lm(resi~PRICE+quarter+zone+size_1+binded+discounted+combo_shop+special_events1+share, data=b_train)
#lnb.mse = mean((predict(ln_b, b_test) - b_test$resi)^2)

### lasso b
#cvg_lasso_lambda <- cv.glmnet(as.matrix(b_train[,x]),b_train$resi, 
#                              type.measure = "mse",nfolds = 8, alpha = 1)$lambda.min
#la_b <- glmnet(as.matrix(b_train[,x]),b_train$profit, 
#             type.measure = "mse",nfolds = 8, alpha = 1,lambda=cvg_lasso_lambda)
#la_b.mse <- mean((predict(la_b,as.matrix(b_test[,x]),s=cvg_lasso_lambda) - b_test$resi)^2)

### randomforest b
rm_b <-h2o.randomForest(x,y,b_train.h2o,mtries = 4,ntrees=250, seed=1122) 
rmb.perf<- h2o.performance(rm_b,b_test.h2o)
rmb.mse <- rmb.perf@metrics[["MSE"]]

### nerual network b
nn_b <-h2o.deeplearning(x,y,b_train.h2o,epochs = 10,hidden = c(64,32,16))
nnb.perf <- h2o.performance(nn_b,b_test.h2o)
nnb.mse <- nnb.perf@metrics[['MSE']]



## c
c <- filter(c,resi>0)
set.seed(0)
test <- sample.int(nrow(c),0.2*nrow(c),replace = F)
c_train <- c[-test,]
c_test <- c[test,]
c_train.h2o <- as.h2o(c_train)
c_test.h2o <- as.h2o(c_test)
### linear c
#ln_c <- lm(resi~PRICE+quarter+zone+size_1+binded+discounted+combo_shop+special_events1+share, data=c_train)
#lnc.mse = mean((predict(ln_c, c_test) - c_test$resi)^2)

### lasso c
#cvg_lasso_lambda <- cv.glmnet(as.matrix(c_train[,x_n]),c_train$resi, 
#                              type.measure = "mse",nfolds = 8, alpha = 1)$lambda.min
#la_c <- glmnet(as.matrix(c_train[,x]),c_train$profit, 
#               type.measure = "mse",nfolds = 8, alpha = 1,lambda=cvg_lasso_lambda)
#la_c.mse <- mean((predict(la_c,as.matrix(c_test[,x]),s=cvg_lasso_lambda) - c_test$resi)^2)

### randomforest c
rm_c <-h2o.randomForest(x,y,c_train.h2o,mtries = 4,ntrees=250, seed=1122) 
rmc.perf<- h2o.performance(rm_v,v_test.h2o)
rmc.mse <- rmc.perf@metrics[["MSE"]]

### nerual network c
nn_c <-h2o.deeplearning(x,y,c_train.h2o,epochs = 10,hidden = c(64,32,16))
nnc.perf <- h2o.performance(nn_c,c_test.h2o)
nnc.mse <- nnc.perf@metrics[['MSE']]


## s
s <- filter(s,resi>0)
set.seed(0)
test <- sample.int(nrow(s),0.2*nrow(s),replase = F)
s_train <- s[-test,]
s_test <- s[test,]
s_train.h2o <- as.h2o(s_train)
s_test.h2o <- as.h2o(s_test)
### linear s
#ln_s <- lm(resi~PRICE+quarter+zone+size_1+binded+discounted+combo_shop+special_events1+share, data=s_train)
#lns.mse = mean((predict(ln_s, s_test) - s_test$resi)^2)

### lasso s
#cvg_lasso_lambda <- cv.glmnet(as.matrix(s_train[,x]),s_train$resi, 
#                              type.measure = "mse",nfolds = 8, alpha = 1)$lambda.min
#la_s <- glmnet(as.matrix(s_train[,x]),s_train$profit, 
#               type.measure = "mse",nfolds = 8, alpha = 1,lambda=cvg_lasso_lambda)
#la_s.mse <- mean((predict(la_s,as.matrix(s_test[,x]),s=cvg_lasso_lambda) - s_test$resi)^2)

### randomforest s
rm_s <-h2o.randomForest(x,y,s_train.h2o,mtries = 4,ntrees=250, seed=1122) 
rms.perf<- h2o.performance(rm_v,v_test.h2o)
rms.mse <- rms.perf@metrics[["MSE"]]

### nerual network s
nn_s <-h2o.deeplearning(x,y,s_train.h2o,epochs = 10,hidden = c(64,32,16))
nns.perf <- h2o.performance(nn_s,s_test.h2o)
nns.mse <- nns.perf@metrics[['MSE']]


## g
g <- filter(g,resi>0)
set.seed(0)
test <- sample.int(nrow(g),0.2*nrow(g),replase = F)
g_train <- g[-test,]
g_test <- g[test,]
g_train.h2o <- as.h2o(g_train)
g_test.h2o <- as.h2o(g_test)
### linear g
#ln_g <- lm(resi~PRICE+quarter+zone+size_1+binded+discounted+combo_shop+special_events1+share, data=g_train)
#lng.mse = mean((predict(ln_g, g_test) - g_test$resi)^2)

### lasso g
#cvg_lasso_lambda <- cv.glmnet(as.matrix(g_train[,x_n]),g_train$resi, 
#                              type.measure = "mse",nfolds = 8, alpha = 1)$lambda.min
#la_g <- glmnet(as.matrix(g_train[,x_n]),g_train$profit, 
#               type.measure = "mse",nfolds = 8, alpha = 1,lambda=cvg_lasso_lambda)
#la_g.mse <- mean((predict(la_g,as.matrix(g_test[,x_n]),s=cvg_lasso_lambda) - g_test$resi)^2)

### randomforest g
rm_g <-h2o.randomForest(x,y,g_train.h2o,mtries = 4,ntrees=500, seed=1122) 
rmg.perf<- h2o.performance(rm_g,g_test.h2o)
rmg.mse <- rmg.perf@metrics[["MSE"]]

### nerual network g
nn_g <-h2o.deeplearning(x,y,g_train.h2o,epochs = 5,hidden = c(64,32,16))
nng.perf <- h2o.performance(nn_g,g_test.h2o)
nng.mse <- nng.perf@metrics[['MSE']]



# 0.6 get the partial dependence plot for random models of b,c,s,g
## b
b[,"special_events"]<- as.factor(b[,"special_events"])
b[,"quarter"]<- as.factor(b[,"quarter"])
b[,"special_events"] <- as.factor(b[,"special_events"])
b[,"zone"] <- as.factor(b[,"zone"])
b.h20 <- as.h2o(b)
rm_b <-h2o.randomForest(x,y,b.h20,mtries = 4,ntrees=100, seed=1122,max_depth = 15) 
b.importance <- h2o.varimp(rm_b)
h2o.partialPlot(rm_b,b.h20,cols = x, save_to ="D:/2020Fall/ECON 725/PROJECT/1/pdp_b")
## c
c[,"special_events"]<- as.factor(c[,"special_events"])
c[,"quarter"]<- as.factor(c[,"quarter"])
c[,"special_events"] <- as.factor(c[,"special_events"])
c[,"zone"] <- as.factor(c[,"zone"])
c.h20 <- as.h2o(c)
rm_c <-h2o.randomForest(x,y,c.h20,mtries = 4,ntrees=100, seed=1122,max_depth = 15) 
c.importance <- h2o.varimp(rm_c)
h2o.partialPlot(rm_c,c.h20,cols = x, save_to ="D:/2020Fall/ECON 725/PROJECT/1/pdp_c")

## s
s[,"special_events"]<- as.factor(s[,"special_events"])
s[,"quarter"]<- as.factor(s[,"quarter"])
s[,"special_events"] <- as.factor(s[,"special_events"])
s[,"zone"] <- as.factor(s[,"zone"])
s.h20 <- as.h2o(s)
rm_s <-h2o.randomForest(x,y,s.h20,mtries = 4,ntrees=100, seed=1122,max_depth = 15) 
s.importance <- h2o.varimp(rm_s)
h2o.partialPlot(rm_s,s.h20,cols = x, save_to ="D:/2020Fall/ECON 725/PROJECT/1/pdp_s") #### get the pdp



## g
g[,"special_events"]<- as.factor(g[,"special_events"])
g[,"quarter"]<- as.factor(g[,"quarter"])
g[,"special_events"] <- as.factor(g[,"special_events"])
g[,"zone"] <- as.factor(g[,"zone"])
g.h20 <- as.h2o(g)
rm_g <-h2o.randomForest(x,y,g.h20,mtries = 4,ntrees=100, seed=1122,max_depth = 15) 
g.importance <- h2o.varimp(rm_g)
h2o.partialPlot(rm_g,g.h20,cols = x, save_to ="D:/2020Fall/ECON 725/PROJECT/1/pdp_g") #### get the pdp


#0.7  Design a detailed promotion strategy

no_prom$predict <- no_prom$profit
no_prom$resi <- 0

large_np <- subset(no_prom,share == 0.2151343)
large_s <- subset(s,share == 0.2151343)
large_c <- subset(c,share == 0.2151343)
large_b <- subset(b,share == 0.2151343)
large_g <- subset(g,share == 0.2151343)

Lf <- rbind(large_np,large_b,large_c,large_g,large_s) #the merged data about largest firm

### Based on Lf, we will conduct our analysi on the distribution of promotion treatment effect
Lf$resi.per <- Lf$resi / Lf$MOVE * Lf$QTY   ##treatment effect of per purchase 

Lf$resi.per <- filter(Lf,resi.per>1) ### select the subset of sales whose promotion effect per purchase over $1

## to see the frequencies of the varibles
size<-table(Lf$size_1)
month <- table(Lf$month)
price <- table(LF$PRICE)
zone <- table(LF$zone)
festival <- table(special_events)


## choose the sale with largest profit and analyse the underlying condition
by_mps <- Lf %>% group_by(PRICE,size_1,month) %>% summarise(resi_mean = mean(resi.per))
by_oz <- Lf %>% group_by(size_1) %>% summarise(resi_mean = mean(resi.per))
by_zone <- Lf %>% group_by(zone) %>% summarise(resi_mean = mean(resi.per))

hist(Lf$month)
hist(Lf$size)
hist(Lf$zone)

oz15 <- filter(Lf, size_1 == 15) # get the subset of products with which size has the best promotion effect


