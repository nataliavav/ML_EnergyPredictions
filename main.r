#############################################################################
# Industry 4.0 and Smart Cities
# Project in R   
#############################################################################

#install.packages("rpart")
#install.packages("randomForest")
#install.packages("gbm")

#libraries 
library(lubridate) 
library(DT) 
library(forecast)
library(plyr)
library(ggplot2) 
library(nnet) 
library(neuralnet)
library(RSNNS)
library(rpart) # Decision tree regressor
library(randomForest) # Random forest
library(gbm)
library(corrplot)
library(tidyverse)

#remove existing variables in the workspace
rm(list=ls())
#change the current directory as a working directory
setwd("C:/ergasia_ind_4.0")
#Read data
train <- read.csv("TrainSet.csv",stringsAsFactors = F,sep=",")

#Identify missing values
missing <- train[is.na(train$Prices.BE)==T,]
missing #print missing values

#Fix missing values
train$Date <- as.Date(train$datetime_utc)
train$DateType <- wday(train$Date)
train$Hour <- hour(train$datetime_utc)
train$Month <- month(train$datetime_utc)
train$Year <- year(train$datetime_utc)

#Create profiles
profiles1 <- ddply(na.omit(train[,-1]), .(DateType,Hour), colwise(mean))
profiles2 <- ddply(na.omit(train[,-1]), .(DateType,Hour, Month), colwise(mean))
profiles3 <- ddply(na.omit(train[,-1]), .(DateType,Hour, Month, Year), colwise(mean))

train1 = train2 = train3 <- train
#Fill Nas  with profiles
for (i in 59300:59535){
  if (is.na(train$Prices.BE[i])==T){
    train1$Prices.BE[i] <- profiles1[(profiles1$Hour==train1$Hour[i])&(profiles1$DateType==train1$DateType[i]),]$Prices.BE
    train2$Prices.BE[i] <- profiles2[(profiles2$Hour==train2$Hour[i])&(profiles2$Month==train2$Month[i])&(profiles2$DateType==train2$DateType[i]),]$Prices.BE
    train3$Prices.BE[i] <- profiles3[(profiles3$Hour==train3$Hour[i])&(profiles3$Month==train3$Month[i])&(profiles3$Year==train3$Year[i])&(profiles3$DateType==train3$DateType[i]),]$Prices.BE
  }
}


plot(train1$Prices.BE[59300:59535],type="l", ylab = "Price",col=2)
lines(train2$Prices.BE[59300:59535],type="l", ylab = "Price",col=3)
lines(train3$Prices.BE[59300:59535],type="l", ylab = "Price",col=4)
lines(train$Prices.BE[59300:59535],type="l")
legend("topleft", legend=c("DateType,Hour", "DateType,Hour,Month", "DateType,Hour,Month,Year"),col=c(2,3, 4), lty=1, cex=0.8)

train<-train2
#gia na grapsw sto file ta nea
write.csv(train[,c("datetime_utc", "Generation_BE", "Generation_FR","Prices.BE", "holidaysBE")], "TrainSet.csv", row.names = FALSE) 

#Fix upper and lower bounds
LimitUp <- quantile(train$Prices.BE,0.999)
LimitDown <- quantile(train$Prices.BE,0.001)
train[train$Prices.BE>LimitUp,]$Prices.BE <- LimitUp
train[train$Prices.BE<LimitDown,]$Prices.BE <- LimitDown

##############################################################################

#Examine possible scenarios
timeseries <- ts(train$Prices.BE, frequency=168)
fh <- 168   #Set the forecasting horizon to 168
insample <- head(timeseries,length(timeseries)-fh) #for training
outsample <- tail(timeseries,fh)    #for evaluation

#Include explanatory variables such as Year, Month, Weekday,
#Weekend,Lags (same case previous day, week , 2 weeks, 3 weeks), and season
Data_ml <- train
Data_ml$Year <- year(Data_ml$datetime_utc) #Define Year
Data_ml$Month <- month(Data_ml$datetime_utc) #Define Month
Data_ml$Weekday <- 1
Data_ml[(Data_ml$DateType==1)|(Data_ml$DateType==7),]$Weekday <- 0
Data_ml$Lag168 = Data_ml$Lag336 <- NA #Define Level
Data_ml$Lag168 <- head(c(rep(NA,168), head(Data_ml,nrow(Data_ml)-168)$Prices.BE),nrow(Data_ml))
Data_ml$Lag336 <- head(c(rep(NA,336), head(Data_ml,nrow(Data_ml)-336)$Prices.BE),nrow(Data_ml))
Data_ml$Lag504 <- head(c(rep(NA,504), head(Data_ml,nrow(Data_ml)-504)$Prices.BE),nrow(Data_ml)) 
Data_ml$Season <- case_when(
  month(Data_ml$datetime_utc) %in% 10:12 ~ 1,
  month(Data_ml$datetime_utc) %in%  1:3  ~ 2,
  month(Data_ml$datetime_utc) %in%  4:6  ~ 3,
  TRUE ~ 4)

#check data in new file
write.csv(Data_ml, "Check.csv", row.names = FALSE) 
Data_ml <- na.omit(Data_ml) #Delete NAs
corrplot(cor(Data_ml[,-c(1,6)]), method="color")

insample_ml <- head(Data_ml,nrow(Data_ml)-fh) #insample for training
outsample_ml <- tail(Data_ml,fh) #outsample for testing

##############################################################################
##############################################################################
##############################################################################
#Supervised learning methods- Forecast Methods - TrainSet
# Test various Forecasting Methods
##############################################################################
##############################################################################
##############################################################################

#######
# Naive
#######
# frc1 <- naive(insample,h=fh)$mean
# mean(200*abs(outsample-frc1)/(abs(outsample)+abs(frc1)))
# 
# plot(outsample)
# lines(frc1+outsample-outsample, col=2)
# legend("topleft", legend=c("Naive"),col=c(2:5), lty=1, cex=0.8)
# 
#########################
# #SES - no decomposition
#########################
# frc2 <- ses(insample,h=fh)$mean
# mean(200*abs(outsample-frc2)/(abs(outsample)+abs(frc2)))
# 
# plot(outsample)
# lines(frc2+outsample-outsample, col=2)
# legend("topleft", legend=c("SES - no decomposition"),col=c(2:5), lty=1, cex=0.8)
# 
###############
#Seasonal Naive
###############
# frc3 <- as.numeric(tail(insample,fh)) + outsample - outsample
# mean(200*abs(outsample-frc3)/(abs(outsample)+abs(frc3)))
# 
# plot(outsample)
# lines(frc3+outsample-outsample, col=2)
# legend("topleft", legend=c("Seasonal Naive"),col=c(2:5), lty=1, cex=0.8)
# 
######################################
#SES - with decomposition (Additive)
######################################
# Indexes_in <- decompose(insample, type = "additive")$seasonal
# Indexes_out <- as.numeric(tail(Indexes_in,fh))
# frc4 <- ses(insample-Indexes_in,h=fh)$mean+Indexes_out
# mean(200*abs(outsample-frc4)/(abs(outsample)+abs(frc4)))
# 
# plot(outsample)
# lines(frc4+outsample-outsample, col=2)
# legend("topleft", legend=c("SES - with decomposition (Additive)"),col=c(2:5), lty=1, cex=0.8)
# 
##########################################
#SES - with decomposition (Multiplicative)
##########################################
# Indexes_in <- decompose(insample, type = "multiplicative")$seasonal
# Indexes_out <- as.numeric(tail(Indexes_in,fh))
# frc5 <- ses(insample/Indexes_in,h=fh)$mean*Indexes_out
# mean(200*abs(outsample-frc5)/(abs(outsample)+abs(frc5)))
# 
# plot(outsample)
# lines(frc5+outsample-outsample, col=2)
# legend("topleft", legend=c("SES - with decomposition (Multiplicative)"),col=c(2:5), lty=1, cex=0.8)
#

###############################################################################
###############################################################################

##################################
#Multiple linear regression models
##################################
# 
# ml_model <- lm(Prices.BE~Generation_FR+ Year+ Hour+ Lag168+ Lag336+ Lag504,data=insample_ml)
# frc6_1 <- predict(ml_model,outsample_ml)
# mean(200*abs(outsample_ml$Prices.BE-frc6_1)/(abs(outsample_ml$Prices.BE)+abs(frc6_1))) 
# 
# plot(outsample)
# lines(frc6_1+outsample-outsample, col=2)
# legend("topleft", legend=c("MLR1"),col=c(2:5), lty=1, cex=0.8)
# 
# ml_model <- lm(Prices.BE~Generation_FR+ Year+ Hour+ Lag168+ Lag336+ Lag504+ Season,data=insample_ml)
# frc6_2 <- predict(ml_model,outsample_ml)
# mean(200*abs(outsample_ml$Prices.BE-frc6_2)/(abs(outsample_ml$Prices.BE)+abs(frc6_2))) 
# 
# plot(outsample)
# lines(frc6_2+outsample-outsample, col=2)
# legend("topleft", legend=c("MLR2"),col=c(2:5), lty=1, cex=0.8)
# 
# ml_model <- lm(Prices.BE~Generation_FR++holidaysBE+ Year+ Hour+ Lag168+ Lag336+ Lag504+ Season,data=insample_ml)
# frc6_3 <- predict(ml_model,outsample_ml)
# mean(200*abs(outsample_ml$Prices.BE-frc6_3)/(abs(outsample_ml$Prices.BE)+abs(frc6_3))) 
# 
# plot(outsample)
# lines(frc6_3+outsample-outsample, col=2)
# legend("topleft", legend=c("MLR3"),col=c(2:5), lty=1, cex=0.8)
# 
# ml_model <- lm(Prices.BE~Generation_FR++holidaysBE+ Year+ Hour+ Month +Lag168+ Lag336+ Lag504+ Season,data=insample_ml)
# frc6_4 <- predict(ml_model,outsample_ml)
# mean(200*abs(outsample_ml$Prices.BE-frc6_4)/(abs(outsample_ml$Prices.BE)+abs(frc6_4))) 
# 
# plot(outsample)
# lines(frc6_4+outsample-outsample, col=2)
# legend("topleft", legend=c("MLR4"),col=c(2:5), lty=1, cex=0.8)
# 
# ml_model <- lm(Prices.BE~Generation_FR+ Year+ Hour+ Weekday+ Lag168+ Lag336+ Lag504+ Season,data=insample_ml)
# frc6_5 <- predict(ml_model,outsample_ml)
# mean(200*abs(outsample_ml$Prices.BE-frc6_5)/(abs(outsample_ml$Prices.BE)+abs(frc6_5))) 
# 
# plot(outsample)
# lines(frc6_5+outsample-outsample, col=2)
# legend("topleft", legend=c("MLR5"),col=c(2:5), lty=1, cex=0.8)
# 
# ml_model <- lm(Prices.BE~Generation_FR+ Year+ Hour+ Weekday + Lag168,data=insample_ml)
# frc6_6 <- predict(ml_model,outsample_ml)
# mean(200*abs(outsample_ml$Prices.BE-frc6_6)/(abs(outsample_ml$Prices.BE)+abs(frc6_6))) 
# 
# plot(outsample)
# lines(frc6_6+outsample-outsample, col=2)
# legend("topleft", legend=c("MLR6"),col=c(2:5), lty=1, cex=0.8)
# 
# ml_model <- lm(Prices.BE~Generation_FR+ Year+ Hour+ Weekday + Lag168 +Lag336,data=insample_ml)
# frc6_7 <- predict(ml_model,outsample_ml)
# mean(200*abs(outsample_ml$Prices.BE-frc6_7)/(abs(outsample_ml$Prices.BE)+abs(frc6_7))) 
# 
# plot(outsample)
# lines(frc6_7+outsample-outsample, col=2)
# legend("topleft", legend=c("MLR7"),col=c(2:5), lty=1, cex=0.8)
# 
# ml_model <- lm(Prices.BE~Generation_FR+ Year+ Hour+ Weekday + Lag168 +Lag336+ Lag504,data=insample_ml)
# frc6_8 <- predict(ml_model,outsample_ml)
# mean(200*abs(outsample_ml$Prices.BE-frc6_8)/(abs(outsample_ml$Prices.BE)+abs(frc6_8))) 
# 
# plot(outsample)
# lines(frc6_8+outsample-outsample, col=2)
# legend("topleft", legend=c("MLR8"),col=c(2:5), lty=1, cex=0.8)
#

###############################################################################
###############################################################################

#################
# Neural Networks
#################
# normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }
# ForScaling <- rbind(insample_ml,outsample_ml)[,c("Year","Month","Hour",
#                                                  "Weekday","holidaysBE",
#                                                  "Lag168","Lag336")]
# ForScaling <- as.data.frame(lapply(ForScaling, normalize))
# trainNN <- head(ForScaling, nrow(ForScaling)-fh)
# validateNN <- normalize(insample_ml$Prices.BE)
# testNN <- tail(ForScaling, fh)
# 
# model1<-mlp(trainNN, validateNN,
#             size = 20, maxit = 100,initFunc = "Randomize_Weights",
#             learnFunc = "BackpropWeightDecay", hiddenActFunc = "Act_Logistic",
#             shufflePatterns = FALSE, linOut = FALSE)
# frc7_1 <- as.numeric(predict(model1,testNN))*(max(insample_ml$Prices.BE) - min(insample_ml$Prices.BE)) + min(insample_ml$Prices.BE)
# mean(200*abs(outsample-frc7_1)/(abs(outsample)+abs(frc7_1))) 
# 
# plot(outsample)
# lines(frc7_1+outsample-outsample, col=2)
# legend("topleft", legend=c("NN1"),col=c(2:5), lty=1, cex=0.8)
# 
# model2<-mlp(trainNN, validateNN,
#             size = c(20,15,10,5), maxit = 100,initFunc = "Randomize_Weights",
#             learnFunc = "BackpropWeightDecay", hiddenActFunc = "Act_Logistic",
#             shufflePatterns = FALSE, linOut = FALSE)
# frc7_2 <- as.numeric(predict(model2,testNN))*(max(insample_ml$Prices.BE) - min(insample_ml$Prices.BE)) + min(insample_ml$Prices.BE)
# mean(200*abs(outsample-frc7_2)/(abs(outsample)+abs(frc7_2))) 
# 
# plot(outsample)
# lines(frc7_2+outsample-outsample, col=2)
# legend("topleft", legend=c("NN2"),col=c(2:5), lty=1, cex=0.8)
# 
# normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }
# ForScaling <- rbind(insample_ml,outsample_ml)[,c("Year","Month","Hour",
#                                                  "Weekday","holidaysBE",
#                                                  "Lag168","Lag336", "Lag504", "Season")]
# ForScaling <- as.data.frame(lapply(ForScaling, normalize))
# trainNN <- head(ForScaling, nrow(ForScaling)-fh)
# validateNN <- normalize(insample_ml$Prices.BE)
# testNN <- tail(ForScaling, fh)
# 
# model3<-mlp(trainNN, validateNN,
#             size = 20, maxit = 100,initFunc = "Randomize_Weights",
#             learnFunc = "BackpropWeightDecay", hiddenActFunc = "Act_Logistic",
#             shufflePatterns = FALSE, linOut = FALSE)
# frc7_3 <- as.numeric(predict(model3,testNN))*(max(insample_ml$Prices.BE) - min(insample_ml$Prices.BE)) + min(insample_ml$Prices.BE)
# mean(200*abs(outsample-frc7_3)/(abs(outsample)+abs(frc7_3))) 
# 
# plot(outsample)
# lines(frc7_3+outsample-outsample, col=2)
# legend("topleft", legend=c("NN3"),col=c(2:5), lty=1, cex=0.8)
# 
# model4<-mlp(trainNN, validateNN,
#             size = c(20,15,10,5), maxit = 100,initFunc = "Randomize_Weights",
#             learnFunc = "BackpropWeightDecay", hiddenActFunc = "Act_Logistic",
#             shufflePatterns = FALSE, linOut = FALSE)
# frc7_4 <- as.numeric(predict(model4,testNN))*(max(insample_ml$Prices.BE) - min(insample_ml$Prices.BE)) + min(insample_ml$Prices.BE)
# mean(200*abs(outsample-frc7_4)/(abs(outsample)+abs(frc7_4))) 
# 
# plot(outsample)
# lines(frc7_4+outsample-outsample, col=2)
# legend("topleft", legend=c("NN4"),col=c(2:5), lty=1, cex=0.8)
# 
# model5<-mlp(trainNN, validateNN,
#             size = 25, maxit = 100,initFunc = "Randomize_Weights",
#             learnFunc = "BackpropWeightDecay", hiddenActFunc = "Act_Logistic",
#             shufflePatterns = FALSE, linOut = FALSE)
# frc7_5 <- as.numeric(predict(model5,testNN))*(max(insample_ml$Prices.BE) - min(insample_ml$Prices.BE)) + min(insample_ml$Prices.BE)
# mean(200*abs(outsample-frc7_5)/(abs(outsample)+abs(frc7_5))) 
# 
# plot(outsample)
# lines(frc7_5+outsample-outsample, col=2)
# legend("topleft", legend=c("NN5"),col=c(2:5), lty=1, cex=0.8)
# 
# model6<-mlp(trainNN, validateNN,
#             size = c(20,15,10), maxit = 100,initFunc = "Randomize_Weights",
#             learnFunc = "BackpropWeightDecay", hiddenActFunc = "Act_Logistic",
#             shufflePatterns = FALSE, linOut = FALSE)
# frc7_6 <- as.numeric(predict(model6,testNN))*(max(insample_ml$Prices.BE) - min(insample_ml$Prices.BE)) + min(insample_ml$Prices.BE)
# mean(200*abs(outsample-frc7_6)/(abs(outsample)+abs(frc7_6))) 
# 
# plot(outsample)
# lines(frc7_6+outsample-outsample, col=2)
# legend("topleft", legend=c("NN6"),col=c(2:5), lty=1, cex=0.8)

###############################################################################
###############################################################################

##############
#Decision tree
##############
#ml_model <- rpart(Prices.BE~Year+Month+Hour+
#                Weekday+holidaysBE+
#                Generation_FR+Generation_BE, 
#                method = "anova",data=insample_ml) # Decision tree sMAPE: 40.71718 %
#frc6_5 <- predict(ml_model,outsample_ml)
#mean(200*abs(outsample-frc6_5)/(abs(outsample)+abs(frc6_5)))

#ml_model <- rpart(Prices.BE~Year+Month+Hour+
#                    Weekday+holidaysBE+
#                    Lag168+Lag336, 
#                  method = "anova",data=insample_ml) # Decision tree sMAPE: 26.73303 %
#frc6_6 <- predict(ml_model,outsample_ml)
#mean(200*abs(outsample-frc6_6)/(abs(outsample)+abs(frc6_6)))

#ml_model <- rpart(Prices.BE~Year+Month+Hour+
#                    Weekday+holidaysBE+
#                    Lag168+Lag336+Lag504, 
#                  method = "anova",data=insample_ml) # Decision tree sMAPE: 26.69543 %
#frc6_7 <- predict(ml_model,outsample_ml)

#mean(200*abs(outsample-frc6_7)/(abs(outsample)+abs(frc6_7)))

###############################################################################
###############################################################################

##############
#Random forest
##############
#ml_model <- randomForest(Prices.BE ~ . , data = insample_ml, ntree = 10)
#frc6_8 <- predict(ml_model,outsample_ml)
#mean(200*abs(outsample-frc6_8)/(abs(outsample)+abs(frc6_8)))

#ml_model <- randomForest(Prices.BE~ . , data = insample_ml, ntree = 100)
#frc6_9 <- predict(ml_model,outsample_ml)
#mean(200*abs(outsample-frc6_9)/(abs(outsample)+abs(frc6_9))) #19.76051 %

#ml_model <- randomForest(Prices.BE~Year+Month+Hour+
#                          Weekday+holidaysBE+
#                           Lag168+Lag336+Lag504 , data = insample_ml, ntree = 100)
#frc6_9_1 <- predict(ml_model,outsample_ml)
#mean(200*abs(outsample-frc6_9_1)/(abs(outsample)+abs(frc6_9_1)))

###############################################################################
###############################################################################

########################
#Gradient boosting trees
########################
#ml_model <- gbm(Prices.BE~Year+Month+Hour+
#                  Weekday+holidaysBE+
#                  Lag168+Lag336+Lag504, data = insample_ml, distribution = "gaussian",n.trees =
#                  10, shrinkage = 0.01, interaction.depth = 4)
#frc6_10 <- predict(ml_model,outsample_ml)
#mean(200*abs(outsample-frc6_10)/(abs(outsample)+abs(frc6_10))) # 38.40503 %


#ml_model <- gbm(Prices.BE~Year+Month+Hour+
#                  Weekday+holidaysBE+
#                  Lag168+Lag336+Lag504, data = insample_ml, distribution = "gaussian",n.trees =
#                  1000, shrinkage = 0.01, interaction.depth = 4)
#frc6_11 <- predict(ml_model,outsample_ml)
#mean(200*abs(outsample-frc6_11)/(abs(outsample)+abs(frc6_11))) # 21.18603 %

ml_model <- gbm(Prices.BE~Year+Month+Hour+
                  Weekday+holidaysBE+
                  Lag168+Lag336+Lag504, data = insample_ml, distribution = "gaussian",n.trees =
                  10000, shrinkage = 0.01, interaction.depth = 4)
frc6_12 <- predict(ml_model,outsample_ml)
mean(200*abs(outsample-frc6_12)/(abs(outsample)+abs(frc6_12))) # sMAPE 21.34316 % 

#ml_model <- gbm(Prices.BE~Year+Month+Hour+
#                  Weekday+holidaysBE+
#                  Lag168, data = insample_ml, distribution = "gaussian",n.trees =
#                  10000, shrinkage = 0.01, interaction.depth = 4)
#frc6_13 <- predict(ml_model,outsample_ml)
#mean(200*abs(outsample-frc6_13)/(abs(outsample)+abs(frc6_13))) # sMAPE 21.77132

ml_model <- gbm(Prices.BE~Year+Month+Hour+
                  Weekday+holidaysBE, data = insample_ml, distribution = "gaussian",n.trees =
                  10000, shrinkage = 0.01, interaction.depth = 4)

frc6_14 <- predict(ml_model,outsample_ml)


mean(200*abs(outsample-frc6_14)/(abs(outsample)+abs(frc6_14))) # 22.77809 %,

frc6_15 <- (frc6_12 + frc6_14)/2 # average between two previous methods
mean(200*abs(outsample-frc6_15)/(abs(outsample)+abs(frc6_15))) # 18.82135 % 

# frc6_16 <- (frc6_9 + frc6_12 + frc6_14)/3 # average between three previous methods
# mean(200*abs(outsample-frc6_16)/(abs(outsample)+abs(frc6_16))) # 19.24452 %

#Plot difference
plot(outsample)
lines(frc6_15+outsample-outsample, col=2)
legend("topleft", legend=c("MLR15"),col=c(2:5), lty=1, cex=0.8)

##############################################################################
##############################################################################
##############################################################################

##############################################################################
##############################################################################
##############################################################################
#Supervised learning methods-Forecast Methods with
#Gradient boosting trees - TestSet
##############################################################################
##############################################################################
test <- read.csv("TestSet.csv") 
test$Date <- as.Date(test$datetime_utc)
test$DateType <- wday(test$Date)
test$Hour <- hour(test$datetime_utc)

#Include explanatory variables such as Year, Month, Weekday, Weekend,
#Lags (same case previous day, week , 2 weeks, 3 weeks), and season
Predictions_ml <- test
Predictions_ml$Year <- year(Predictions_ml$datetime_utc) #Define Year
Predictions_ml$Month <- month(Predictions_ml$datetime_utc) #Define Month
Predictions_ml$Weekday <- 1
Predictions_ml[(Predictions_ml$DateType==1)|(Predictions_ml$DateType==7),]$Weekday <- 0
Predictions_ml$Lag168 = Predictions_ml$Lag336 = Predictions_ml$Lag504 <- NA #Define Level
Predictions_ml$Lag168 <- head(c(tail(Data_ml,nrow(Data_ml))$Prices.BE),nrow(Predictions_ml))
Predictions_ml$Lag336 <- head(c(tail(Data_ml,nrow(Data_ml)-168)$Prices.BE),nrow(Predictions_ml))
Predictions_ml$Lag504 <- head(c(tail(Data_ml,nrow(Data_ml)-336)$Prices.BE),nrow(Predictions_ml))
Predictions_ml$Season <- case_when(
  month(Predictions_ml$datetime_utc) %in% 10:12 ~ 1,
  month(Predictions_ml$datetime_utc) %in%  1:3  ~ 2,
  month(Predictions_ml$datetime_utc) %in%  4:6  ~ 3,
  TRUE ~ 4)
#Predictions_ml$Prices.BE
Predictions_ml$Prices.BE <-head(c(tail(Data_ml,nrow(Data_ml))$Prices.BE),nrow(Predictions_ml)) #giati me NA den problepei, alla den me ephreazei auth h sthlh sthn problepsh, tha mporoyse na einai otidhpote
write.csv(Predictions_ml, "Check_test.csv", row.names = FALSE) 
#Predictions_ml <- na.omit(Predictions_ml) 

rm(insample_ml)
insample_ml<-Data_ml
rm(outsample_ml)
outsample_ml<-Predictions_ml
rm(ml_model)

########################
#Gradient boosting trees
########################
ml_model <- gbm(Prices.BE~Year+Month+Hour+
                  Weekday+holidaysBE+
                  Lag168+Lag336+Lag504, data = insample_ml, distribution = "gaussian",n.trees =
                  10000, shrinkage = 0.01, interaction.depth = 4)
frc6_12_new <- predict(ml_model,outsample_ml) #
frc6_12_new
ml_model <- gbm(Prices.BE~Year+Month+Hour+
                  Weekday+holidaysBE, data = insample_ml, distribution = "gaussian",n.trees =
                  10000, shrinkage = 0.01, interaction.depth = 4)
frc6_14_new <- predict(ml_model,outsample_ml)

frc6_15_new <- (frc6_12_new + frc6_14_new)/2 # average between two previous methods
test$Prices.BE <- frc6_15_new
write.csv(test[,c("datetime_utc", "Generation_BE", "Generation_FR","Prices.BE", "holidaysBE")], "TestSet.csv", row.names = FALSE)

