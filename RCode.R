set.seed(999);

getClassificationAccuracy <- function(trueY, predictedY){
  confusionMatrix <- table(trueY, predictedY);
  
  truePos <- confusionMatrix[1, 1];
  trueNeg <- confusionMatrix[2, 2];
  falsePos <- confusionMatrix[1, 2];
  falseNeg <- confusionMatrix[2, 1];
  
  accuracy <- (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg);
  
  library(ROCR);
  pred <- prediction(as.numeric(predictedY), trueY);
  perf <- performance(pred,"tpr","fpr");
  plot(perf,colorize=TRUE);
  
  auc.tmp <- performance(pred,"auc"); 
  auc <- as.numeric(auc.tmp@y.values);
  
  print("Accuracy:");
  print(accuracy);
  print("AUC Score:");
  print(auc);
}

#Read file
data <- read.csv(file = 'C:/Users/Kyle/OneDrive/Documents/Data Analytics/Project/haberman.data');

#Data is arranged in order of age. Rearrange to be random order
rows <- sample(nrow(data))
data <- data[rows,]

#split into training and testing
trainingSet <- data[1:250,];
colnames(trainingSet) <- c("x1", "x2", "x3", "class");
testingSet <- data[251:306,];
colnames(testingSet) <- c("x1", "x2", "x3", "class");
y_train <- trainingSet[,4];
X_train <- trainingSet[,1:3];
y_test <- testingSet[,4];
X_test <- testingSet[,1:3];

#training set counts
count1 <- 0;
count2 <- 0;

for(i in unlist(trainingSet["class"])){
  ifelse(i == 1, count1 <- count1 + 1, count2 <- count2 + 1);
}

#Set up SMOTE
library(smotefamily)
trainingSMOTE <- SMOTE(X_train, y_train);

SMOTEcount1 <- 0;
SMOTEcount2 <- 0;

for(i in trainingSMOTE$data$class){
  ifelse(i == "1", SMOTEcount1 <- SMOTEcount1 + 1, SMOTEcount2 <- SMOTEcount2 + 1);
}

#Set up BSMOTE
trainingBSMOTE <- BLSMOTE(X_train, y_train);

BSMOTEcount1 <- 0;
BSMOTEcount2 <- 0;

for(i in trainingBSMOTE$data$class){
  ifelse(i == "1", BSMOTEcount1 <- BSMOTEcount1 + 1, BSMOTEcount2 <- BSMOTEcount2 + 1);
}

#Set up ADASYN
trainingADASYN <- ANS(X_train, y_train);

ADASYNcount1 <- 0;
ADASYNcount2 <- 0;

for(i in trainingADASYN$data$class){
  ifelse(i == "1", ADASYNcount1 <- ADASYNcount1 + 1, ADASYNcount2 <- ADASYNcount2 + 1);
}



#KNN
library(kknn); 
kNNModel <- kknn(as.factor(class) ~ ., trainingSet, testingSet);
getClassificationAccuracy(y_test, kNNModel$fit);

#KNN with SMOTE
kNNModelSMOTE <- kknn(as.factor(class) ~ ., trainingSMOTE[["data"]], testingSet);
getClassificationAccuracy(y_test, kNNModelSMOTE$fit);

#KNN with BSMOTE
kNNModelBSMOTE <- kknn(as.factor(class) ~ ., trainingBSMOTE[["data"]], testingSet);
getClassificationAccuracy(y_test, kNNModelBSMOTE$fit);

#KNN with ADASYN
kNNModelADASYN <- kknn(as.factor(class) ~ ., trainingADASYN[["data"]], testingSet);
getClassificationAccuracy(y_test, kNNModelADASYN$fit);



#Naive Bayes
library(e1071); 
NBModel <- naiveBayes(as.factor(class) ~ ., data = trainingSet);
NBPredict <- predict(NBModel, X_test); 
getClassificationAccuracy(y_test, NBPredict)

#NB with SMOTE
NBModelSMOTE <- naiveBayes(as.factor(class) ~ ., data = trainingSMOTE[["data"]]);
NBPredictSMOTE <- predict(NBModelSMOTE, newdata = X_test);
getClassificationAccuracy(y_test, NBPredictSMOTE);

#NB with BSMOTE
NBModelBSMOTE <- naiveBayes(as.factor(class) ~ ., data = trainingBSMOTE[["data"]]);
NBPredictBSMOTE <- predict(NBModelBSMOTE, newdata = X_test);
getClassificationAccuracy(y_test, NBPredictBSMOTE);

#NB with ADASYN
NBModelADASYN <- naiveBayes(as.factor(class) ~ ., data = trainingADASYN[["data"]]);
NBPredictADASYN <- predict(NBModelADASYN, newdata = X_test);
getClassificationAccuracy(y_test, NBPredictADASYN);



#Logistic Regression
LRModel <- glm(as.factor(class) ~., family = binomial(link='logit'), data = trainingSet); 
LRPredict <- predict(LRModel, newdata = testingSet, type = "response"); 
LRPredictClass <- ifelse(LRPredict > 0.5,1,0); 
getClassificationAccuracy(y_test, LRPredictClass);

#LR with SMOTE
LRModelSMOTE <- glm(as.factor(class) ~., family = binomial(link='logit'), data = trainingSMOTE[["data"]]);
LRPredictSMOTE <- predict(LRModelSMOTE, newdata = testingSet, type = "response");
LRPredictClassSMOTE <- ifelse(LRPredictSMOTE > 0.5, 1,0);
getClassificationAccuracy(y_test, LRPredictClassSMOTE);

#LR with BSMOTE
LRModelBSMOTE <- glm(as.factor(class) ~., family = binomial(link='logit'), data = trainingBSMOTE[["data"]]);
LRPredictBSMOTE <- predict(LRModelBSMOTE, newdata = testingSet, type = "response");
LRPredictClassBSMOTE <- ifelse(LRPredictBSMOTE > 0.5, 1, 0);
getClassificationAccuracy(y_test, LRPredictClassBSMOTE);

#LR with ADASYN
LRModelADASYN <- glm(as.factor(class)~., family = binomial(link='logit'), data = trainingADASYN[["data"]]);
LRPredictADASYN <- predict(LRModelADASYN, newdata = testingSet, type = "response");
LRPredictClassADASYN <- ifelse(LRPredictADASYN > 0.5, 1, 0);
getClassificationAccuracy(y_test, LRPredictClassADASYN);



#Linear Discriminant analysis
library(MASS);
LDAModel <- lda(as.factor(class)~., data = trainingSet);
LDAPredict <- predict(LDAModel, newdata = testingSet);
getClassificationAccuracy(y_test, LDAPredict$class);

#LDA with SMOTE
LDAModelSMOTE <- lda(as.factor(class)~., data = trainingSMOTE[["data"]]);
LDAPredictSMOTE <- predict(LDAModelSMOTE, newdata = testingSet);
getClassificationAccuracy(y_test, LDAPredictSMOTE$class);

#LDA with BsMOTE
LDAModelBSMOTE <- lda(as.factor(class)~., data = trainingBSMOTE[["data"]]);
LDAPredictBSMOTE <- predict(LDAModelBSMOTE, newdata = testingSet);
getClassificationAccuracy(y_test, LDAPredictBSMOTE$class);

#LDA with ADASYN
LDAModelADASYN <- lda(as.factor(class)~., data = trainingADASYN[["data"]]);
LDAPredictADASYN <- predict(LDAModelADASYN, newdata = testingSet);
getClassificationAccuracy(y_test, LDAPredictADASYN$class);



#Support Vector Machine
library(kernlab);
SVMModel <- ksvm(as.factor(class) ~ ., data = trainingSet, kernel = "rbfdot");
SVMPredict <- predict(SVMModel, newdata = testingSet);
getClassificationAccuracy(y_test, SVMPredict);

#SVM with SMOTE
SVMModelSMOTE <- ksvm(as.factor(class)~., data = trainingSMOTE[["data"]], kernel = "rbfdot");
SVMPredictSMOTE <- predict(SVMModelSMOTE, newdata = testingSet);
getClassificationAccuracy(y_test, SVMPredictSMOTE);

#SVM with BSMOTE
SVMModelBSMOTE <- ksvm(as.factor(class)~., data = trainingBSMOTE[["data"]], kernel = "rbfdot");
SVMPredictBSMOTE <- predict(SVMModelBSMOTE, newdata = testingSet);
getClassificationAccuracy(y_test, SVMPredictBSMOTE);

#SVM with ADASYN
SVMModelADASYN <- ksvm(as.factor(class)~., data = trainingADASYN[["data"]], kernel = "rbfdot");
SVMPredictADASYN <- predict(SVMModelADASYN, newdata = testingSet);
getClassificationAccuracy(y_test, SVMPredictADASYN);
