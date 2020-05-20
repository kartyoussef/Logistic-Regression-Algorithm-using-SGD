# -*- coding: utf-8 -*-
######################################################################
#    Regression logistique avec descente de gradient stochastique
#               Author :  Youssef Kartit
######################################################################

##### Libraries :
from math import exp
import dataprep as dp

### Fonction Logistique
def logistic(x):
    return 1.0 / (1.0 + exp(-x))

####
def accuracy_metric(test, predicted):
    correct = 0
    for k in range(len(test)):
        if test[k] == predicted[k]:
            correct += 1
    return (correct / len(test)) * 100

####
def evaluate_algorithm(dataset, algorithm, K, *args):
    ''' This function '''
    folds = dp.crossvalidation_split(dataset, K)
    scores = []
    for fold in folds: 
        trainset = list(folds)
        trainset.remove(fold)
        trainset = sum(trainset, [])
        testset = []
        for row in fold:
            row_copy = list(row)
            testset.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(trainset, testset, *args)
        test = [row[-1] for row in fold]
        accuracy = accuracy_metric(test, predicted)
        scores.append(accuracy)
    return scores

#### Prédiction de la valeur à partir d'un x  
def predict(x, w):
    ''' This fuction computes the prediction y based on a new value x and w '''
    y_ = w[0]
    for k in range(len(x)-1):
        y_ += w[k + 1] * x[k]
    return logistic(y_)

#### coeff
def coefficients_sgd(train, learning_rate, T):
    ''' This function compute the optimal w'''
    w_ = [0 for i in range(len(train[0]))] 
    for t in range(T):
        for row in train: 
            y_ = predict(row, w_)
            v = row[-1] - y_
            w_[0] = w_[0] + learning_rate * v * y_ * (1 - y_)
            for k in range(len(row)-1):
                w_[k + 1] = w_[k + 1] + learning_rate * v * y_ * (1 - y_) * row[k]  
    return w_

#### Régression logistique avec SGD: 
def logistic_regression_sgd(train, test, learning_rate, T):
    predictions = []
    w_ = coefficients_sgd(train, learning_rate, T)
    for k in range(len(test)):
        y_ = predict(test[k], w_)
        y_ = round(y_)
        predictions.append(y_)
    return predictions