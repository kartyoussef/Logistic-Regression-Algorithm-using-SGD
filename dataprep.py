######################################################################
#                       Data Preparation
#                   Author :  Youssef Kartit
######################################################################

##### Libraries :
import csv
import random as rd
from copy import copy
#### Load the CSV file (dataset)
def csv_2_list(filename, delm):
    ''' This funcion converts a CSV file delimited by <delm> to a Python list''' 
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter = delm)
        dataset = list(csv_reader)
    return dataset

#### Convert a column values to float values
def str_2_float(dataset, column):
    ''' This function converts a <column> values of a <dataset>  into float values'''
    for i in range(len(dataset)):
        dataset[i][column] = float(dataset[i][column])
    return dataset

#### Normalisation des données
def minmax_dataset(dataset):
    ''' '''
    min_max = []
    for j in range(len(dataset[0])):
        column_values = [dataset[i][j] for i in range(len(dataset))]
        min_value, max_value = min(column_values), max(column_values)
        min_max.append([min_value, max_value])
    return min_max

def normalize_data(dataset):
    ''' this'''
    min_max = minmax_dataset(dataset)
    for i in range(len(dataset)):
        for j  in range(len(dataset[0])):
            dataset[i][j] = (dataset[i][j] - min_max[j][0]) / (min_max[j][1] - min_max[j][0])

#### K-folds Cross Validation
def crossvalidation_split(dataset, K):
    ''' This function '''
    dataset_copy = list(dataset)
    dataset_cv = []
    fold_size = int(len(dataset) / K)
    for k in range(K):
        fold = []
        while len(fold) < fold_size:
            index = rd.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_cv.append(fold)
    return dataset_cv

#### Adding Polynomial features to the dataset:
###### 2nd degree polynomial features
def polynomial_2(dataset):
    dataset_copy = [list(row) for row in dataset]
    dataset_ = []
    for p in range(len(dataset_copy)):
        a = dataset_copy[p].pop(-1)
        for i in range(len(dataset_copy[p])):
            for j in range(i+1):
                dataset_copy[p].append(dataset_copy[p][i] * dataset_copy[p][j])
        dataset_copy[p].append(a)
        dataset_.append(dataset_copy[p])
    return dataset_

###### 3rd degree polynomial features
def polynomial_3(dataset):
    dataset_copy = [list(row) for row in dataset]
    dataset_ = []
    for row in dataset_copy:
        a = row.pop(-1)
        for i in range(len(row)):
            for j in range(i+1):
                row.append(row[i] * row[j])
                for k in range(j+1):
                    row.append(row[i] * row[j] * row[k])
        row.append(a)
        dataset_.append(row)
    return dataset_