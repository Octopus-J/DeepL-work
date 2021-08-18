import numpy as np
import pandas as pd
import torch

def dataThreat():
    data_train=pd.read_csv('./data/train.csv')
    data_test=pd.read_csv('./data/test.csv')

    train_labels=data_train.iloc[:,-1]

    all_features = pd.concat((data_train.iloc[:, 1:-1], data_test.iloc[:, 1:]))    # remove the first column

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # find the index of columns that are numbers
    all_features[numeric_features]=all_features[numeric_features].apply(lambda x:(x-x.mean())/x.std())  # normalization
    all_features[numeric_features]=all_features[numeric_features].fillna(0)        # replace NA to 0
    all_features=pd.get_dummies(all_features,dummy_na=True)                        # one-hot encoding the text column, consider the NA is meanful, the number of features will increase

    train_features=torch.tensor(all_features[:data_train.shape[0]].values,dtype=torch.float32)    # get the values of train features
    test_features=torch.tensor(all_features[data_train.shape[0]:].values,dtype=torch.float32)
    train_labels=torch.tensor(train_labels.values.reshape(-1,1),dtype=torch.float32)           # get the labels (price)

    return train_features,test_features,train_labels,data_test