# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 00:18:15 2021

@author: Fokhrul
""" 

"""K-fold cross validation """

# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection
if __name__ == "__main__":
    # Training data is in a CSV file called train.csv
    df = pd.read_csv("dataset/winequality-red.csv")
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)
    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    # save the new csv with kfold column
    df.to_csv("dataset/winequality-red_train_folds.csv", index=False)
    

""" stratified k-fold """

# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection
if __name__ == "__main__":
    # Training data is in a CSV file called train.csv
    df = pd.read_csv("dataset/winequality-red.csv")
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # fetch targets
    y = df.quality.values
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold
    # save the new csv with kfold column
    df.to_csv("dataset/winequality-red_train_stratified_kfolds.csv", index=False)
    
# see distribution of dataset
import seaborn as sns
b = sns.countplot(x='quality', data=df)
b.set_xlabel("quality", fontsize=20)
b.set_ylabel("count", fontsize=20)

# hold-out based validation