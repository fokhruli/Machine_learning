# -*- coding: utf-8 -*-

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import config
import os
import argparse


def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    # train dataset where fold != kfold
    # reset the window
    df_train = df[df.kfold != fold].reset_index(drop =True)
    # validation is oposite
    df_valid = df[df.kfold == fold].reset_index(drop =True)
    
    X_train = df_train.drop("label", axis =1).values
    y_train = df_train.label.values
    
    X_valid = df_valid.drop("label", axis =1).values
    y_valid = df_valid.label.values
    
    model = tree.DecisionTreeClassifier()
    
    model.fit(X_train, y_train)
    
    predict = model.predict(X_valid)
    
    accuracy = metrics.accuracy_score(y_valid, predict)
    print(f"Fold =  {fold}, Accuracy = {accuracy}")
    
    # save model 
    print("saving file to " + str(os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")))
    joblib.dump(
        model,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
        )
    
    
if __name__ == "__main__":
    # first initialize argument parser class 
    parser = argparse.ArgumentParser()
    
    # define a argumwnt 
    parser.add_argument("--fold", type = int, help= "put k fold value")
    
    # read the argument from command line
    args = parser.parse_args()
    
    # run the specific line with args
    
    run(args.fold)