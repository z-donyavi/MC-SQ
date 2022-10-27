# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:49:22 2022

@author: Zahra
"""

# Model Building functions

import numpy as np
from sklearn import svm, linear_model, ensemble, naive_bayes, discriminant_analysis, model_selection
import lightgbm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict



def getScores(X_train, X_test, Y_train, nclasses, algs):
    
   #check if there is any ensemble method in the algs list 
    Ensembles = ['EnsembleDyS', 'EnsembleEM', 'EnsembleGAC', 'EnsembleGPAC', 'EnsembleFM']
    compare_list = [a == b for a in algs for b in Ensembles]
    
    if any(compare_list):
        models = [linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000),
                      discriminant_analysis.LinearDiscriminantAnalysis(),
                      ensemble.RandomForestClassifier(),
                      svm.SVC(probability=True),
                      lightgbm.LGBMClassifier(),
                      naive_bayes.GaussianNB(),
                      ensemble.GradientBoostingClassifier()]
    #if there is not any ensemble method in the algs list, use only LR as classifier    
    else: 
        models = [linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)]
            
   
    train_scores = np.zeros((len(models), len(X_train), nclasses))
    test_scores = np.zeros((len(models), len(X_test), nclasses))
    Y_cts = np.unique(Y_train, return_counts=True)
    nfolds = min(10, min(Y_cts[1]))
    for i, model in enumerate(models):
       
        if nfolds > 1:
            kfold = model_selection.StratifiedKFold(n_splits=nfolds, random_state=1, shuffle=True)
            # kfold = model_selection.StratifiedKFold(n_splits=nfolds)
            # train_scores[i] = cross_val_predict(model, X_train, Y_train, cv=kfold, n_jobs=-1, method='predict_proba')
            for train, test in kfold.split(X_train, Y_train):
                model.fit(X_train[train], Y_train[train])
                train_scores[i][test] = model.predict_proba(X_train)[test]
       
        model.fit(X_train, Y_train)
        test_scores[i] = model.predict_proba(X_test)
       
        if nfolds < 2:
            train_scores[i] = model.predict_proba(X_train)
            
    return train_scores, test_scores, len(models)

def getCalibratedScores(X_train, X_test, Y_train, nclasses, algs):
    
    #check if there is any ensemble method in the algs list 
    Ensembles = ['EnsembleDyS', 'EnsembleEM', 'EnsembleGAC', 'EnsembleGPAC', 'EnsembleFM']
    compare_list = [a == b for a in algs for b in Ensembles]
     
    if any(compare_list):
        models = [linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000),
                       discriminant_analysis.LinearDiscriminantAnalysis(),
                       ensemble.RandomForestClassifier(),
                       svm.SVC(probability=True),
                       lightgbm.LGBMClassifier(),
                       naive_bayes.GaussianNB(),
                       ensemble.GradientBoostingClassifier()]
     #if there is not any ensemble method in the algs list, use only LR as classifier    
    else: 
        models = [linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)]
       
    train_scores = np.zeros((len(models), len(X_train), nclasses))
    test_scores = np.zeros((len(models), len(X_test), nclasses))
    Y_cts = np.unique(Y_train, return_counts=True)
    nfolds = min(10, min(Y_cts[1]))
    nfolds2 = min(5, nfolds-1)
    for i, model in enumerate(models):
       
        #get training scores with calibration and CV
        if nfolds > 2:
            kfold = model_selection.StratifiedKFold(n_splits=nfolds, random_state=1, shuffle=True)
            # kfold = model_selection.StratifiedKFold(n_splits=nfolds)
            # train_scores[i] = cross_val_predict(model, X_train, Y_train, cv=kfold, n_jobs=-1, method='predict_proba')
            for train, test in kfold.split(X_train, Y_train):
                calibratedModel = CalibratedClassifierCV(model, cv=nfolds2, n_jobs=-1)
                calibratedModel.fit(X_train[train], Y_train[train])
                train_scores[i][test] = calibratedModel.predict_proba(X_train)[test]
                
       #get training scores with CV and without calibration
        if nfolds == 2:
            kfold = model_selection.StratifiedKFold(n_splits=nfolds, random_state=1, shuffle=True)
            # kfold = model_selection.StratifiedKFold(n_splits=nfolds)
            # train_scores[i] = cross_val_predict(model, X_train, Y_train, cv=kfold, n_jobs=-1, method='predict_proba')
            for train, test in kfold.split(X_train, Y_train):
                model.fit(X_train[train], Y_train[train])
                train_scores[i][test] = model.predict_proba(X_train)[test]
       
        #get test scores with calibration and CV
        if nfolds > 2:
            calibratedModel = CalibratedClassifierCV(model, cv=nfolds2, n_jobs=-1)
            calibratedModel.fit(X_train, Y_train)
            test_scores[i] = calibratedModel.predict_proba(X_test)   
        else:
        # get test scores without calibration    
            model.fit(X_train, Y_train)
            test_scores[i] = model.predict_proba(X_test)
        
        #get training scores without CV and calibration
        if nfolds <= 1:
            train_scores[i] = model.predict_proba(X_train)
                
    return train_scores, test_scores, len(models)


def to_str(var):
    if type(var) is list:
        return str(var)[1:-1] # list
    if type(var) is np.ndarray:
        try:
            return str(list(var[0]))[1:-1] # numpy 1D array
        except TypeError:
            return str(list(var))[1:-1] # numpy sequence
    return str(var) # everything else

