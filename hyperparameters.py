#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:09:08 2025

@author: apm
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import statistics
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.compose import TransformedTargetRegressor
#from sklearn.preprocessing import PowerTransformer
#from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import PowerTransformer

from sklearn.compose import TransformedTargetRegressor as TTR
from sklearn.preprocessing import FunctionTransformer




def exponential_transform(X):
    return np.exp(X)

def log_transform(X):
    return np.log(X)




#=====================================================================
"""                    Y1,Y2,Y3,Y4,Y5                       """
#=====================================================================

'''


Y1 =  Pore size (mm2)

Y2 = Porosity(%)

Y3 = Modulus(MPa)

Y4 = Strength(MPa)

Y5 = Specific Strength(kNm/kg)


'''









#=====================================================================
"""                    LR HYPERPARAMETERS                        """
#=====================================================================
myparams=[{'regressor__poly':[None],
          'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)],
          'transformer': [None,StandardScaler(),MinMaxScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)]},
          {'regressor__poly__degree':[2,3,4,5],
          'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)],
          'transformer': [None,StandardScaler(),MinMaxScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)]}]



#=====================================================================
"""                    KNN HYPERPARAMETERS                        """
#=====================================================================

myparams=[{'regressor__poly':[None],
          'regressor__scaler': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)],
          'regressor__model__n_neighbors':[2,4,6,8],
          'regressor__model__metric':['minkowski','euclidean','manhattan'],
          'transformer': [None,StandardScaler(),MinMaxScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)]},
          {'regressor__poly__degree':[2,3],
          'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)],
          'regressor__model__n_neighbors':[2,4,6,8],
          'regressor__model__metric':['minkowski','euclidean','manhattan'],
          'transformer': [None,StandardScaler(),MinMaxScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)]}]


#=====================================================================
"""                    DT HYPERPARAMETERS                        """
#=====================================================================
myparams=[{'regressor__poly':[None],
          'regressor__scaler': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)],
          'regressor__model__max_depth':[2,3,4,5,7],
          'regressor__model__min_samples_split':[2,3,4,5],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]},
          {'regressor__poly__degree':[2],
           'regressor__model__max_depth':[2,3,4,5,7],
           'regressor__model__min_samples_split':[2,3,4,5],
          'regressor__scaler': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}]


#=====================================================================
"""                    RF HYPERPARAMETERS                        """
#=====================================================================
myparams=[{'regressor__poly':[None],
          'regressor__model__n_estimators':[10,20,50,100],
          'regressor__model__max_depth':[2,3,4,5,6],
          'regressor__scaler': [None,StandardScaler(),FunctionTransformer(log_transform)],
          'transformer': [None,StandardScaler(),FunctionTransformer(log_transform)]},
          {'regressor__poly__degree':[2],
          'regressor__model__n_estimators':[10,20,50,100],
          'regressor__model__max_depth':[2,3,4,5,10],
          'regressor__scaler': [None,StandardScaler(),FunctionTransformer(log_transform)],
          'transformer': [None,StandardScaler(),FunctionTransformer(log_transform)]}]


#=====================================================================
"""                    SVR HYPERPARAMETERS                        """
#=====================================================================

#----------------------y1----------------
#kernel='rbf'
myparams = {
    # Feature transformer before model
    'regressor__scaler': [PowerTransformer(), StandardScaler()],
    
    # SVR regularization strength
    'regressor__model__C': [0.1, 1, 10, 50, 100, 200, 500],
    
    # SVR epsilon-insensitive zone
    'regressor__model__epsilon': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
    
    # SVR kernel width (only used for 'rbf', optional if kernel is fixed)
    'regressor__model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    
    # Transformer for the target variable
    'transformer': [None, StandardScaler(), PowerTransformer()]
}

#----------------------y2----------------

myparams=[{'regressor__model__kernel':['poly'],
           'regressor__scaler': [StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)],
    'regressor__model__C': [0.7,0.8,0.9,1,50],
          'regressor__model__epsilon':[0.0001,0.001,0.01,0.1],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]} ,
{'regressor__model__kernel':['linear'],
       'regressor__scaler': [StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)],
    'regressor__model__C': [0.7,0.8,0.9,1,50],
          'regressor__model__epsilon':[0.0001,0.001,0.01,0.1],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}]


#----------------------y3----------------
myparams=[{'regressor__model__kernel':['poly'],
           'regressor__scaler': [StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)],
    'regressor__model__C': [0.7,0.8,0.9,1,50],
          'regressor__model__epsilon':[0.0001,0.001,0.01,0.1],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]} ,
{'regressor__model__kernel':['linear'],
       'regressor__scaler': [StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)],
    'regressor__model__C': [0.7,0.8,0.9,1,50],
          'regressor__model__epsilon':[0.0001,0.001,0.01,0.1],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}]


#----------------------y4----------------

myparams = [
    {
        'regressor__model__kernel': ['poly'],
        'regressor__scaler': [StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
        'regressor__model__C': [0.1, 1, 10, 50],
        'regressor__model__epsilon': [0.001, 0.01, 0.1],
        'regressor__model__degree': [2, 3, 4],  # specific to poly
        'regressor__model__coef0': [0.0, 0.5, 1.0],
        'transformer': [None, StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
    },
    {
        'regressor__model__kernel': ['linear'],
        'regressor__scaler': [StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
        'regressor__model__C': [0.1, 1, 10, 50],
        'regressor__model__epsilon': [0.001, 0.01, 0.1],
        'transformer': [None, StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
    },
    {
        'regressor__model__kernel': ['rbf'],
        'regressor__scaler': [StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
        'regressor__model__C': [0.1, 1, 10, 50],
        'regressor__model__epsilon': [0.001, 0.01, 0.1],
        'regressor__model__gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
        'transformer': [None, StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
    },
]


#----------------------y5----------------

myparams = [
    {
        'regressor__model__kernel': ['poly'],
        'regressor__scaler': [StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
        'regressor__model__C': [0.1, 1, 10, 50],
        'regressor__model__epsilon': [0.001, 0.01, 0.1],
        'regressor__model__degree': [2, 3, 4],  # specific to poly
        'regressor__model__coef0': [0.0, 0.5, 1.0],
        'transformer': [None, StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
    },
    {
        'regressor__model__kernel': ['linear'],
        'regressor__scaler': [StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
        'regressor__model__C': [0.1, 1, 10, 50],
        'regressor__model__epsilon': [0.001, 0.01, 0.1],
        'transformer': [None, StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
    },
    {
        'regressor__model__kernel': ['rbf'],
        'regressor__scaler': [StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
        'regressor__model__C': [0.1, 1, 10, 50],
        'regressor__model__epsilon': [0.001, 0.01, 0.1],
        'regressor__model__gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
        'transformer': [None, StandardScaler(), PowerTransformer(), FunctionTransformer(log_transform)],
    },
]










#=====================================================================
"""                    MLP HYPERPARAMETERS                        """
#=====================================================================




#----------------------y1----------------
myparams={'regressor__poly__degree':[1,2,3],
    'regressor__scaler': [StandardScaler(),MinMaxScaler(),PowerTransformer()],
           'regressor__model__solver':['adam'],
    'regressor__model__hidden_layer_sizes': [(100,100),
                                (10,10,10),(200,200),(100,200,100),(10,25,10),(100,200),(200,200)],
    'regressor__model__activation':['relu','identity'],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}


#----------------------y2----------------
myparams={'regressor__poly__degree':[1,2,3],
    'regressor__scaler': [StandardScaler(),MinMaxScaler(),PowerTransformer()],
           'regressor__model__solver':['adam'],
    'regressor__model__hidden_layer_sizes': [(100,100),
                                (10,10,10),(200,200),(100,200,100),(10,25,10),(100,200),(200,200)],
    'regressor__model__activation':['relu','identity'],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}



#----------------------y3----------------


myparams={ 'regressor__scaler': [StandardScaler(),PowerTransformer()],
           'regressor__model__solver':['adam'],
    'regressor__model__hidden_layer_sizes': [ (5,),  (6,), (8,),   (10,),  (15,),  (50,), (80,),  
                                (85,),  (90,), (100,),  (200,),(10,10),(20,20),(50,50),(100,100),
                                (10,10,10)],
    'regressor__model__alpha':[0.001,0.0001],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}





#----------------------y4----------------


myparams={ 'scaler': [None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
           'model__solver':['sgd','lbfgs'],
    'model__hidden_layer_sizes': [ (2,),  (3,),  (4,),  (5,),  (6,),  
                                (7,),  (8,),  (9,),  (10,),  (15,),  
                                (25,),  (30,),  (40,),  (50,),  (60,),  (70,),  (80,), (90,)],
    'model__learning_rate_init':[0.0001,0.001],
    'model__activation':['relu','identity']}




#----------------------y5----------------
myparams={ 'regressor__scaler': [MinMaxScaler(),StandardScaler(),PowerTransformer()],
           'regressor__model__solver':['sgd','lbfgs'],
    'regressor__model__hidden_layer_sizes': [ (2,),  (3,),  (4,),  (5,),  (6,),  
                                (7,),  (8,),  (9,),  (10,),  (15,),  
                                (25,),  (30,),  (40,),  (50,),  (60,),  (70,),  (80,), (90,)],
    'regressor__model__alpha':[0.0001,0.1],
    'regressor__model__activation':['relu','logistic'],
          'transformer': [None]}






































