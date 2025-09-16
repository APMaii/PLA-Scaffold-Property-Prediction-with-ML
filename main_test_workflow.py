#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baraye shahrivar 2025


"""

#====================================

'''         Import Libraries      '''

#====================================


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



#====================================

'''          Import data          '''

#====================================

#f_path='/Users/apm/Desktop/MAIN/Profs/Shadab Bagheri/3d printing-First phase/Data/data2003new.xlsx'
f_path='/Users/apm/Desktop/MAIN/Profs/Shadab Bagheri/3d printing-First phase/Shahrivar2025/2003D FRESULT.xlsx'
data=pd.read_excel(f_path)


data.drop(columns=data.columns[0],inplace=True)
data.drop(columns=data.columns[0],inplace=True)
data.drop(columns=data.columns[0],inplace=True)


data.drop(index=0,inplace=True)
data.drop(index=1,inplace=True)
data.drop(index=2,inplace=True)





data.columns=data.iloc[0]

data.drop(index=3,inplace=True)


data.drop(index=10,inplace=True)


data.drop(columns=['Pore size (mm2)','SHAPE FACTOR','Specific Modulus(kNm/kg)','Specific Strength(MPa/g)'],inplace=True)


#data.columns


'''

['Nozzle size', 'Infill Percentage', 'Raster Angle','Pore size (mm2)','porosity(%)', 'Modulus(MPa)', 'Strength(MPa)','Specific Strength(kNm/kg)']
'''

data.columns=['Nozzle size', 'Infill Percentage', 'Raster Angle','Pore size (mm2)','porosity(%)', 'Modulus(MPa)', 'Strength(MPa)','Specific Strength(kNm/kg)']


data.reset_index(inplace=True,drop=True)




main_data=data



x=pd.concat([data['Nozzle size'],data['Infill Percentage'],data['Raster Angle']],axis=1)


y1=data['Pore size (mm2)']
y2=data['porosity(%)']
y3=data['Modulus(MPa)']
y4=data['Strength(MPa)']
y5=data['Specific Strength(kNm/kg)']





#-------3d plot------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Extract as numpy arrays and ensure numeric
x = pd.to_numeric(main_data['Nozzle size'], errors='coerce').to_numpy()
y = pd.to_numeric(main_data['Infill Percentage'], errors='coerce').to_numpy()
z = pd.to_numeric(main_data['Raster Angle'], errors='coerce').to_numpy()


# Create plot
fig = plt.figure(figsize=(16,6))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, s=50)

ax.set_xlabel('Nozzle size', fontsize=10, labelpad=4)
ax.set_ylabel('Infill Percentage', fontsize=10, labelpad=4)
ax.set_zlabel('Raster Angle', fontsize=10, labelpad=4)

ax.tick_params(axis='both', which='major', labelsize=6)
ax.tick_params(axis='z', which='major', labelsize=6)

plt.show()



#------------Correlation-------------
pearson_corr = data.corr(method='pearson')
spearman_corr = data.corr(method='spearman')
kendall_corr = data.corr(method='kendall')

# Define a function to plot heatmap
def plot_corr_map(corr_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Plot each correlation heatmap
plot_corr_map(pearson_corr, "Pearson Correlation")
plot_corr_map(spearman_corr, "Spearman Correlation")
plot_corr_map(kendall_corr, "Kendall Correlation")





# Correlation methods
correlation_methods = {
    "Pearson Correlation": "pearson",
    "Spearman Correlation": "spearman",
    "Kendall Correlation": "kendall"
}

def plot_corr_map(corr_matrix, title):
    """Plot a heatmap for the given correlation matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        annot_kws={"size": 8},  # smaller numbers
        cmap="coolwarm",
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")  # rotate x-axis labels
    plt.yticks(rotation=0)               # keep y-axis labels horizontal
    plt.tight_layout()
    plt.show()

# Generate and plot heatmaps
for title, method in correlation_methods.items():
    corr_matrix = data.corr(method=method)
    plot_corr_map(corr_matrix, title)




# === Pairplot ===
sns.pairplot(data)
plt.suptitle("Pairplot", y=1.02)
plt.tight_layout()
plt.show()










#====================================

'''            ML Models          '''

#====================================
x=np.array(x).astype(float)
y1=np.array(y1).astype(float)
y2=np.array(y2).astype(float)
y3=np.array(y3).astype(float)
y4=np.array(y4).astype(float)
y5=np.array(y5).astype(float)





def exponential_transform(X):
    return np.exp(X)

def log_transform(X):
    return np.log(X)


def lr_regressor(x,yy):
    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    #fold2=KFold(n_splits=5,shuffle=True,random_state=42)
    #fold1=KFold(n_splits=5,shuffle=True,random_state=10)
    #fold2=KFold(n_splits=4,shuffle=True,random_state=10) 
    model = LinearRegression()
    pipe_model = Pipeline([('poly',PolynomialFeatures()), ("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())
    myparams=[{'regressor__poly':[None],
              'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)],
              'transformer': [None,StandardScaler(),MinMaxScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)]},
              {'regressor__poly__degree':[2,3,4,5],
              'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)],
              'transformer': [None,StandardScaler(),MinMaxScaler(),PowerTransformer(),FunctionTransformer(exponential_transform),FunctionTransformer(log_transform)]}]
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    #return score

#-------------Linear Regression-----------------
yy=[y1,y2,y3,y4,y5]

score_list_lr=[]
for i in yy:
    score=lr_regressor(x,i)
    score_list_lr.append(score)

print(score_list_lr)

'''
[-0.15865244240572018, -0.04439567600585234, -0.3206637277058036, -0.20329634505272154, -0.22680893165457205]

'''
    
    
    
    
    
    
    
def knn_regressor(x,yy):
    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)
    #fold1=KFold(n_splits=5,shuffle=True,random_state=10)
    #fold2=KFold(n_splits=4,shuffle=True,random_state=10)
    model = KNeighborsRegressor()
    pipe_model = Pipeline([('poly',PolynomialFeatures()), ("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())
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
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    #return score


#-------------------KNN-------------------------
yy=[y1,y2,y3,y4,y5]
score_list_knn=[]
for i in yy:
    score=knn_regressor(x,i)
    score_list_knn.append(score)
print(score_list_knn)

'''
[-0.2689551615863644, -0.08553110452739221, -0.31812573672387273, -0.36479988407718356, -0.25480386593636273]
'''







def dt_regressor(x,yy):
    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)
    #fold1=KFold(n_splits=5,shuffle=True,random_state=10)
    #fold2=KFold(n_splits=4,shuffle=True,random_state=10)
    model = DecisionTreeRegressor()
    pipe_model = Pipeline([('poly',PolynomialFeatures()), ("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())
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
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    #return score
#---------------Decision Tree-----------------
yy=[y1,y2,y3,y4,y5]
score_list_dt=[]
for i in yy:
    score=dt_regressor(x,i)
    score_list_dt.append(score)
    
print(score_list_dt)

'''
[-0.31980090201534256, -0.04809185813352504, -0.4265923138002537, -0.27699207553935273, -0.2173962793113666]

'''







def rf_regressor(x,yy):
    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)
    #fold1=KFold(n_splits=5,shuffle=True,random_state=10)
    #fold2=KFold(n_splits=4,shuffle=True,random_state=10)
    model = RandomForestRegressor(random_state=0)
    pipe_model = Pipeline([('poly',PolynomialFeatures()), ("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())
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


    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    #return score
#---------------Random Forest-----------------
yy=[y1,y2,y3,y4,y5]
score_list_rf=[]
for i in yy:
    score=rf_regressor(x,i)
    score_list_rf.append(score)
print(score_list_rf)

'''
[-0.4359752604799489, -0.051724123739360466, -0.4567754715075834, -0.36258020550714853, -0.2959438304951553]
'''






#=============================================
#=============================================
#=============================================

'''       Support vector machine         '''

#=============================================
#=============================================
#=============================================



def svr_tuning(yy,model,scaler,myparams,name_param_1,name_param_2,grid_param_1,grid_param_2):
    pipe_model = Pipeline([("scaler",scaler), ("model", model)])
    
    fold3=KFold(n_splits=6,shuffle=True,random_state=42)

    gs = GridSearchCV(pipe_model, param_grid=myparams, cv=fold3, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    cv_results=gs.cv_results_ 
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
    _, ax = plt.subplots(1,1)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()


    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
    _, ax = plt.subplots(1,1)
    for idx, val in enumerate(grid_param_1):
        ax.plot(grid_param_2, scores_mean[:,idx], '-o', label= name_param_1 + ': ' + str(val))
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_2, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()
    
    
    
    
#------------------------------------------------
'''                    y1                   '''
#------------------------------------------------

def svr_regressor(yy,model,myparams):

    pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_
    
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    #return score


model=SVR(kernel='rbf')
model=SVR(kernel='linear')
myparams={'regressor__scaler': [PowerTransformer(),Normalizer()],
    'regressor__model__C': [0.000001,0.001,0.01,0.1,0.7,0.8,0.9,1,50,100,200,300,400,500],
          'regressor__model__epsilon':[0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,1],
          'transformer': [None,StandardScaler(),PowerTransformer(),Normalizer()]}

score=svr_regressor(y1,model,myparams)
print(score)

'''
-0.12362373115708832

'''
    

model=SVR(kernel='rbf')
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
score=svr_regressor(y1,model,myparams)
print(score)


'''
-0.08023138002541304
'''
        
    

#------------------------------------------------
'''                    y2                   '''
#------------------------------------------------


model=SVR()
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

score=svr_regressor(y2,model,myparams)
print(score)


'''
-0.045244128613262456

'''
    
    


    
#------------------------------------------------
'''                    y3                   '''
#------------------------------------------------

model=SVR()

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

score=svr_regressor(y3,model,myparams)
print(score)

'''
-0.2051873119367493

'''






#------------------------------------------------
'''                    y4                  '''
#------------------------------------------------

model = SVR()

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

score = svr_regressor(y4, model, myparams)

print(score)



'''
-0.19484785619903341

'''

    
    

#------------------------------------------------
'''                    y5                  '''
#------------------------------------------------
    
model = SVR()

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

score = svr_regressor(y5, model, myparams)

print(score)



'''

-0.19213152308566142

'''







#----also
def y1_svr_regressor():
    
    yy=y1
    model = SVR(kernel='linear')
    pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={'regressor__scaler': [StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)],
        'regressor__model__C': [0.1,1,70,80,100],
              'regressor__model__epsilon':[0.01,0.1,0.2,0.3],
              'regressor__model__cache_size':[10,100,200,500],
              'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}
    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    gs.fit(x,yy)
    return gs.best_score_

def y2_svr_regressor():
    yy=y2
    model = SVR()
    pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={'regressor__scaler': [StandardScaler()],
               'regressor__model__kernel':['linear'],
        'regressor__model__C': [1,10],
              'regressor__model__epsilon':[0.001,0.01,0.1],
              'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}
    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    gs.fit(x,yy)
    return gs.best_score_


def y3_svr_regressor():
    yy=y3
    model = SVR()
    pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={'regressor__scaler': [StandardScaler(),MinMaxScaler()],
               'regressor__model__kernel':['poly'],
               'regressor__model__degree':[2,3,4],
        'regressor__model__C': [0.001,0.01,30,40,50,60],
              'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}
              
    #,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)



    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    gs.fit(x,yy)
    return gs.best_score_
    

def y4_svr_regressor():
    #yy=y5
    yy=y4
    model = SVR()
    pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={'regressor__scaler': [StandardScaler(),MinMaxScaler()],
               'regressor__model__kernel':['linear'],
        'regressor__model__C': [200,201,202,203,204,205,210,220,230],
              'regressor__model__epsilon':[1,1.8,1.9,2,2.1,2.2,2],
              'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}



    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    gs.fit(x,yy)
    return gs.best_score_


def y5_svr_regressor():
    yy=y5
    model = SVR()
    pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={'regressor__scaler': [StandardScaler(),MinMaxScaler(),PowerTransformer()],
               'regressor__model__kernel':['linear'],
        'regressor__model__C': [1,2,3,4,5,5.5,6,6.5,7,8,9],
              'regressor__model__epsilon':[0.001,0.01,0.1],
              'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}

    fold1=KFold(n_splits=6,shuffle=True,random_state=41)
    fold2=KFold(n_splits=5,shuffle=True,random_state=41)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    gs.fit(x,yy)
    return gs.best_score_

funcs=[y1_svr_regressor(),y2_svr_regressor(),y3_svr_regressor(),y4_svr_regressor(),y5_svr_regressor()]

score_list_svr=[]
for i in funcs:
    score=i
    score_list_svr.append(score)
    
    
    
    
print(score_list_svr)

'''
[-0.14212953411808635, -0.0506041357429512, -0.3464133519193117, -0.32819313197466354, -0.20201558285519816]

'''








#=============================================
#=============================================
#=============================================

'''        Multi Layer Perceptron          '''

#=============================================
#=============================================
#=============================================


def mlp_tuning(yy,model,scaler,myparams,name_param_1,name_param_2,grid_param_1,grid_param_2):
    pipe_model = Pipeline([("scaler",scaler), ("model", model)])
    
    fold3=KFold(n_splits=6,shuffle=True,random_state=42)
    
    gs = GridSearchCV(pipe_model, param_grid=myparams, cv=fold3, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    cv_results=gs.cv_results_
    
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
    _, ax = plt.subplots(1,1)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()


    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
    _, ax = plt.subplots(1,1)
    for idx, val in enumerate(grid_param_1):
        ax.plot(grid_param_2, scores_mean[:,idx], '-o', label= name_param_1 + ': ' + str(val))
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_2, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()
    
    




    
    
#------------------------------------------------
'''                    y1                   '''
#------------------------------------------------
    
    
#------------------------------------------------
'''                    y2                   '''
#------------------------------------------------
    
    
    
    
    
  
    
#------------------------------------------------
'''                    y3                   '''
#------------------------------------------------





#------------------------------------------------
'''                    y4                  '''
#------------------------------------------------
#---------------------------------
def mlp1_regressor_(yy,model,myparams):
    pipe_model = Pipeline([("scaler", PowerTransformer()),("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=None)
    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    score=gs.best_score_
    return score


yy=y4
model=MLPRegressor(random_state=42)
myparams={'regressor__scaler': [MinMaxScaler(),PowerTransformer()],
           'regressor__model__solver':['adam'],
    'regressor__model__hidden_layer_sizes': [(75,),  (80,), 
                                (85,),  (90,),  (95,),  (100,)],
    'regressor__model__activation':['tanh','relu'],
          'transformer': [None,StandardScaler(),PowerTransformer()]}


#score=mlp_regressor(yy, model, myparams)
score=mlp1_regressor_(yy, model, myparams)

print(score)





def y4_mlp_regressor(inputt):
    yy=inputt
    model = MLPRegressor(random_state=42,alpha=0.0001)
    regressor = Pipeline([("scaler", PowerTransformer()), ("model", model)])

    myparams={ 'scaler': [None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
               'model__solver':['sgd','lbfgs'],
        'model__hidden_layer_sizes': [ (2,),  (3,),  (4,),  (5,),  (6,),  
                                    (7,),  (8,),  (9,),  (10,),  (15,),  
                                    (25,),  (30,),  (40,),  (50,),  (60,),
                                    (70,),  (80,), (90,)],
        'model__learning_rate_init':[0.0001,0.001,0.01,0.1,0.3],
        'model__activation':['relu','identity']}



    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    #fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_
    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    #print(score)
    #return score
    



def y5_mlp_regressor(inputt):
    yy=inputt
    model = MLPRegressor(random_state=42)
    pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={ 'regressor__scaler': [MinMaxScaler(),StandardScaler(),PowerTransformer()],
               'regressor__model__solver':['sgd','lbfgs'],
        'regressor__model__hidden_layer_sizes': [ (2,),  (3,),  (4,),  (5,),  (6,),  
                                    (7,),  (8,),  (9,),  (10,),  (15,),  
                                    (25,),  (30,),  (40,),  (50,),  (60,),  (70,), 
                                    (80,), (90,)],
        'regressor__model__alpha':[0.0001,0.001,0.01,0.1,0.3],
        'regressor__model__activation':['relu','logistic'],
              'transformer': [None]}


    #(100,),  (200,),(10,10),(20,20),(50,50),(100,100),(10,10,10)
    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_

    #score=np.mean(cross_val_score(estimator=gs,X=x,y=yy,cv=fold1,scoring='neg_mean_absolute_percentage_error',n_jobs=11))
    #print(score)
    #return score





y4_score=y4_mlp_regressor(y4)
print(y4_score)

'''
-0.16567967125185853
'''



y4_score=y5_mlp_regressor(y4)
print(y4_score)

'''
-0.15806198829911178

'''















#------------------------------------------------
'''                    y5                  '''
#------------------------------------------------



y5_score=y4_mlp_regressor(y5)
print(y5_score)

'''
-0.19457318931470602
'''


y5_score=y5_mlp_regressor(y5)
print(y5_score)
'''
-0.19457318931470602
'''













#------also

    
def y1_mlp_regressor():
    yy=y1
    model = MLPRegressor(random_state=42,max_iter=10000,learning_rate_init=0.1)
    pipe_model = Pipeline([('poly',PolynomialFeatures()),("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={'regressor__poly__degree':[1,2,3],
        'regressor__scaler': [StandardScaler(),MinMaxScaler(),PowerTransformer()],
               'regressor__model__solver':['adam'],
        'regressor__model__hidden_layer_sizes': [(100,100),
                                    (10,10,10),(200,200),(100,200,100),(10,25,10),(100,200),(200,200)],
        'regressor__model__activation':['relu','identity'],
              'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}


    fold1=KFold(n_splits=6,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_
    
    

def y2_mlp_regressor():
    yy=y2
    model = MLPRegressor(random_state=42,max_iter=10000,learning_rate_init=0.1)
    pipe_model = Pipeline([('poly',PolynomialFeatures()),("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={'regressor__poly__degree':[1,2,3],
        'regressor__scaler': [StandardScaler(),MinMaxScaler(),PowerTransformer()],
               'regressor__model__solver':['adam'],
        'regressor__model__hidden_layer_sizes': [(100,100),
                                    (10,10,10),(200,200),(100,200,100),(10,25,10),(100,200),(200,200)],
        'regressor__model__activation':['relu','identity'],
              'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}


    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_
    

    
def y3_mlp_regressor():
    yy=y3
    model = MLPRegressor(random_state=42,max_iter=10000,learning_rate_init=0.1)
    pipe_model = Pipeline([('poly',PolynomialFeatures()),("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={'regressor__poly__degree':[1,2,3],
        'regressor__scaler': [StandardScaler(),MinMaxScaler(),PowerTransformer()],
               'regressor__model__solver':['adam'],
        'regressor__model__hidden_layer_sizes': [(100,100),
                                    (10,10,10),(200,200),(100,200,100),(10,25,10),(100,200),(200,200)],
        'regressor__model__activation':['relu','identity'],
              'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}


    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_



def another_y3_mlp_regressor():
    yy=y3
    model = MLPRegressor(random_state=42,max_iter=10000,solver='lbfgs',activation='relu')
    pipe_model = Pipeline([('poly',PolynomialFeatures()),("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={ 'regressor__scaler': [StandardScaler(),PowerTransformer()],
               'regressor__model__solver':['adam'],
        'regressor__model__hidden_layer_sizes': [ (5,),  (6,), (8,),   (10,),  (15,),  (50,), (80,),  
                                    (85,),  (90,), (100,),  (200,),(10,10),(20,20),(50,50),(100,100),
                                    (10,10,10)],
        'regressor__model__alpha':[0.001,0.0001],
              'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}



    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_


def y4_mlp_regressor():
    yy=y4
    model = MLPRegressor(random_state=42,alpha=0.0001)
    regressor = Pipeline([("scaler", PowerTransformer()), ("model", model)])

    myparams={ 'scaler': [None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
               'model__solver':['sgd','lbfgs'],
        'model__hidden_layer_sizes': [ (2,),  (3,),  (4,),  (5,),  (6,),  
                                    (7,),  (8,),  (9,),  (10,),  (15,),  
                                    (25,),  (30,),  (40,),  (50,),  (60,),  (70,),  (80,), (90,)],
        'model__learning_rate_init':[0.0001,0.001],
        'model__activation':['relu','identity']}



    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_
    

def y5_mlp_regressor():
    yy=y5
    model = MLPRegressor(random_state=42)
    pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

    myparams={ 'regressor__scaler': [MinMaxScaler(),StandardScaler(),PowerTransformer()],
               'regressor__model__solver':['sgd','lbfgs'],
        'regressor__model__hidden_layer_sizes': [ (2,),  (3,),  (4,),  (5,),  (6,),  
                                    (7,),  (8,),  (9,),  (10,),  (15,),  
                                    (25,),  (30,),  (40,),  (50,),  (60,),  (70,),  (80,), (90,)],
        'regressor__model__alpha':[0.0001,0.1],
        'regressor__model__activation':['relu','logistic'],
              'transformer': [None]}


    #(100,),  (200,),(10,10),(20,20),(50,50),(100,100),(10,10,10)
    fold1=KFold(n_splits=6,shuffle=True,random_state=42)
    fold2=KFold(n_splits=5,shuffle=True,random_state=42)

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,yy)
    return gs.best_score_





funcs=[y1_mlp_regressor(),y2_mlp_regressor(),y3_mlp_regressor(),another_y3_mlp_regressor(),y4_mlp_regressor(),y5_mlp_regressor()]

score_list_svr=[]
for i in funcs:
    score=i
    score_list_svr.append(score)


print(score_list_svr)


'''
[-0.1286688363363733, -0.043121911238996065, -0.24171096675685974, -0.22110827071790382, 
 -0.16567967125185853, -0.19457318931470602]

'''








#=============================================
#=============================================
#=============================================
#=============================================

'''
SCORES-----------------

'''
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================








#=============================================
#=============================================
#=============================================
#=============================================

'''

RASMMM

'''
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================
#=============================================



def generate_bumpy():
    a1_values = np.arange(200,600, 10)
    a2_values = np.arange(10,100, 1)
    a3_values = np.array([60,90,120])

    a3, a2, a1 = np.meshgrid(a3_values, a2_values, a1_values, indexing='ij')

    result = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))

    return result

input_space = generate_bumpy()



x__=pd.concat([main_data['Nozzle size'],main_data['Infill Percentage'],main_data['Raster Angle']],axis=1)
x_=np.array(x__).astype(float)






#------------------------------- best y1---------------------------------

yyyy='y1'
yy=y1
mm='SVR'

yyy = 'Pore size (mm2)'



model=SVR(kernel='rbf')
pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

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

fold1=KFold(n_splits=6,shuffle=True,random_state=42)

gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
gs.fit(x_,yy)
l_yp=gs.predict(input_space)



#------------------------------- best y2---------------------------------

yyyy='y2'
yy=y2
mm='MLP'

yyy = 'Porosity(%)'



model = MLPRegressor(random_state=42,max_iter=10000,learning_rate_init=0.1)
pipe_model = Pipeline([('poly',PolynomialFeatures()),("scaler", PowerTransformer()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

myparams={'regressor__poly__degree':[1,2,3],
    'regressor__scaler': [StandardScaler(),MinMaxScaler(),PowerTransformer()],
           'regressor__model__solver':['adam'],
    'regressor__model__hidden_layer_sizes': [(100,100),
                                (10,10,10),(200,200),(100,200,100),(10,25,10),(100,200),(200,200)],
    'regressor__model__activation':['relu','identity'],
          'transformer': [None,StandardScaler(),PowerTransformer(),FunctionTransformer(log_transform)]}


fold1=KFold(n_splits=6,shuffle=True,random_state=42)

gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
gs.fit(x_,yy)
l_yp=gs.predict(input_space)






#------------------------------- best y3---------------------------------

yyyy='y3'
yy=y3
mm='SVR'
yyy='Modulus(MPa)'


model=SVR()




model=SVR()
pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=RobustScaler())


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


fold1=KFold(n_splits=6,shuffle=True,random_state=42)

gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
gs.fit(x_,yy)
l_yp=gs.predict(input_space)


#------------------------------- best y4---------------------------------

yyyy='y4'
yy=y4
mm='MLP'
yyy='Strength(MPa)'



model = MLPRegressor(random_state=42,alpha=0.0001)
regressor = Pipeline([("scaler", PowerTransformer()), ("model", model)])

myparams={ 'scaler': [None,MinMaxScaler(),StandardScaler(),PowerTransformer()],
           'model__solver':['sgd','lbfgs'],
    'model__hidden_layer_sizes': [ (2,),  (3,),  (4,),  (5,),  (6,),  
                                (7,),  (8,),  (9,),  (10,),  (15,),  
                                (25,),  (30,),  (40,),  (50,),  (60,),  (70,),  (80,), (90,)],
    'model__learning_rate_init':[0.0001,0.001],
    'model__activation':['relu','identity']}
fold1=KFold(n_splits=6,shuffle=True,random_state=42)

gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
gs.fit(x_,yy)
l_yp=gs.predict(input_space)



#------------------------------- best y5---------------------------------

yyyy='y5'
yy=y5
mm='SVR'
yyy='Specific Strength(kNm/kg)'





model = SVR()
pipe_model = Pipeline([("scaler", PowerTransformer()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=RobustScaler())

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

fold1=KFold(n_splits=6,shuffle=True,random_state=42)

gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
gs.fit(x_,yy)
l_yp=gs.predict(input_space)














yp=pd.Series(l_yp)
yp.name='p'
pred_data=pd.concat([pd.DataFrame(input_space),pd.Series(yp)],axis=1)


pred_data.info()




sns.boxplot(pred_data['p'])
plt.show()


plt.hist(pred_data['p'])
plt.show()




'''
Q1 = data['p'].quantile(0.25)  # First quartile
Q3 = data['p'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1                  # Interquartile range

# Define the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


limit=upper_bound
'''


limit=70
count=0
for i in range(0,len(pred_data)):
    if pred_data['p'][i]<0:
        if i!=0:
            a=pred_data['p'][i-1]
            pred_data['p'][i]=a
            count=count+1
        
    if pred_data['p'][i]>limit :
        if i!=0:
            a=pred_data['p'][i-1]
            pred_data['p'][i]=a
            count=count+1
        
        
sns.boxplot(pred_data['p'])
plt.show()
print(count)



bottom_value = 0
top_value = 70














import matplotlib.pyplot as plt
import numpy as np

# Example variable names
d = pred_data['p']   # your values

angles = ['60', '90', '120']

# High resolution
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

# Create figure
fig = plt.figure(figsize=(12, 5), dpi=600)
#fig.suptitle(yyy, fontsize=12, fontweight='bold', y=0.97)  # smaller main title

gs = fig.add_gridspec(1, 3, wspace=0.35)

# First scatter to reuse for colorbar
first_scatter = None
norm = plt.Normalize(bottom_value, top_value)

for i, angle in enumerate(angles):
    if angle == '60':
        xx1 = input_space[0:3599, 0]
        xx2 = input_space[0:3599, 1]
        yy3 = d[0:3599]
    elif angle == '90':
        xx1 = input_space[3600:7199, 0]
        xx2 = input_space[3600:7199, 1]
        yy3 = d[3600:7199]
    elif angle == '120':
        xx1 = input_space[7200:10799, 0]
        xx2 = input_space[7200:10799, 1]
        yy3 = d[7200:10799]

    # Subplot 3D
    ax = fig.add_subplot(gs[0, i], projection='3d')

    scatter = ax.scatter(
        xx1, xx2, yy3,
        c=yy3,
        cmap='viridis',
        marker='o',
        s=4,          # tiny points
        alpha=0.7,
        edgecolors='none',
        norm=norm
    )

    if i == 0:
        first_scatter = scatter  # save reference for shared colorbar

    # Axis limits and labels
    ax.set_zlim(bottom_value, top_value)
    ax.set_xlabel('Nozzle Size', fontsize=10, labelpad=4)
    ax.set_ylabel('Infill %', fontsize=10, labelpad=4)
    ax.set_zlabel(f'{yyy}', fontsize=10, labelpad=4)

    # Subplot title
    ax.set_title(f'Raster Angle {angle}°', fontsize=9, fontweight='bold', pad=6)

    # Smaller tick labels
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='z', which='major', labelsize=6)

    # Light grid
    ax.grid(True, linestyle='--', alpha=0.25)

# === Global colorbar (not inside the 3 subplots) ===
cbar_ax = fig.add_axes([0.98, 0.2, 0.015, 0.6])  # pushed outside grid
cbar = fig.colorbar(first_scatter, cax=cbar_ax)
cbar.set_label(f'{yyy}', fontsize=8, fontweight='bold')
cbar.ax.tick_params(labelsize=6)

# Save high-quality image
plt.savefig(f'multi_angle_3d_plot_.png', dpi=600, bbox_inches='tight')
plt.show()


#======================================

















