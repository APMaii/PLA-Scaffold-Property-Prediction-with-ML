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

data.columns=['Nozzle size', 'Infill Percentage', 'Raster Angle','Pore size (mm2)','porosity(%)', 'Modulus(MPa)', 'Strength(MPa)','Specific Strength(kNm/kg)']

data.reset_index(inplace=True,drop=True)


main_data=data

x=pd.concat([data['Nozzle size'],data['Infill Percentage'],data['Raster Angle']],axis=1)

y1=data['Pore size (mm2)']
y2=data['porosity(%)']
y3=data['Modulus(MPa)']
y4=data['Strength(MPa)']
y5=data['Specific Strength(kNm/kg)']





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





#=============================================
#=============================================
#=============================================

'''        Multi Layer Perceptron          '''

#=============================================
#=============================================
#=============================================




#------------------------------------------------
'''                    y1                  '''
#------------------------------------------------
    
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
    
    

#------------------------------------------------
'''                    y2                  '''
#------------------------------------------------

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
    


#------------------------------------------------
'''                    y3                  '''
#------------------------------------------------

def y3_mlp_regressor():
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



#------------------------------------------------
'''                    y4                  '''
#------------------------------------------------

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
    



#------------------------------------------------
'''                    y5                  '''
#------------------------------------------------

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





funcs=[y1_mlp_regressor(),y2_mlp_regressor(),y3_mlp_regressor(),y4_mlp_regressor(),y5_mlp_regressor()]

score_list_svr=[]
for i in funcs:
    score=i
    score_list_svr.append(score)


print(score_list_svr)


'''
[-0.1286688363363733, -0.043121911238996065, -0.22110827071790382, 
 -0.16567967125185853, -0.19457318931470602]

'''





