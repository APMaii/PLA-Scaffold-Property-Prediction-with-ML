#====================================

'''         Import Libraries      '''




*******vorodie tabe ha bauayd ok she
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








import matplotlib.pyplot as plt
import numpy as np




def plot_prediction():
    
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
    plt.savefig(f'multi_angle_3d_plot_{yyy}.png', dpi=600, bbox_inches='tight')
    plt.show()

    











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









yp=pd.Series(l_yp)
yp.name='p'
pred_data=pd.concat([pd.DataFrame(input_space),pd.Series(yp)],axis=1)


pred_data.info()




sns.boxplot(pred_data['p'])
plt.show()


plt.hist(pred_data['p'])
plt.show()






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




plot_prediction(bottom_value,top_value)






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







yp=pd.Series(l_yp)
yp.name='p'
pred_data=pd.concat([pd.DataFrame(input_space),pd.Series(yp)],axis=1)


pred_data.info()




sns.boxplot(pred_data['p'])
plt.show()


plt.hist(pred_data['p'])
plt.show()




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




plot_prediction(bottom_value,top_value)





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








yp=pd.Series(l_yp)
yp.name='p'
pred_data=pd.concat([pd.DataFrame(input_space),pd.Series(yp)],axis=1)


pred_data.info()




sns.boxplot(pred_data['p'])
plt.show()


plt.hist(pred_data['p'])
plt.show()



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



plot_prediction(bottom_value,top_value)










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







yp=pd.Series(l_yp)
yp.name='p'
pred_data=pd.concat([pd.DataFrame(input_space),pd.Series(yp)],axis=1)


pred_data.info()




sns.boxplot(pred_data['p'])
plt.show()


plt.hist(pred_data['p'])
plt.show()



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




plot_prediction(bottom_value,top_value)










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






plot_prediction(bottom_value,top_value)











