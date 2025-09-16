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




#----------- Pairplot-------------------
sns.pairplot(data)
plt.suptitle("Pairplot", y=1.02)
plt.tight_layout()
plt.show()







