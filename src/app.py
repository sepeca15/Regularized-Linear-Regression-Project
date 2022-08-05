import numpy as numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

url = 'https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv'
df = pd.read_csv(url)
target = 'ICU Beds_x'
X = df.drop(columns=[target])
y = df[target]
X = X.select_dtypes(exclude=['object'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)
pipeline = make_pipeline(StandardScaler(), Lasso(alpha=3))
pipeline.fit(X_train, y_train)
lista_coeficientes = pipeline[1].coef_
lista_indices = []
for i in range(len(lista_coeficientes)):
    if lista_coeficientes[i]!=0:
        lista_indices.append(i)
lista_columnas = list(df.columns[lista_indices])
X_reg = X[lista_columnas]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y, test_size=0.3, random_state=37)
scaler = StandardScaler()
scaler.fit(X_train_reg)
X_train_reg = scaler.transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
filename = '/workspace/Regularized-Linear-Regression-Project/models/RLR.pickle'
pickle.dump(lin_reg, open(filename, 'wb'))