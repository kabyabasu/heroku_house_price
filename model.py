import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import pickle
X_data,y_data = load_boston(return_X_y= True)
X = pd.DataFrame(X_data)
X.columns =  ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
y = pd.DataFrame(y_data)
y.columns = ['MEDV']
print([X.loc[0,:]])
lr = LinearRegression()
lr.fit(X,y)
print(lr.score(X,y))

pickle.dump(lr,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([X.loc[0,:]])[0][0])



