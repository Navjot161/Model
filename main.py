import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyexpat import model
from sklearn import linear_model
from numpy import random
import pickle
from sklearn.externals import joblib

df = pd.read_csv("income.csv")
# print(df)

plt.scatter(df.year,df.income)
# plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['year']],df['income'])

prediction = reg.predict([[2020]])
print(prediction)

#
# joblib.dump(model,'model_joblib')
# # ['model_joblib']



