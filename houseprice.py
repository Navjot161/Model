import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import math

df = pd.read_csv("houseprice.csv")
# print(df)
bedroomsMedian = math.floor(df.bedrooms.median())
# print(bedroomsMedian)

df.bedrooms = df.bedrooms.fillna(bedroomsMedian)
# print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
#
prediction = reg.predict([[3500,3,40]])
print(prediction)
# print(reg.predict([[2500,4,5]]))

# print(reg.coef_)
# print(reg.intercept_)