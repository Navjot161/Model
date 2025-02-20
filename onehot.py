import pandas as pd
from sklearn import linear_model

from hr import mapping
from main import prediction

df = pd.read_csv("homeprices.csv")
# print(df)

dummies = pd.get_dummies(df.town)

merged = df.concat([df,dummies],axis='columns')
print(merged)
# mapping = {'robinsville':0, 'west windsor':1, 'monroe township':2}
# df['town'] = df['town'].map(mapping)
X = df.drop('price',axis='columns')
y = df.price

model = linear_model.LinearRegression()
model.fit(X,y)
# print(model.predict([[3400,2]]))

# print(model.score(X,y))


