import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("HR_comma_sep.csv")

df = data.drop(['promotion_last_5years','Department','Work_accident','left'],axis=1)

mapping = {'low': 0, 'medium': 1, 'high': 2}
df['salary'] = df['salary'].map(mapping)

X = df.drop(['salary'],axis=1)
y = df.salary

model = LogisticRegression()
model.fit(X,y)

print(model.predict([[0.11,0.88,7,272,4]]))