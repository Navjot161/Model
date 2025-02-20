import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("carprices.csv")
# print(df)

dummies = pd.get_dummies(df['Car Model']).astype(int)
# print(dummies)

merged = pd.concat([df,dummies],axis=1)
# print(merged)

final = merged.drop(['Mercedez Benz C class','Car Model'],axis=1)
# print(final)

model = LinearRegression()

X = final.drop(['Sell Price($)'],axis=1)
Y = final['Sell Price($)']

# print(Y)
model.fit(X,Y)
print(model.score(X,Y))
print(model.predict([[45000,4,0,0]]))