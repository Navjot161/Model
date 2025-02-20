import pandas as pd
from word2number import w2n
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv("titanic.csv")
# print(df.head())

mean_Age = df['Age'].mean()

df['Age'] = df['Age'].fillna(mean_Age)
# print(df['Age'])

first_drop = df.drop(['PassengerId','Cabin','Name','SibSp','Ticket','Parch','Embarked'],axis=1)
# print(droped.head(7))

input = first_drop.drop(['Survived'],axis=1)
target = df.Survived

input.Sex = input.Sex.map({'male':1,'female':2})

X_train, X_test, y_train, y_test = train_test_split(input,target,test_size=0.2,random_state=42)

model = tree.DecisionTreeClassifier()

model.fit(X_train,y_train)
# print(X_test)
# print(model.score(X_train,y_train))
# print(model.predict(X_test))
