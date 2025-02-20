import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()

# ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']
# print(len(iris.data))
# print(iris.feature_names)

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

X = df.drop(['target','flower_name'],axis=1)
y = df.target

# print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# print(len(X_test))
# print(len(X_train))

model = RandomForestClassifier()
model.fit(X_train,y_train)

print(model.score(X_test,y_test))
print(model.predict(X_test))