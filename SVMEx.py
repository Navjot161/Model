import pandas as pd
from pyexpat import features
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
# print(iris.feature_names)

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
# dj = df[df['target'] == 1]
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:150]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'])
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'])
# plt.show()
plt.xlabel('petal Length')
plt.ylabel('petal Width')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'])
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'])
# plt.show()

X = df.drop(['target','flower_name'],axis="columns")
# print(X)
y = df.target
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#
# print(len(X_test))
#
# print(len(X_train))

model = SVC()
# model_linear_kernal = SVC(kernel='linear')
# model_gamma = SVC(gamma=10)
model.fit(X_train,y_train)
# model_linear_kernal.fit(X_train,y_train)
# model_gamma.fit(X_train,y_train)
print(model.score(X_test,y_test))
# print(model.predict([[4.8,3.4,1.9,0.3]]))
# print(X_test)