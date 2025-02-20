import pandas as pd
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
# print(dir(load_digits))

# print(digits.data.shape)
# print(digits.target.shape)

df = pd.DataFrame(digits.data,columns=digits.feature_names)

df['target'] = digits['target']
# print(df.head())
# print(df[:50])
# print(len(df))
X = df.drop(['target'],axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42)
# print(len(X_test))
# print(X_test)
# print(len(X_train))
#
model = SVC(kernel='linear')
model.fit(X_train,y_train)
#
print(model.score(X_test,y_test))
# print(model.predict(X_test))
#
# plt.matshow(digits.images[1])
# plt.show()